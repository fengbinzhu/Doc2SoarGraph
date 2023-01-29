import copy
import math
import pickle
import re
from copy import deepcopy
from .tools import allennlp as util
import torch
import torch.nn as nn
from torch.nn import functional
from tatqa_utils import *


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Default_FNN(nn.Module):
    def __init__(self, input_size, mid_size, output_size, dropout, activation_fn=None, layer_norm=True):
        super(Default_FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, mid_size)
        if layer_norm:
            self.ln = nn.LayerNorm(mid_size)
        else:
            self.ln = None
        if activation_fn:
            self.afn = activation_fn
        else:
            self.afn = gelu
        self.dropout_fn = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mid_size, output_size)

    def forward(self, input: torch.LongTensor):
        out = self.afn(self.fc1(self.dropout_fn(input)))
        if self.ln:
            out = self.ln(out)
        return self.fc2(out)


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)


class EncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        return pade_outputs, problem_output


class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)
        # B x 1 x H
        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)
        # B x 1 x S
        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x H

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


def masked_cross_entropy(logits, target, length):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    # if loss.item() > 10:
    #     print(losses, target)
    return loss


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand)).cuda()
    else:
        seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def out_expression_list(test, output_lang, num_list, num_stack=None):
    max_index = output_lang.n_words
    res = []
    for i in test:
        if i < max_index - 1:
            idx = str(output_lang.index2word[i])
            if idx[0] == "N":
                if int(idx[1:]) >= len(num_list):
                    return None
                res.append(num_list[int(idx[1:])])
            else:
                res.append(idx)
        elif num_stack:
            pos_list = num_stack.pop()
            c = num_list[pos_list[0]]
            res.append(c)
    return res

def simple_to_number(text):
    num = extract_one_num_from_str(text)
    if num is not None:
        return round(num , 4)
    return None

def compute_prefix_expression(pre_fix: object) -> object:
    st = list()
    operators = ["+", "-", "^", "*", "/", "**", ">"]
    pre_fix = deepcopy(pre_fix)
    pre_fix.reverse()
    for p in pre_fix:
        p = str(p)
        if p not in operators:
            # change 17 Dec
            st.append(simple_to_number(p))
            # pos = re.search("\d+\(", p)
            # if pos:
            #     st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
            # elif p[-1] == "%":
            #     st.append(to_number(p))
            # else:
            #     st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if b == 0:
                return None
            st.append(a / b)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a - b)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
                return None
            st.append(a ** b)
        elif p == "**" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a ** b)
        elif p == ">" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a > b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None


def split_ques_context(token_representations, question_mask, context_mask):
    question_output = torch.mean(util.replace_masked_values(token_representations, question_mask.unsqueeze(-1), 0),
                                 dim=1)
    context_output = util.replace_masked_values(token_representations, context_mask.unsqueeze(-1), 0)
    return context_output.transpose(1, 0), question_output


def get_all_number_encoder_outputs(encoder_outputs, table_num_pos, para_num_pos, table_cell_index, paragraph_index,
                                   num_size):
    st = []
    batch = 0
    for t_pos, p_pos in zip(table_num_pos, para_num_pos):
        cache = []
        for single_t in t_pos:
            emb = encoder_outputs[batch][table_cell_index[batch] == single_t]
            if emb.dim() == 1:
                emb.unsqueeze(0)
            cache.append(torch.mean(emb, 0))
        for single_p in p_pos:
            p_cache = []
            for i in range(single_p[0], single_p[1]+1): # note here, +1 when use non-block tokenize; otherwise no +1
                emb = encoder_outputs[batch][paragraph_index[batch] == i]
                if emb.dim() == 1:
                    emb.unsqueeze(0)
                elif emb.shape[0] < 1:
                    emb = encoder_outputs[batch]
                p_cache.append(torch.mean(emb, 0))
            emb = torch.stack(p_cache)
            cache.append(torch.mean(emb, 0))
        if len(cache) == 0:
            cache.append(encoder_outputs[batch][0])
        while len(cache) < num_size:
            cache.append(encoder_outputs[batch][0])
        
        st.append(torch.stack(cache))
        batch += 1
    # B x num_size x H
    return torch.stack(st)


def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target[i] == unk:
            max_score = -float("1e12")
            target[i] = num_start
            for num in range(len(nums_stack_batch[i])):
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


class Train_Tree(nn.Module):
    def __init__(self, hidden_size, embedding_size, max_copy_num, out_lang, dropout=0.5):
        super(Train_Tree, self).__init__()
        self.hidden_size = hidden_size
        self.out_lang = out_lang
        self.input_size = self.out_lang.gen_num
        self.op_num = self.out_lang.num_start
        self.max_copy_num = max_copy_num
        self.max_num_size = self.max_copy_num + self.input_size #
        self.predict_y = Prediction(hidden_size=hidden_size,
                                    op_nums=self.op_num,
                                    input_size=self.input_size,
                                    dropout=dropout)
        self.generate = GenerateNode(hidden_size=hidden_size,
                                     op_nums=self.op_num,
                                     embedding_size=embedding_size,
                                     dropout=dropout)
        self.merge = Merge(hidden_size=hidden_size, 
                           embedding_size=embedding_size,
                           dropout=dropout)

    def forward(self,
                seq_repr,  #  S * B * H
                root_repr,  # B * H
                all_num_repr,  # B * num_size * H
                copy_nums,  # B * N
                attention_mask,  # B * S
                out_len,  # B * N
                out_seq): # B * M

        # para_num_pos is [token] level 
        batch_size = len(copy_nums)
        
        num_mask = []
        for i in copy_nums:
            d = len(i) + self.input_size
            num_mask.append([0] * d + [1] * (self.max_num_size - d))
        num_mask = torch.BoolTensor(num_mask)
        unk = self.out_lang.word2index["UNK"]
        target = torch.LongTensor(out_seq).transpose(0, 1)
        if target.size(0) == 0:
            target = torch.zeros([batch_size, 1], dtype=torch.long).transpose(0, 1)
        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0)

        if torch.cuda.is_available():
            padding_hidden = padding_hidden.cuda()
            num_mask = num_mask.cuda()

        node_stacks = [[TreeNode(_)] for _ in root_repr.split(1, dim=0)]

        max_target_length = max(1, max(out_len))

        all_node_outputs = []

        copy_num_len = [len(_) for _ in copy_nums]
        num_size = self.max_copy_num # max(copy_num_len) 
        batch_num_reprs = []
        assert len(all_num_repr) == batch_size
        for bz, one_num_reprs in enumerate(all_num_repr):
            it = one_num_reprs + [padding_hidden.squeeze()] * (num_size - len(one_num_reprs))
            batch_num_reprs.append(torch.stack(it))

        all_nums_encoder_outputs = torch.stack(batch_num_reprs)

        num_start = self.out_lang.num_start
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]

        # 对节点遍历，遍历到target的max_len
        for t in range(max_target_length):
            # token y num/op score, context vector c, goal vector q
            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict_y(
                node_stacks, left_childs, seq_repr, all_nums_encoder_outputs, padding_hidden,
                (1 - attention_mask).bool(), num_mask)

            # all_leafs.append(p_leaf)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)
            # UNK -> num, num_ind -> 0
            target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, copy_nums, num_start,
                                                           unk)
            target[t] = target_t
            if torch.cuda.is_available():
                generate_input = generate_input.cuda()
            # subtree: [q,c,e(y|P)] -> subtree ql, qr           # B x 1 x H         # B x 1        # B x 1 x H
            left_child, right_child, node_label = self.generate(current_embeddings, generate_input, current_context)
            left_childs = []
            # 对当前节点根据predict token 进行不同操作
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                                   node_stacks, target[t].tolist(), embeddings_stacks):
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue

                if i < num_start:
                    node_stack.append(TreeNode(r))
                    node_stack.append(TreeNode(l, left_flag=True))
                    o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                else:
                    # current_nums_embeddings B * # of consts * H
                    current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)

                    while len(o) > 0 and o[-1].terminal:
                        sub_stree = o.pop()
                        op = o.pop()
                        current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                    o.append(TreeEmbedding(current_num, True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)

        all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x O
        out_mask = getmask(out_len)
        out = torch.argmax(all_node_outputs, 2)
        output = []
        for sig_out, sig_l in zip(out.tolist(), out_len):
            output.append(sig_out[:sig_l])
        target = target.transpose(0, 1).contiguous()
        if torch.cuda.is_available():
            all_node_outputs = all_node_outputs.cuda()
            target = target.cuda()
            out_mask = out_mask.cuda()
        acc = (out.masked_fill_(out_mask, 0) == target).min(1)[0].sum()

        loss = masked_cross_entropy(all_node_outputs, target, out_len)
        
        return loss, output, [int(acc), batch_size], all_node_outputs, (1 - out_mask.long()), target

    def predict(self,
                token_representations,  # S * B * H
                question_repr,  # B * H (1 * H)
                selected_node_vals,  # B * num of nodes
                select_node_embs,  # B * num of nodes
                attention_mask,
                beam_size=5,
                max_length=12):
        batch_size = 1

        # 1 x O x H
        all_nums_encoder_outputs = torch.stack(select_node_embs).unsqueeze(0)
        if torch.cuda.is_available():
            all_nums_encoder_outputs = all_nums_encoder_outputs.cuda()
        # S x 1 x H        1 x H

        padding_hidden = torch.FloatTensor([0.0 for _ in range(self.hidden_size)]).unsqueeze(0)
        node_stacks = [[TreeNode(question_repr)]]
        num_start = self.out_lang.num_start

        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]

        beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

        for t in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                # left_childs = torch.stack(b.left_childs)
                left_childs = b.left_childs

                num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict_y(
                    b.node_stack, left_childs, token_representations, all_nums_encoder_outputs, padding_hidden,
                    (1 - attention_mask).bool(), None)

                # leaf = p_leaf[:, 0].unsqueeze(1)
                # repeat_dims = [1] * leaf.dim()
                # repeat_dims[1] = op.size(1)
                # leaf = leaf.repeat(*repeat_dims)
                #
                # non_leaf = p_leaf[:, 1].unsqueeze(1)
                # repeat_dims = [1] * non_leaf.dim()
                # repeat_dims[1] = num_score.size(1)
                # non_leaf = non_leaf.repeat(*repeat_dims)
                #
                # p_leaf = torch.cat((leaf, non_leaf), dim=1)
                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

                # out_score = p_leaf * out_score

                topv, topi = out_score.topk(beam_size)

                # is_leaf = int(topi[0])
                # if is_leaf:
                #     topv, topi = op.topk(1)
                #     out_token = int(topi[0])
                # else:
                #     topv, topi = num_score.topk(1)
                #     out_token = int(topi[0]) + num_start

                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)

                    out_token = int(ti)
                    current_out.append(out_token)

                    node = current_node_stack[0].pop()

                    if out_token < num_start:
                        generate_input = torch.LongTensor([out_token])
                        if torch.cuda.is_available():
                            generate_input = generate_input.cuda()
                        left_child, right_child, node_label = self.generate(current_embeddings, generate_input,
                                                                            current_context)

                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks,
                                                  current_left_childs, current_out))
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break

        return beams[0].out


def getmask(l):
    ml = max(1, max(l))
    ret = []
    for i in l:
        ret.append([0] * i + [1] * (ml - i))
    return torch.BoolTensor(ret)


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)