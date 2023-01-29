from typing import Dict, List

import torch
import torch.nn.functional as F
from data.data_util import *
from data.data_tools import normalize_derivation
from tatqa_metric import TATEmAndF1
from .file_utils import is_scatter_available
from .tools.allennlp import replace_masked_values
from .heads.sequence_tag_head import SequenceTagHead
from .heads.single_span_head import SingleSpanHead
from .tree_model import *
from .tools import allennlp as util
from data.tag_constant import *
from .heads.node_predictor import NodePredictor
from .gnn.gat import *
from .gnn.graph_network import *
from tatqa_utils import *
import pandas as pd
import numpy as np
from .gnn.attention import DotProductAttention
from etr.model.tools.allennlp import replace_masked_values, masked_log_softmax


if is_scatter_available():
    from torch_scatter import scatter


def get_arithmetic_input(arithmetic_head_index, copy_nums, table_num_pos, para_num_pos, out_seq, out_len):
    copy_nums_c, table_num_pos_c, para_num_pos_c, out_seq_c, out_len_c = [], [], [], [], []
    for i, v in enumerate(arithmetic_head_index):
        if v:
            copy_nums_c.append(copy_nums[i])
            table_num_pos_c.append(table_num_pos[i])
            para_num_pos_c.append(para_num_pos[i])
            out_seq_c.append(out_seq[i])
            out_len_c.append(out_len[i])
    return copy_nums_c, table_num_pos_c, para_num_pos_c, out_seq_c, out_len_c


def get_copy_nums_from_paragraph(paragraph_token_tag_prediction, paragraph_tokens, paragraph_index, emb):
    para_copy_nums = []
    para_emb = []
    pattern = re.compile("\d+\.\d+%?|\d+%?")
    neg_pat = re.compile("\((\d+\.\d+%?|\d+%?)\)")
    for i in range(1, min(len(paragraph_tokens) + 1, len(paragraph_token_tag_prediction))):
        if paragraph_token_tag_prediction[i] != 0 and re.search('\d', paragraph_tokens[i - 1]):
            s = normalize_derivation(paragraph_tokens[i - 1])
            neg_p = re.search(neg_pat, s)
            if neg_p:
                para_copy_nums.append('-' + neg_p.group()[1:-1])
                para_emb.append(torch.mean(emb[paragraph_index == i], 0))
            else:
                p = re.search(pattern, s)
                if p:
                    para_copy_nums.append(p.group())
                    para_emb.append(torch.mean(emb[paragraph_index == i], 0))

    return para_copy_nums, para_emb


def get_copy_nums_from_table(table_cell_tag_prediction, table_cell_tokens, table_index, emb):
    table_copy_nums = []
    table_emb = []
    pattern = re.compile("\d+\.\d+%?|\d+%?")
    neg_pat = re.compile("\((\d+\.\d+%?|\d+%?)\)")
    for i in range(1, len(table_cell_tag_prediction)):
        if table_cell_tag_prediction[i] != 0:
            s = normalize_derivation(table_cell_tokens[i - 1])
            neg_p = re.search(neg_pat, s)
            if neg_p:
                table_copy_nums.append('-' + neg_p.group()[1:-1])
                table_emb.append(torch.mean(emb[table_index == i], 0))
            else:
                p = re.search(pattern, s)
                if p:
                    table_copy_nums.append(p.group())
                    table_emb.append(torch.mean(emb[table_index == i], 0))
    return table_copy_nums, table_emb


def get_best_span(span_start_probs, span_end_probs, span_index, span_tokens):
    if span_start_probs.dim() != 1 or span_end_probs.dim() != 1:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    passage_length = span_start_probs.shape[0]
    device = span_start_probs.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_probs.unsqueeze(1) + span_end_probs.unsqueeze(0)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_log_mask = torch.triu(torch.ones((passage_length, passage_length), device=device)).log()
    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
    best_spans = valid_span_log_probs.view(-1).argmax(-1)
    span_start_indice = best_spans // passage_length
    span_end_indice = best_spans % passage_length

    token_start_indice = int(span_index[span_start_indice])
    token_end_indice = int(span_index[span_end_indice])

    return [" ".join(span_tokens[token_start_indice: token_end_indice + 1])]

def get_span_from_words(token_tag_prediction, bz_selected_node_word_idx, bz_selected_node_words) -> List[str]:
    spans = []
    one_span = False
    for i in range(len(token_tag_prediction)):
        token2word_idx = int(bz_selected_node_word_idx[i])
        if token_tag_prediction[i] == 2:
            one_span = True
            spans.append([token2word_idx, token2word_idx])
        elif token_tag_prediction[i] == 1:
            if one_span:
                spans[-1][-1] = token2word_idx
            else:
                one_span = False
        elif token_tag_prediction[i] not in[1,2]:
            one_span = False

    selected_spans = [" ".join(bz_selected_node_words[s: e+1]) for s, e in spans]
    return list(set(selected_spans))


def get_span_from_tags(bz_tag_prediction, bz_selected_node_word_idx, bz_selected_node_words):
    token_tag_prediction = torch.argmax(bz_tag_prediction, dim=-1).float().squeeze()
    # token_tag_prediction = reduce_mean_index(token_tag_prediction, bz_selected_node_word_idx)
    token_tag_prediction = token_tag_prediction.detach().cpu().numpy()
    return get_span_from_words(token_tag_prediction, bz_selected_node_word_idx, bz_selected_node_words)


class Doc2SoarGraph(nn.Module):
    def __init__(self,
                 bert,
                 config,
                 scale_classes: int = 5,
                 head_count: int = 4,
                 out_lang=None,
                 hidden_size: int = None,
                 embedding_size: int = 256,
                 dropout_prob: float = None,
                 scale_criterion: nn.CrossEntropyLoss = None,
                 dataset='finqa',
                 encoder="roberta",
                 max_seq_len = 561,
                 max_input_len = 512,
                 max_copy_num=6,
                 mode='train'
                 ):
        super(Doc2SoarGraph, self).__init__()
        self.lm = bert
        self.encoder = encoder
        self.config = config
        self.scale_classes = scale_classes
        self._metrics = TATEmAndF1()
        if hidden_size is None:
            hidden_size = self.config.hidden_size
        if dropout_prob is None:
            dropout_prob = self.config.hidden_dropout_prob
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.max_input_len = max_input_len
        self.head_count = head_count
        self.NodeNLLLoss = nn.NLLLoss(weight=torch.from_numpy(np.array([1.0, 200.0])).float(), reduction="mean")
        self.NLLLoss = nn.NLLLoss(reduction="mean")
        self.scale_criterion = scale_criterion
        self.scale_predictor = Default_FNN(hidden_size, hidden_size // 2, scale_classes, dropout_prob)
        self.paragraph_summary_vector_module = nn.Linear(hidden_size, 1)
        self.table_summary_vector_module = nn.Linear(hidden_size, 1)
        self.question_summary_vector_module = nn.Linear(hidden_size, 1)
        self.paragraph_summary_vector_module = nn.Linear(hidden_size, 1)
        self.node_summary_vector_module = nn.Linear(hidden_size, 1)

        self.gcn_dropout_prob = 0.6
        self.tree_dropout_prob = 0.5
        self.num_encoder = EncoderSeq(hidden_size=hidden_size, dropout=self.gcn_dropout_prob)
        self.date_encoder =    EncoderSeq(hidden_size=hidden_size, dropout=self.gcn_dropout_prob)  
        self.semantic_encoder =  EncoderSeq(hidden_size=hidden_size, dropout=self.gcn_dropout_prob)
        self.full_encoder = EncoderSeq(hidden_size=hidden_size, dropout=self.gcn_dropout_prob) #  #

        self.head_predictor = Default_FNN(hidden_size, hidden_size // 2, head_count, dropout_prob)
        self.node_predictor = NodePredictor(hidden_size, dropout_prob)

        self.node_attention_layer = DotProductAttention(hidden_size)
        self.single_span_head = SingleSpanHead(2 * hidden_size)
        # self.single_span_head = SingleSpanHead(hidden_size)
        self.sequence_tag_head = SequenceTagHead( 2 * hidden_size, dropout_prob)

        self.HEAD_CLASSES = HEAD_CLASSES_
        self.out_lang = out_lang
        # self.max_copy_num = self.out_lang.max_copy_num
        self.max_copy_num = max_copy_num # if max_copy_num < self.out_lang.max_copy_num else self.out_lang.max_copy_num
        self.copy_num_start = self.out_lang.copy_num_start

        self.train_tree = Train_Tree(hidden_size, embedding_size, self.max_copy_num, out_lang, dropout=self.tree_dropout_prob)
        self.skip_batch = 0
        self.acc = [0, 0]
        self.anwserlist = []
        self.arith_nan = []
        self.pred_err = []
        self.count = []
        self.dataset = dataset
        self.mode = mode
        self.node_mask = False

    def heads_indices(self):
        return list(self._heads.keys())

    def summary_vector(self, encoding, mask, in_type='paragraph'):

        if in_type == 'paragraph':
            # Shape: (batch_size, seqlen)
            alpha = self.paragraph_summary_vector_module(encoding).squeeze()
        elif in_type == 'question':
            # Shape: (batch_size, seqlen)
            alpha = self.question_summary_vector_module(encoding).squeeze()
        elif in_type == 'table':
            alpha = self.table_summary_vector_module(encoding).squeeze()
        elif in_type == 'node':
            alpha = self.node_summary_vector_module(encoding).squeeze()
        else:
            # Shape: (batch_size, #num of numbers, seqlen)
            alpha = torch.zeros(encoding.shape[:-1], device=encoding.device)
        # Shape: (batch_size, seqlen)
        # (batch_size, #num of numbers, seqlen) for numbers
        alpha = util.masked_softmax(alpha, mask)
        # Shape: (batch_size, out)
        # (batch_size, #num of numbers, out) for numbers
        h = util.weighted_sum(encoding, alpha)
        return h

    def compute_prefix_tree_result(self, out, num_list):

        try:
            out_list = out_expression_list(out, self.out_lang, num_list)
            answer = compute_prefix_expression(out_list)
        except Exception as e:
            # print('warning: computing the expression failed, will return none as the answer', e)
            answer = ''
        return answer

    def offset2mask(self, offset, seq_len=561, val=1):
        mask = torch.zeros([seq_len])

        if torch.cuda.is_available():
            mask = mask.cuda()

        if offset[1] > offset[0]:
            mask[offset[0]: offset[1]] = val

        return mask.unsqueeze(-1)

    def offset2seq(self, offset, seq_len=561):
        seq = np.zeros(seq_len)

        if offset[1] > offset[0]:
            seq[offset[0]: offset[1]] = np.arange(offset[1] - offset[0])

        return seq.astype(int)


    def get_seq_tags(self,
                     head_idx,
                     selected_nodes,
                     selected_node_seq_embs,
                     selected_node_seq_masks):

        gold_seq_tags = []
        filtered_node_seq_embs = []
        filtered_node_seq_masks = []
        for bz, is_target in enumerate(head_idx):
            gold_nodes = [node for node in selected_nodes[bz] if node[-1] == 1]
            self_nodes = [node for node in gold_nodes if node[0] == TAG_SELF[0]]
            if is_target and self_nodes:
                bz_gold_tags = np.zeros(self.max_input_len)
                for node in gold_nodes:
                    tag, val, emb, seq_offset, ans_offsets, token2word_idx, words, ord, label = node
                    if isinstance(ans_offsets[0], int):
                         ans_offsets = [ans_offsets]
                    for one_offset in ans_offsets:
                        s = one_offset[0]
                        e = one_offset[1]
                        node_seq_tags = np.ones(e -s)
                        node_seq_tags[0] = 2
                        bz_gold_tags[s:e] = node_seq_tags
                gold_seq_tags.append(bz_gold_tags)
                filtered_node_seq_embs.append(selected_node_seq_embs[bz])
                filtered_node_seq_masks.append(selected_node_seq_masks[bz])

        gold_seq_tags = torch.from_numpy(np.array(gold_seq_tags)).long().cuda()
        if len(gold_seq_tags) > 0:
            filtered_node_seq_embs = torch.stack(filtered_node_seq_embs)
            filtered_node_seq_masks = torch.stack(filtered_node_seq_masks)
        return gold_seq_tags, filtered_node_seq_embs, filtered_node_seq_masks

    def get_node_seq_words(self, nodes):
        selected_node_word_idx = self.offset2mask(offset=[0, self.max_input_len], val=0)
        selected_node_words_list = []
        cur_word_count = 0
        for node in nodes:
            tag, self_node_val, self_node_emb, self_seq_offsets, _, node_token2word_idx, node_words, ord, label = node

            s = self_seq_offsets[0]
            e = self_seq_offsets[1]
            valid_token2word_idx = node_token2word_idx[:e - s]
            valid_token2word_idx = np.array(valid_token2word_idx) + cur_word_count
            valid_token2word_idx = torch.from_numpy(valid_token2word_idx)

            selected_node_word_idx[s:e, :] = valid_token2word_idx.unsqueeze(-1)
            selected_node_words_list.extend(node_words)
            cur_word_count += len(node_words)
        return selected_node_word_idx, selected_node_words_list

    def get_node_emb(self, one_seq_repr, masks):
        h = torch.sum(one_seq_repr * masks.float(), dim=0) / torch.sum(masks).float()
        return h

    def get_bz_node_embs(self, bz_token_embs, bz_nodes):
        '''
         node = (sign, TAG_SELF[0], val, source, seq_offsets, answer_offsets, tokens, -1, is_target)
        :param bz_token_embs:
        :param bz_nodes:
        :return:
        '''
        bz_node_embs = []
        for node in bz_nodes:
            seq_offsets = node[4]
            node_emb = self.get_node_emb(bz_token_embs, self.offset2mask(seq_offsets, self.max_input_len))
            bz_node_embs.append(node_emb)
        return bz_node_embs

    def get_node_val_and_embs(self, batch_node_list, max_length, token_representations):
        # B * x * x
        final_node_tags = []
        final_node_vals = []
        final_node_embeddings = []
        final_node_masks = []
        final_node_labels = []
        final_node_label_ords = []

        final_node_seq_offsets = []
        final_node_ans_offsets = []
        final_node_token2word_idx = []
        final_node_words = []

        # node = (sign, tag, val, source, offset, answer_offsets, tokens, order, is_target))
        for bz, one_seq_nodes in enumerate(batch_node_list):

            ori_node_vals = np.array([None] * max_length)
            embs = torch.zeros(max_length, self.hidden_size)
            node_labels = torch.zeros(max_length)
            node_label_ords = torch.zeros(max_length)

            node_tags = np.array([None] * max_length)
            node_seq_offsets = np.array([None] * max_length)
            node_ans_offsets = np.array([None] * max_length)
            node_words = np.array([None] * max_length)
            node_token2word_idx = np.array([None] * max_length)
            bz_token_embs = token_representations[bz]

            if torch.cuda.is_available():
                embs = embs.cuda()
                node_labels = node_labels.cuda()
                node_label_ords = node_label_ords.cuda()

            if len(one_seq_nodes) > 0:
                end = len(one_seq_nodes)
                node_tags[:end] = [it[1] for it in one_seq_nodes]
                ori_node_vals[:end] = [it[2] for it in one_seq_nodes]
                bz_node_embs = self.get_bz_node_embs(bz_token_embs, one_seq_nodes)
                embs[0:end, :] = torch.stack(bz_node_embs)
                node_labels[:end] = torch.from_numpy(np.array([it[-1] for it in one_seq_nodes]))
                node_label_ords[:end] = torch.from_numpy(np.array([it[-2] for it in one_seq_nodes]))

                node_seq_offsets[:end] = [it[4] for it in one_seq_nodes]
                node_ans_offsets[:end] = [it[5] for it in one_seq_nodes]
                node_token2word_idx[:end] = [it[6] for it in one_seq_nodes]
                node_words[:end] = [it[7] for it in one_seq_nodes]

            seq_mask = self.offset2mask(offset=[0, len(one_seq_nodes)], seq_len=max_length)
            final_node_masks.append(seq_mask)

            final_node_tags.append(node_tags)
            final_node_vals.append(ori_node_vals)
            final_node_embeddings.append(embs)
            final_node_labels.append(node_labels)
            final_node_label_ords.append(node_label_ords)

            final_node_seq_offsets.append(node_seq_offsets)
            final_node_ans_offsets.append(node_ans_offsets)
            final_node_token2word_idx.append(node_token2word_idx)
            final_node_words.append(node_words)

        # B * max_len * h
        final_node_embeddings = torch.stack(final_node_embeddings, 0)
        # B * max_len * h
        final_node_masks = torch.stack(final_node_masks, 0)
        # B * max_len * h
        final_node_labels = torch.stack(final_node_labels, 0)
        # B * max_len * h
        final_node_label_ords = torch.stack(final_node_label_ords, 0)

        return final_node_tags, final_node_vals, final_node_embeddings, final_node_masks, final_node_labels, \
               final_node_label_ords, final_node_seq_offsets, final_node_ans_offsets, final_node_token2word_idx, final_node_words

    def gat_encode(self, features, graph, encoder):
        gat_features = []
        for bz in range(len(features)):
            gat_features.append(encoder(features[bz], graph[bz]))
        gat_outputs = torch.stack(gat_features)
        return gat_outputs

    def expand_graphs(self, graph_list, max_len):
        result = []
        for g in graph_list:
            new_graph = np.diag(np.zeros(max_len))
            for row in range(len(g)):
                new_graph[row, 0:len(g[row])] = g[row]
            result.append(new_graph)
        return result

    def dot_product_scores(self, q_vectors, node_vectors):
        """
        calculates q->ctx scores for every row in ctx_vector
        :param q_vector:
        :param node_vectors:
        :return:
        """
        # q_vector: b x 1 x D, ctx_vectors: b x n2 x D, result b x n1 x n2
        r = torch.matmul(q_vectors.unsqueeze(1), torch.transpose(node_vectors, 1, 2))
        return torch.transpose(r, 1, 2)

    def forward(self,
                input_ids: torch.LongTensor,
                input_bbox:torch.LongTensor,
                attention_mask: torch.LongTensor,
                token_type_ids: torch.LongTensor,
                question_mask: torch.LongTensor,
                bbox_mask: torch.LongTensor,
                # question_metas,
                # table_metas,
                # paragraph_metas,
                answer_dict,
                question_ids: List[str],
                facts,
                consts,
                # answer_mappings,
                out_len,
                out_seq,
                num_nodes,
                num_graphs,
                date_nodes,
                date_graphs,
                semantic_nodes,
                semantic_graphs,
                full_graphs,
                image: torch.LongTensor,
                position_ids: torch.LongTensor = None,
                mode='train',
                epoch=None,
                dataset='tatqa') -> Dict[str, torch.Tensor]:

        self.dataset='tatdqa'
        output_dict = {}

        # bert embedding
        outputs = None
        if self.encoder == 'bert':
            outputs = self.lm(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids)
        elif self.encoder == 'roberta':
            outputs = self.lm(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids)
        elif self.encoder == 'layoutlm':
            outputs = self.lm(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              bbox=input_bbox)

        elif self.encoder == 'layoutlm_v2':
            outputs = self.lm(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              bbox=input_bbox,
                              image=image)

        token_representations = outputs[0][:,:self.max_input_len,:]

        # cls token output
        cls_output = token_representations[:, 0, :]

        batch_size = token_representations.shape[0]
        seq_len = token_representations.shape[1]
        hidden_size = token_representations.shape[-1]

        full_nodes = [n_nodes + d_nodes + s_nodes for n_nodes, d_nodes, s_nodes in list(zip(num_nodes, date_nodes, semantic_nodes))]
        question_node_idxes = [len(n_nodes) + len(d_nodes) for n_nodes, d_nodes in list(zip(num_nodes, date_nodes))]

        max_num_length = max([len(it) for it in num_nodes])
        max_date_length = max([len(it) for it in date_nodes])
        max_semantic_length = max([len(it) for it in semantic_nodes])
        max_full_length = max([len(it) for it in full_nodes])

        _, num_node_vals, num_node_embs, num_node_masks, num_node_labels, num_node_label_ords, num_node_seq_offsets, num_node_ans_offsets, _, _ = \
            self.get_node_val_and_embs(num_nodes, max_num_length, token_representations)
        _, date_node_vals, date_node_embs, date_node_masks, date_node_labels, date_node_label_ords, date_node_seq_offsets, date_node_ans_offsets, _, _ = \
            self.get_node_val_and_embs(date_nodes, max_date_length, token_representations)
        _, semantic_node_vals, semantic_node_embs, semantic_node_masks, semantic_node_labels, semantic_node_label_ords, semantic_node_seq_offsets, semantic_node_ans_offsets, _, _  = \
            self.get_node_val_and_embs(semantic_nodes, max_semantic_length, token_representations)
        full_node_tags, full_node_vals, full_node_embs, full_node_masks, full_node_labels, full_node_label_ords, full_node_seq_offsets, full_node_ans_offsets, final_node_token2word_idx, full_node_words = \
            self.get_node_val_and_embs(full_nodes, max_full_length, token_representations)

        '''1.1 number graph'''
        num_encoder_outputs = num_node_embs
        if max_num_length > 0:
            expand_num_graphs = self.expand_graphs(num_graphs, max_num_length)
            num_graph_tensor = torch.LongTensor(np.array(expand_num_graphs))
            if torch.cuda.is_available():
                num_graph_tensor = num_graph_tensor.cuda()
            num_encoder_outputs = self.num_encoder(num_node_embs, num_graph_tensor)

        ''' 1.2 date graph'''
        date_encoder_outputs = date_node_embs
        if max_date_length > 0:
            expand_date_graphs = self.expand_graphs(date_graphs, max_date_length)
            date_graph_tensor = torch.LongTensor(np.array(expand_date_graphs))
            if torch.cuda.is_available():
                date_graph_tensor = date_graph_tensor.cuda()
            date_encoder_outputs = self.date_encoder(date_node_embs, date_graph_tensor)

        ''' 1.3 semantic graph'''
        semantic_encoder_outputs = semantic_node_embs
        expand_semantic_graphs = self.expand_graphs(semantic_graphs, max_semantic_length)
        semantic_graph_tensor = torch.LongTensor(np.array(expand_semantic_graphs))
        if torch.cuda.is_available():
            semantic_graph_tensor = semantic_graph_tensor.cuda()
        semantic_encoder_outputs = self.semantic_encoder(semantic_node_embs, semantic_graph_tensor)

        '''2 To build full/hybrid graph '''
        # B * # of num + date + semantic nodes
        # full_node_vals = np.concatenate([num_node_vals, date_node_vals, semantic_node_vals], axis=1)
        # B * # of num + date +  semanticnodes * H

        full_node_reps = torch.zeros(batch_size, max_full_length, hidden_size)
        if torch.cuda.is_available():
            full_node_reps = full_node_reps.cuda()

        for bz in range(batch_size):
            num_len = int(sum(num_node_masks[bz]).item())
            date_len = int(sum(date_node_masks[bz]).item()) if len(date_node_masks[bz]) > 0  else 0
            semantic_len = int(sum(semantic_node_masks[bz]).item())
            full_node_reps[bz, 0:num_len, :] = num_encoder_outputs[bz, 0:num_len, :]
            full_node_reps[bz, num_len:num_len + date_len, :] = date_encoder_outputs[bz, 0:date_len, :]
            full_node_reps[bz, num_len + date_len:num_len + date_len + semantic_len, :] = semantic_encoder_outputs[bz, 0:semantic_len, :]

        expand_full_graphs = self.expand_graphs(full_graphs, max_full_length)
        full_graph_tensor = torch.LongTensor(np.array(expand_full_graphs))
        if torch.cuda.is_available():
            full_graph_tensor = full_graph_tensor.cuda()

        # b * num of nodes * H
        full_encoder_outputs = full_node_reps
        full_encoder_outputs = self.full_encoder(full_node_reps, full_graph_tensor)

        full_encoder_outputs = util.replace_masked_values(full_encoder_outputs, full_node_masks, 0)

        # B * H (Avg Pooling)
        # full_graph_embs = cls_output
        
        full_graph_embs = torch.mean(full_encoder_outputs, dim=1, keepdim=False)

        question_reprs = torch.stack([full_encoder_outputs[bz,idx,:].squeeze() for bz, idx in enumerate(question_node_idxes)])

        # B * max_node_len * 1
        # q_mask = torch.stack([self.offset2mask(offset=[idx, idx+1],seq_len=max_full_length, val=-1) + 1 for bz, idx in enumerate(question_node_idxes)])
        # q_sim_node_masks = full_node_masks * q_mask
        # q_node_sim_scores = self.dot_product_scores(question_reprs, full_encoder_outputs * q_sim_node_masks)
        # node_prediction_log_probs = self.node_predictor(q_node_sim_scores, q_sim_node_masks.squeeze(-1))

        loss = 0
        # sim_node_masks = full_node_masks[bz:idx] = 0 for bz, idx in enumerate(question_node_idxes)
        # B * max_node_len * 2
        node_prediction_log_probs = self.node_predictor(full_encoder_outputs, full_node_masks.squeeze(-1))
        node_loss = self.NodeNLLLoss(node_prediction_log_probs.view(-1, 2), full_node_labels.long().view(-1))

        # B * max_node_len
        node_predictions = torch.argmax(node_prediction_log_probs, dim=-1).cpu().detach()
        loss += 2 * node_loss

        output_dict['loss_sp'] = {}
        output_dict['head_acc'] = 1
        output_dict['scale_acc'] = 1

        output_dict['loss_sp']['node_precision'] = []
        output_dict['loss_sp']['node_recall'] = []
        output_dict['loss_sp']['node_metric_detail'] = []

        # max_num_of_selected_nodes = 6
        # k_selected_indices = []
        # if mode != 'train':
        #     # B * topk * 1
        #     _, k_selected_indices = torch.topk(q_node_sim_scores, dim=1, k=max_num_of_selected_nodes)
        #     k_selected_indices = k_selected_indices.squeeze(-1)

        # analyze the prediction
        gold_full_nodes = []
        # gold_full_vals = []
        # gold_full_embs = []
        # gold_full_seq_offsets = []
        gold_full_ans_offsets = []

        full_model_selected_nodes = []

        for bz in range(batch_size):
            one_bz_selected_nodes = []
            one_bz_gold_nodes = []
            one_bz_gold_ans_offsets = []
            for idx, (cls, tag, val, emb, label, ord, seq_offset, ans_offsets, mask, token2word_idx, words) in enumerate(list(
                    zip(node_predictions[bz], full_node_tags[bz], full_node_vals[bz], full_encoder_outputs[bz], full_node_labels[bz], full_node_label_ords[bz],
                        full_node_seq_offsets[bz], full_node_ans_offsets[bz], full_node_masks[bz], final_node_token2word_idx[bz], full_node_words[bz]))):
                if label == 1:
                    one_bz_gold_nodes.append((tag, val, emb, seq_offset, ans_offsets, token2word_idx, words, ord, label))
                    if isinstance(ans_offsets[0], int):
                        ans_offsets = [ans_offsets]
                    one_bz_gold_ans_offsets.append(ans_offsets)

                if  mask == 1 and cls == 1: # mode == 'train' and
                    one_bz_selected_nodes.append((tag, val, emb, seq_offset, ans_offsets, token2word_idx, words, ord, label))
                
                # if mode != 'train' and idx in k_selected_indices[bz]:
                #     one_bz_selected_nodes.append((tag, val, emb, seq_offset, ans_offsets, token2word_idx, words, ord, label))

        
            one_bz_gold_nodes = sorted(one_bz_gold_nodes, key=lambda it: it[-2])
            gold_full_ans_offsets.append(one_bz_gold_ans_offsets)
            gold_full_nodes.append(one_bz_gold_nodes)
            full_model_selected_nodes.append(one_bz_selected_nodes)
        
        full_selected_nodes = []
        full_selected_num_and_date_vals = []
        full_selected_num_and_date_embs = []
        full_out_seq = []
        for bz in range(batch_size):
            one_bz_q_repr = question_reprs[bz]
            one_bz_out_seq = out_seq[bz]
            one_bz_gold_nodes = gold_full_nodes[bz]
            bz_selected_nodes = full_model_selected_nodes[bz]
            # if mode == 'train':
            one_bz_sorted_selected_nodes = sorted(bz_selected_nodes, key=lambda it: torch.dot(it[2], one_bz_q_repr), reverse=True)[:self.max_copy_num]
            # else:
            #     one_bz_sorted_selected_nodes = bz_selected_nodes  # bz_selected_nodes[:self.max_copy_num] #  
            # one_bz_sorted_selected_nodes = bz_selected_nodes 
            selected_node_labels = [it[-1] for it in one_bz_sorted_selected_nodes]
            correct_node = sum(selected_node_labels)
            if type(correct_node) is torch.Tensor:
                correct_node = correct_node.item()
            all_pred_node = len(selected_node_labels)

            total_node = len(one_bz_gold_nodes) 
            node_precision = 0 if all_pred_node == 0 else correct_node / all_pred_node
            recall = 0 if total_node == 0 else correct_node / total_node
            output_dict['loss_sp']['node_precision'].append(node_precision)
            output_dict['loss_sp']['node_recall'].append(recall)
            output_dict['loss_sp']['node_metric_detail'].append(f'qid:{question_ids[bz]},  precision: {node_precision} = {correct_node} / {all_pred_node}, recall: {recall} = {correct_node} / {total_node})')

            if mode == 'train': # add missing nodes if missing
                missed_gold_nodes = []
                # max len
                # one_bz_sorted_selected_nodes =
                for one_gold_node in one_bz_gold_nodes:
                    found = False
                    for select_node in one_bz_sorted_selected_nodes:
                        if select_node[-1] == 1 and select_node[-2] == one_gold_node[-2]:
                            found = True
                            break
                    if not found:
                        missed_gold_nodes.append(one_gold_node)
                one_bz_sorted_selected_nodes.extend(missed_gold_nodes)
                if len(one_bz_sorted_selected_nodes) > self.max_copy_num:
                    one_bz_sorted_selected_nodes = sorted(one_bz_sorted_selected_nodes, key=lambda it: it[-1], reverse=True)[:self.max_copy_num]

            one_bz_label_ord_idx_mapping = {int(self.copy_num_start + node[-2]): int(self.copy_num_start + idx) for idx, node in enumerate(one_bz_sorted_selected_nodes) if node[-1] == 1}

            one_bz_updated_out_seq = []
            for idx in one_bz_out_seq:
                new_idx = one_bz_label_ord_idx_mapping[idx] if idx in one_bz_label_ord_idx_mapping else idx
                one_bz_updated_out_seq.append(new_idx)

            full_selected_num_and_date_vals.append([it[1] for it in one_bz_sorted_selected_nodes if it[0] in [TAG_NUMBER[0], TAG_DATE[0]]])
            full_selected_num_and_date_embs.append([it[2] for it in one_bz_sorted_selected_nodes if it[0] in [TAG_NUMBER[0], TAG_DATE[0]]])
            full_selected_nodes.append(one_bz_sorted_selected_nodes)
            full_out_seq.append(one_bz_updated_out_seq)

        # B * H (Avg Pooling)
        para_span_loss = 0
       
        scale_labels = torch.from_numpy(np.array([SCALE.index(it['scale']) for it in answer_dict])).cuda()
        head_labels = torch.from_numpy(np.array([it['head_class'] for it in answer_dict])).cuda()

        scale_predict_logits = self.scale_predictor(full_graph_embs)
        answer_head_logits = self.head_predictor(full_graph_embs)
        answer_head_log_probs = F.log_softmax(answer_head_logits, -1)
        scale_log_probs = F.log_softmax(scale_predict_logits, -1)

        scale_loss = self.NLLLoss(scale_log_probs, scale_labels)
        scale_prediction = torch.argmax(scale_log_probs, dim=-1).cpu().detach()

        head_loss = self.NLLLoss(answer_head_log_probs, head_labels)
        predict_head = torch.argmax(answer_head_log_probs, dim=-1).cpu().detach()

        loss += head_loss + scale_loss
        #
        head_acc = (predict_head == head_labels.cpu().detach()).float().mean()
        scale_acc = (scale_prediction == scale_labels.cpu().detach()).float().mean()

        output_dict["head"] = predict_head
        output_dict["scale"] = [SCALE[int(it)] for it in scale_prediction]
        output_dict['head_acc'] = head_acc
        output_dict['scale_acc'] = scale_acc
        output_dict['loss_sp']['scale_loss'] = scale_loss.item()
        output_dict['loss_sp']['head_loss'] = head_loss.item()

        selected_node_seq_embs = []
        selected_node_seq_masks = []
        for bz in range(batch_size):

            bz_select_nodes = full_selected_nodes[bz]

            bz_node_embs = torch.zeros(seq_len, hidden_size)
            
            default_mask_val = 0
            if not self.node_mask :
                default_mask_val = 1
                
            bz_node_masks = self.offset2mask(offset=[0, seq_len],seq_len=seq_len, val=default_mask_val)
            if torch.cuda.is_available():
                bz_node_embs = bz_node_embs.cuda()

            if self.node_mask:
                for node in bz_select_nodes:
                    tag, val, emb, seq_offset, ans_offsets, token2word_idx, words, ord, label = node
                
                    s = seq_offset[0]
                    e = seq_offset[1]
                    bz_node_embs[s:e, :] = emb.expand(e - s, -1)
                    bz_node_masks[s:e, :] = 1
            selected_node_seq_embs.append(bz_node_embs)
            selected_node_seq_masks.append(bz_node_masks)

        selected_node_seq_embs = torch.stack(selected_node_seq_embs)
        selected_node_seq_masks = torch.stack(selected_node_seq_masks).squeeze(-1)

        # B * S * H
        selected_node_seq_embs = torch.cat([token_representations, selected_node_seq_embs], 2)# cat
        # selected_node_seq_embs, _ = self.node_attention_layer(query=selected_node_seq_embs, value=token_representations)

        # HEAD_CLASSES_ = {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "COUNT": 3, "ARITHMETIC": 4}

        para_span_head_idx = (head_labels.squeeze() == 0).cpu().detach()
        multi_span_head_idx = (head_labels.squeeze() == 2).cpu().detach()
        count_head_idx = (head_labels.squeeze() == 3).cpu().detach()
        arithmetic_head_index = (head_labels.squeeze() == 4).cpu().detach()

        # single SPAN
        if para_span_head_idx.dim() > 0 and node_predictions[para_span_head_idx].shape[0] != 0:
            '''only paragraph span need to compute the loss'''
            gold_ans_offsets = []
            for bz, is_span in enumerate(para_span_head_idx):
                if is_span:
                    gold_ans_offsets.append(gold_full_ans_offsets[bz][0][0])

            gold_ans_offsets = torch.from_numpy(np.array(gold_ans_offsets)).cuda()

            # gold_answer_offsets
            gold_span_starts = gold_ans_offsets[:, 0]
            gold_span_ends = gold_ans_offsets[:, 1] - 1  # end need to minus one

            # start_log_probs, end_log_probs = self.single_span_head(token_representations, attention_mask)
            start_log_probs, end_log_probs = self.single_span_head(selected_node_seq_embs[para_span_head_idx], selected_node_seq_masks[para_span_head_idx])

            start_pos_loss = self.NLLLoss(start_log_probs, gold_span_starts)
            end_pos_loss = self.NLLLoss(end_log_probs, gold_span_ends)

            para_span_loss = start_pos_loss + end_pos_loss
           
            loss += para_span_loss
            output_dict['loss_sp']['para_span_loss'] = para_span_loss.item()

        #  multiple SPANS
        if multi_span_head_idx.dim() > 0 and node_predictions[multi_span_head_idx].shape[0] != 0:

            gold_seq_tags, multi_span_node_seq_embs, multi_span_node_seq_masks =self.get_seq_tags(multi_span_head_idx,
                                                                                                    full_selected_nodes,
                                                                                                    selected_node_seq_embs ,
                                                                                                    selected_node_seq_masks)

            if len(gold_seq_tags) > 0 :
                seq_token_tag_log_probs = self.sequence_tag_head(multi_span_node_seq_embs, multi_span_node_seq_masks)
                multi_span_head_loss = self.NLLLoss(seq_token_tag_log_probs.transpose(1, 2),  gold_seq_tags)

                loss += multi_span_head_loss
                output_dict['loss_sp']['multi_span_head_loss'] = multi_span_head_loss.item()

        # Count
        if count_head_idx.dim() > 0 and  node_predictions[count_head_idx].shape[0] != 0:

            gold_seq_tags, count_node_seq_embs, count_node_seq_masks =self.get_seq_tags(count_head_idx,
                                                                                        full_selected_nodes,
                                                                                        selected_node_seq_embs ,
                                                                                        selected_node_seq_masks)
            if len(gold_seq_tags) > 0 :
                seq_token_tag_log_probs = self.sequence_tag_head(count_node_seq_embs, count_node_seq_masks)
                count_head_loss = self.NLLLoss(seq_token_tag_log_probs.transpose(1, 2),  gold_seq_tags)

                loss += count_head_loss
                output_dict['loss_sp']['count_head_loss'] = count_head_loss.item()

        # seq_repr,  # B * S * H
        # question_repr,  # B * H
        # all_num_repr,  # B * num_size * H
        # copy_nums,  # B * num_size
        # attention_mask,  # B * S
        # out_len,  # B * N
        # out_seq):  # B * M
        # gold_nodes_embs =  gold_nodes_embeddings
        # facts,
        # assert len(gold_num_and_date_vals) == len(facts)
        tree_loss = 0

        # context_output = util.replace_masked_values(token_representations, attention_mask.unsqueeze(-1), 0)
        # using the node embedding as the tree input
        context_output = util.replace_masked_values(full_encoder_outputs, full_node_masks, 0)
        context_output = context_output.transpose(1, 0)
        # attention_mask = full_node_masks.squeeze(-1)

        # Arithmetic 
        # arithmetic_head_index.dim() == 0 
        if arithmetic_head_index.dim() > 0 and node_predictions[arithmetic_head_index].shape[0] != 0:
            tree_full_node_embs = []
            tree_full_node_vals = []
            tree_out_len = []
            tree_out_seq = []
            for bz, is_arithmetic in enumerate(arithmetic_head_index):
                if is_arithmetic:
                    tree_full_node_embs.append(full_selected_num_and_date_embs[bz])
                    tree_full_node_vals.append(full_selected_num_and_date_vals[bz])
                    tree_out_len.append(out_len[bz])
                    tree_out_seq.append(full_out_seq[bz])

            if tree_full_node_embs:
                tree_loss, pred_out_seq, acc_c, _, _, _ = self.train_tree(context_output[:,arithmetic_head_index,:],
                                                                          question_reprs[arithmetic_head_index], # full_graph_embs[arithmetic_head_index],
                                                                          tree_full_node_embs,
                                                                          tree_full_node_vals, # must use facts, some times
                                                                          full_node_masks[arithmetic_head_index].squeeze(-1),
                                                                          tree_out_len,
                                                                          tree_out_seq)
            if torch.isnan(tree_loss) or tree_loss > 1000:
                print(f'tree loss is nan or greater than 1000')
                self.skip_batch += 1
            else:
                loss += tree_loss
                output_dict['loss_sp']['tree_loss'] = tree_loss.item()

            self.acc[0] += acc_c[0]
            self.acc[1] += acc_c[1]

       
        output_dict['loss_sp']['node_loss'] = node_loss.item()
        output_dict['loss_sp']['total_loss'] = loss.item()
        output_dict['loss'] = loss
        output_dict["question_id"] = []
        output_dict["prediction"] = []
        output_dict["answer"] = []

        # Evaluation Part 
        with torch.no_grad():
            for bz in range(batch_size):
                pred_val = ""
                pred_head = ""
                pred_scale = ""
                gold_head = ""
                pred_seq = ""

                bz_selected_nodes = full_selected_nodes[bz]

                bz_selected_node_vals = [it[1] for it in bz_selected_nodes]

                pred_head = predict_head[bz]
                gold_head = head_labels[bz]
                pred_scale = SCALE[int(scale_prediction[bz])]

                if bz_selected_nodes:
                    if predict_head[bz] == self.HEAD_CLASSES["ARITHMETIC"]:
                        try:
                            selected_num_and_date_vals = [it[1] for it in bz_selected_nodes if it[0] in [TAG_NUMBER[0], TAG_DATE[0]]]
                            selected_num_and_date_embs = [it[2] for it in bz_selected_nodes if it[0] in [TAG_NUMBER[0], TAG_DATE[0]]]
                            if not selected_num_and_date_embs:
                                pred_val = None
                            else:
                                pred_seq = self.train_tree.predict(context_output[:,bz,:].unsqueeze(1),
                                                                   question_reprs[bz].unsqueeze(0), # full_graph_embs[bz].unsqueeze(0),
                                                                   selected_num_and_date_vals,
                                                                   selected_num_and_date_embs,
                                                                   full_node_masks.squeeze(-1)[bz].unsqueeze(0))

                                pred_val = self.compute_prefix_tree_result(pred_seq, selected_num_and_date_vals)
                        except SyntaxError as e:
                            print(f"tree prediction has syntax error:{e}")
                            pred_val = None
                        except Exception as e:
                            print(f"tree prediction has unexpected error:{e}")
                            pred_val = None
                    else:
                        ''''''
                        bz_selected_node_seq_embs = selected_node_seq_embs[bz].unsqueeze(0)
                        bz_selected_node_seq_masks = selected_node_seq_masks[bz].unsqueeze(0)
                        if predict_head[bz] == self.HEAD_CLASSES["SPAN-TEXT"]:
                            self_nodes = [node for node in bz_selected_nodes if node[0] == TAG_SELF[0]]
                            if len(self_nodes) == 0: # no self node, return the predicted node
                                pred_val = bz_selected_nodes[0][1]
                            else:
                                start_log_probs, end_log_probs = self.single_span_head(selected_node_seq_embs[bz], selected_node_seq_masks[bz])
                                # start_log_probs, end_log_probs = self.single_span_head(token_representations[bz], attention_mask[bz])
                                
                                start_log_probs = start_log_probs.detach().cpu()
                                end_log_probs = end_log_probs.detach().cpu()

                                selected_node_word_idx, selected_node_words_list = self.get_node_seq_words(bz_selected_nodes)

                                pred_val = get_best_span(start_log_probs, end_log_probs, selected_node_word_idx, selected_node_words_list)

                        elif predict_head[bz] in [self.HEAD_CLASSES["COUNT"], self.HEAD_CLASSES["MULTI_SPAN"]] :  # Note how many we cannot get the count
                            other_node_vals = [node[1] for node in bz_selected_nodes if node[0] != TAG_SELF[0]]
                            self_nodes = [node for node in bz_selected_nodes if node[0] == TAG_SELF[0]]
                            # 1.% 空格
                            # 2.-
                            # 3.remove duplicated 
                            all_vals  = list()
                            # all_vals.extend(other_node_vals)
                            all_vals = other_node_vals

                            if len(self_nodes) > 0:
                                seq_token_tag_log_probs = self.sequence_tag_head(bz_selected_node_seq_embs, bz_selected_node_seq_masks)

                                selected_node_word_idx, selected_node_words_list = self.get_node_seq_words(bz_selected_nodes)

                                spans = get_span_from_tags(seq_token_tag_log_probs,selected_node_word_idx, selected_node_words_list)

                                all_vals = spans
                          
                            clean_vals = set()
                            for x in all_vals:
                                x = str(x)
                                x = x.replace(' %', '%')
                                x = x.replace('- ', '-')
                                if x == '' or x in clean_vals:
                                    continue
                                clean_vals.add(x)
                            
                            if predict_head[bz] == self.HEAD_CLASSES["COUNT"]:
                                pred_val = len(clean_vals)
                            else:
                                pred_val = list(clean_vals)

                output_dict["prediction"].append(pred_val)
                output_dict["answer"].append(answer_dict[bz]['answer'])
                output_dict["question_id"].append(question_ids[bz])

                self._metrics({**answer_dict[bz], 
                               "qid": question_ids[bz],
                               "gold_facts":facts[bz], 
                               "gold_consts":consts[bz],
                               "gold_seq":out_seq[bz],
                               "pred_seq": pred_seq,
                               "pred_nodes":bz_selected_node_vals},
                              pred_head=pred_head,
                              gold_head=gold_head,
                              pred_scale=pred_scale,
                              prediction=pred_val,
                              node_precision=output_dict['loss_sp']['node_precision'][bz],
                              node_recall=output_dict['loss_sp']['node_recall'][bz])

        output_dict['loss_sp']['avg_node_precision'] = np.mean(output_dict['loss_sp']['node_precision'])
        output_dict['loss_sp']['avg_node_recall'] =  np.mean(output_dict['loss_sp']['node_recall'])
      
        return output_dict

    def predict(self,
                 input_ids: torch.LongTensor,
                input_bbox:torch.LongTensor,
                attention_mask: torch.LongTensor,
                token_type_ids: torch.LongTensor,
                question_mask: torch.LongTensor,
                bbox_mask: torch.LongTensor,
                answer_dict,
                question_ids: List[str],
                facts,
                consts,
                out_len,
                out_seq,
                num_nodes,
                num_graphs,
                date_nodes,
                date_graphs,
                semantic_nodes,
                semantic_graphs,
                full_graphs,
                image: torch.LongTensor,
                position_ids: torch.LongTensor = None,
                mode='test',
                epoch=None,
                dataset='tatdqa') -> Dict[str, torch.Tensor]:

        self.dataset='tatdqa'
        output_dict = {}

        # bert embedding
        outputs = None

        outputs = self.lm(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            bbox=input_bbox,
                            image=image)

        token_representations = outputs[0][:,:self.max_input_len,:]

        # cls token output
        cls_output = token_representations[:, 0, :]

        batch_size = token_representations.shape[0]
        seq_len = token_representations.shape[1]
        hidden_size = token_representations.shape[-1]

        full_nodes = [n_nodes + d_nodes + s_nodes for n_nodes, d_nodes, s_nodes in list(zip(num_nodes, date_nodes, semantic_nodes))]
        question_node_idxes = [len(n_nodes) + len(d_nodes) for n_nodes, d_nodes in list(zip(num_nodes, date_nodes))]

        max_num_length = max([len(it) for it in num_nodes])
        max_date_length = max([len(it) for it in date_nodes])
        max_semantic_length = max([len(it) for it in semantic_nodes])
        max_full_length = max([len(it) for it in full_nodes])

        _, num_node_vals, num_node_embs, num_node_masks, num_node_labels, num_node_label_ords, num_node_seq_offsets, num_node_ans_offsets, _, _ = \
            self.get_node_val_and_embs(num_nodes, max_num_length, token_representations)
        _, date_node_vals, date_node_embs, date_node_masks, date_node_labels, date_node_label_ords, date_node_seq_offsets, date_node_ans_offsets, _, _ = \
            self.get_node_val_and_embs(date_nodes, max_date_length, token_representations)
        _, semantic_node_vals, semantic_node_embs, semantic_node_masks, semantic_node_labels, semantic_node_label_ords, semantic_node_seq_offsets, semantic_node_ans_offsets, _, _  = \
            self.get_node_val_and_embs(semantic_nodes, max_semantic_length, token_representations)
        full_node_tags, full_node_vals, full_node_embs, full_node_masks, full_node_labels, full_node_label_ords, full_node_seq_offsets, full_node_ans_offsets, final_node_token2word_idx, full_node_words = \
            self.get_node_val_and_embs(full_nodes, max_full_length, token_representations)

        '''1.1 number graph'''
        num_encoder_outputs = num_node_embs
        if max_num_length > 0:
            expand_num_graphs = self.expand_graphs(num_graphs, max_num_length)
            num_graph_tensor = torch.LongTensor(np.array(expand_num_graphs))
            if torch.cuda.is_available():
                num_graph_tensor = num_graph_tensor.cuda()
            num_encoder_outputs = self.num_encoder(num_node_embs, num_graph_tensor)

        ''' 1.2 date graph'''
        date_encoder_outputs = date_node_embs
        if max_date_length > 0:
            expand_date_graphs = self.expand_graphs(date_graphs, max_date_length)
            date_graph_tensor = torch.LongTensor(np.array(expand_date_graphs))
            if torch.cuda.is_available():
                date_graph_tensor = date_graph_tensor.cuda()
            date_encoder_outputs = self.date_encoder(date_node_embs, date_graph_tensor)

        ''' 1.3 semantic graph'''
        semantic_encoder_outputs = semantic_node_embs
        expand_semantic_graphs = self.expand_graphs(semantic_graphs, max_semantic_length)
        semantic_graph_tensor = torch.LongTensor(np.array(expand_semantic_graphs))
        if torch.cuda.is_available():
            semantic_graph_tensor = semantic_graph_tensor.cuda()

        semantic_encoder_outputs = self.semantic_encoder(semantic_node_embs, semantic_graph_tensor)

        '''2 To build full/hybrid graph '''
    
        full_node_reps = torch.zeros(batch_size, max_full_length, hidden_size)
        if torch.cuda.is_available():
            full_node_reps = full_node_reps.cuda()

        for bz in range(batch_size):
            num_len = int(sum(num_node_masks[bz]).item())
            date_len = int(sum(date_node_masks[bz]).item()) if len(date_node_masks[bz]) > 0  else 0
            semantic_len = int(sum(semantic_node_masks[bz]).item())
            full_node_reps[bz, 0:num_len, :] = num_encoder_outputs[bz, 0:num_len, :]
            full_node_reps[bz, num_len:num_len + date_len, :] = date_encoder_outputs[bz, 0:date_len, :]
            full_node_reps[bz, num_len + date_len:num_len + date_len + semantic_len, :] = semantic_encoder_outputs[bz, 0:semantic_len, :]

        expand_full_graphs = self.expand_graphs(full_graphs, max_full_length)
        full_graph_tensor = torch.LongTensor(np.array(expand_full_graphs))
        if torch.cuda.is_available():
            full_graph_tensor = full_graph_tensor.cuda()

        # b * num of nodes * H
        full_encoder_outputs = full_node_reps
        full_encoder_outputs = self.full_encoder(full_node_reps, full_graph_tensor)

        full_encoder_outputs = util.replace_masked_values(full_encoder_outputs, full_node_masks, 0)

        # B * H (Avg Pooling)
        # full_graph_embs = cls_output
        full_graph_embs = torch.mean(full_encoder_outputs, dim=1, keepdim=False)

        question_reprs = torch.stack([full_encoder_outputs[bz,idx,:].squeeze() for bz, idx in enumerate(question_node_idxes)])

        # B * max_node_len * 1
        # q_node_sim_scores = self.dot_product_scores(question_reprs, full_encoder_outputs)
        # node_prediction_log_probs = self.node_predictor(q_node_sim_scores, full_node_masks.squeeze(-1))
        # sim_node_masks = full_node_masks[bz:idx] = 0 for bz, idx in enumerate(question_node_idxes)
        # B * max_node_len * 2
        node_prediction_log_probs = self.node_predictor(full_encoder_outputs, full_node_masks.squeeze(-1))

        # B * max_node_len
        node_predictions = torch.argmax(node_prediction_log_probs, dim=-1).cpu().detach()
        output_dict['loss_sp'] = {}
        output_dict['head_acc'] = 1
        output_dict['scale_acc'] = 1

        output_dict['loss_sp']['node_precision'] = []
        output_dict['loss_sp']['node_recall'] = []
        output_dict['loss_sp']['node_metric_detail'] = []

        full_model_selected_nodes = []

        for bz in range(batch_size):
            one_bz_selected_nodes = []
            for idx, (cls, tag, val, emb, label, ord, seq_offset, ans_offsets, mask, token2word_idx, words) in enumerate(list(
                    zip(node_predictions[bz], full_node_tags[bz], full_node_vals[bz], full_encoder_outputs[bz], full_node_labels[bz], full_node_label_ords[bz],
                        full_node_seq_offsets[bz], full_node_ans_offsets[bz], full_node_masks[bz], final_node_token2word_idx[bz], full_node_words[bz]))):

                if   mask == 1 and cls == 1: # mode == 'train' and
                    one_bz_selected_nodes.append((tag, val, emb, seq_offset, ans_offsets, token2word_idx, words, ord, label))
            
            full_model_selected_nodes.append(one_bz_selected_nodes)
        
        full_selected_nodes = []
        full_selected_num_and_date_vals = []
        full_selected_num_and_date_embs = []
        for bz in range(batch_size):
            one_bz_q_repr = question_reprs[bz]
            bz_selected_nodes = full_model_selected_nodes[bz]
        
            one_bz_sorted_selected_nodes = sorted(bz_selected_nodes, key=lambda it: torch.dot(it[2], one_bz_q_repr))[:self.max_copy_num] # bz_selected_nodes[:self.max_copy_num] #  

            full_selected_num_and_date_vals.append([it[1] for it in one_bz_sorted_selected_nodes if it[0] in [TAG_NUMBER[0], TAG_DATE[0]]])
            full_selected_num_and_date_embs.append([it[2] for it in one_bz_sorted_selected_nodes if it[0] in [TAG_NUMBER[0], TAG_DATE[0]]])
            full_selected_nodes.append(one_bz_sorted_selected_nodes)

       
        scale_predict_logits = self.scale_predictor(full_graph_embs)
        answer_head_logits = self.head_predictor(full_graph_embs)
        answer_head_log_probs = F.log_softmax(answer_head_logits, -1)
        scale_log_probs = F.log_softmax(scale_predict_logits, -1)

        scale_prediction = torch.argmax(scale_log_probs, dim=-1).cpu().detach()

        predict_head = torch.argmax(answer_head_log_probs, dim=-1).cpu().detach()

        output_dict["head"] = predict_head
        output_dict["scale"] = [SCALE[int(it)] for it in scale_prediction]

        selected_node_seq_embs = []
        selected_node_seq_masks = []
        for bz in range(batch_size):

            bz_select_nodes = full_selected_nodes[bz]

            bz_node_embs = torch.zeros(seq_len, hidden_size)
            default_mask_val = 0
            if not self.node_mask:
                default_mask_val = 1
                
            bz_node_masks = self.offset2mask(offset=[0, seq_len],seq_len=seq_len, val=default_mask_val)
            if torch.cuda.is_available():
                bz_node_embs = bz_node_embs.cuda()

            if self.node_mask:
                for node in bz_select_nodes:
                    tag, val, emb, seq_offset, ans_offsets, token2word_idx, words, ord, label = node
                
                    s = seq_offset[0]
                    e = seq_offset[1]
                    bz_node_embs[s:e, :] = emb.expand(e - s, -1)
                    bz_node_masks[s:e, :] = 1
            selected_node_seq_embs.append(bz_node_embs)
            selected_node_seq_masks.append(bz_node_masks)

        selected_node_seq_embs = torch.stack(selected_node_seq_embs)
        selected_node_seq_masks = torch.stack(selected_node_seq_masks).squeeze(-1)

        selected_node_seq_embs = torch.cat([token_representations, selected_node_seq_embs], 2)# cat
        # selected_node_seq_embs, _ = self.node_attention_layer(query=selected_node_seq_embs, value=token_representations)

        # HEAD_CLASSES_ = {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "COUNT": 3, "ARITHMETIC": 4}

        # seq_repr,  # B * S * H
        # question_repr,  # B * H
        # all_num_repr,  # B * num_size * H
        # copy_nums,  # B * num_size
        # attention_mask,  # B * S
        # out_len,  # B * N
        # out_seq):  # B * M
        # gold_nodes_embs =  gold_nodes_embeddings
        # facts,
        # assert len(gold_num_and_date_vals) == len(facts)


        # context_output = util.replace_masked_values(token_representations, attention_mask.unsqueeze(-1), 0)
        # using the node embedding as the tree input
        context_output = util.replace_masked_values(full_encoder_outputs, full_node_masks, 0)
        context_output = context_output.transpose(1, 0)
        # attention_mask = full_node_masks.squeeze(-1)
       
        output_dict["question_id"] = []
        output_dict["prediction"] = []
        output_dict["answer"] = []

        # Evaluation Part 
        with torch.no_grad():
            for bz in range(batch_size):
                pred_val = ""
                pred_head = ""
                pred_scale = ""
                gold_head = ""
                pred_seq = ""

                bz_selected_nodes = full_selected_nodes[bz]

                bz_selected_node_vals = [it[1] for it in bz_selected_nodes]

                pred_head = predict_head[bz]
                pred_scale = SCALE[int(scale_prediction[bz])]

                if bz_selected_nodes:
                    if predict_head[bz] == self.HEAD_CLASSES["ARITHMETIC"]:
                        try:
                            selected_num_and_date_vals = [it[1] for it in bz_selected_nodes if it[0] in [TAG_NUMBER[0], TAG_DATE[0]]]
                            selected_num_and_date_embs = [it[2] for it in bz_selected_nodes if it[0] in [TAG_NUMBER[0], TAG_DATE[0]]]
                            if not selected_num_and_date_embs:
                                pred_val = None
                            else:
                                pred_seq = self.train_tree.predict(context_output[:,bz,:].unsqueeze(1),
                                                                   question_reprs[bz].unsqueeze(0), # full_graph_embs[bz].unsqueeze(0),
                                                                   selected_num_and_date_vals,
                                                                   selected_num_and_date_embs,
                                                                   full_node_masks.squeeze(-1)[bz].unsqueeze(0))

                                pred_val = self.compute_prefix_tree_result(pred_seq, selected_num_and_date_vals)
                        except SyntaxError as e:
                            print(f"tree prediction has syntax error:{e}")
                            pred_val = None
                        except Exception as e:
                            print(f"tree prediction has unexpected error:{e}")
                            pred_val = None
                    else:
                        ''''''
                        bz_selected_node_seq_embs = selected_node_seq_embs[bz].unsqueeze(0)
                        bz_selected_node_seq_masks = selected_node_seq_masks[bz].unsqueeze(0)
                        if predict_head[bz] == self.HEAD_CLASSES["SPAN-TEXT"]:
                            self_nodes = [node for node in bz_selected_nodes if node[0] == TAG_SELF[0]]
                            if len(self_nodes) == 0: # no self node, return the predicted node
                                pred_val = bz_selected_nodes[0][1]
                            else:
                                start_log_probs, end_log_probs = self.single_span_head(selected_node_seq_embs[bz], selected_node_seq_masks[bz])
                                # start_log_probs, end_log_probs = self.single_span_head(token_representations[bz], attention_mask[bz])
                                
                                start_log_probs = start_log_probs.detach().cpu()
                                end_log_probs = end_log_probs.detach().cpu()

                                selected_node_word_idx, selected_node_words_list = self.get_node_seq_words(bz_selected_nodes)

                                pred_val = get_best_span(start_log_probs, end_log_probs, selected_node_word_idx, selected_node_words_list)

                        else:  # Note how many we cannot get the count
                            other_node_vals = [node[1] for node in bz_selected_nodes if node[0] != TAG_SELF[0]]
                            self_nodes = [node for node in bz_selected_nodes if node[0] == TAG_SELF[0]]
                            # 1.% 空格
                            # 2.-
                            # 3.remove duplicated 
                            all_vals  = list()
                            # all_vals.extend(other_node_vals)
                            all_vals = other_node_vals

                            if len(self_nodes) > 0:
                                seq_token_tag_log_probs = self.sequence_tag_head(bz_selected_node_seq_embs, bz_selected_node_seq_masks)

                                selected_node_word_idx, selected_node_words_list = self.get_node_seq_words(bz_selected_nodes)

                                spans = get_span_from_tags(seq_token_tag_log_probs,selected_node_word_idx, selected_node_words_list)

                                all_vals = spans
                            
                            clean_vals = set()
                            for x in all_vals:
                                x = str(x)
                                x = x.replace(' %', '%')
                                x = x.replace('- ', '-')
                                if x == '' or x in clean_vals:
                                    continue
                                clean_vals.add(x)

                            bz_selected_node_vals = list(clean_vals)
                            if predict_head[bz] == self.HEAD_CLASSES["COUNT"]:
                                pred_val = len(clean_vals)
                            else:
                                pred_val = list(clean_vals)

                output_dict["prediction"].append(pred_val)
                # output_dict["answer"].append(answer_dict[bz]['answer'])
                output_dict["question_id"].append(question_ids[bz])

                self._metrics({**answer_dict[bz], 
                               "qid": question_ids[bz],
                               "pred_seq": pred_seq,
                               "pred_nodes":bz_selected_node_vals},
                              pred_head=pred_head,
                              pred_scale=pred_scale,
                              prediction=pred_val)

      
        return output_dict

    def reset(self):
        self._metrics.reset()
        self.acc = [0, 0]

    def get_metrics(self, logger=None, reset: bool = False) -> Dict[str, float]:
        detail_em_pivot, detail_f1_pivot, detail_em_type, detail_f1_type = self._metrics.get_detail_metric()
        # raw_detail = self._metrics.get_raw_pivot_table()
        exact_match, f1_score, scale_score, head_score, node_precision, node_recall, node_f1 = self._metrics.get_overall_metric(reset)

        if logger is not None:
            # logger.info(f"raw matrix:{raw_detail}\r\n")
            logger.info(f"detail em:{detail_em_pivot}\r\n")
            logger.info(f"detail f1:{detail_f1_pivot}\r\n")
            logger.info(f"global em:{exact_match}\r\n")
            logger.info(f"global f1:{f1_score}\r\n")
            logger.info(f"global scale:{scale_score}\r\n")
            logger.info(f"global head:{head_score}\r\n")

            logger.info(f"detail em_type:{detail_em_type}\r\n")
            logger.info(f"detail f1_type:{detail_f1_type}\r\n")
        else:
            # print(f"raw matrix:{raw_detail}\r\n")
            print(f"detail em:{detail_em_pivot}\r\n")
            print(f"detail f1:{detail_f1_pivot}\r\n")
            print(f"global em:{exact_match}\r\n")
            print(f"global f1:{f1_score}\r\n")
            print(f"global scale:{scale_score}\r\n")
            print(f"global head:{head_score}\r\n")

            print(f"detail em_type:{detail_em_type}\r\n")
            print(f"detail f1_type:{detail_f1_type}\r\n")

        return {'em': exact_match,
                'f1': f1_score,
                "node_precision":node_precision,
                "node_recall": node_recall,
                "node_f1": node_f1,
                "tree_acc": self.acc[0] / max(self.acc[1], 1e-5),
                "scale": scale_score,
                "head_score": head_score}

    def get_metrics_predict(self, logger=None, reset: bool = False):
        detail_em, detail_f1, detail_em_type, detail_f1_type= self._metrics.get_detail_metric()
        raw_detail = self._metrics.get_raw_pivot_table()
        exact_match, f1_score, scale_score, head_score,_,_,_ = self._metrics.get_overall_metric(reset)

        if logger is not None:
            logger.info(f"raw matrix:{raw_detail}\r\n")
            logger.info(f"detail em:{detail_em}\r\n")
            logger.info(f"detail f1:{detail_f1}\r\n")
            logger.info(f"global em:{exact_match}\r\n")
            logger.info(f"global f1:{f1_score}\r\n")
            logger.info(f"global scale:{scale_score}\r\n")
        else:
            print(f"raw matrix:{raw_detail}\r\n")
            print(f"detail em:{detail_em}\r\n")
            print(f"detail f1:{detail_f1}\r\n")
            print(f"global em:{exact_match}\r\n")
            print(f"global f1:{f1_score}\r\n")
            print(f"global scale:{scale_score}\r\n")
            print(f"global head:{head_score}\r\n")
        score_dict = {'em': exact_match, 'f1': f1_score, "scale": scale_score, "head": head_score}

        return raw_detail, detail_em, detail_f1, score_dict

    def get_df(self):
        raws = self._metrics.get_raw()
        return pd.DataFrame(raws)


def reduce_mean_index(values, index, max_length=561, name="index_reduce_mean"):
    return _index_reduce(values, index, max_length, "mean", name)


"""
def _index_reduce(values, index, max_length, index_reduce_fn, name):
    flat_index, num_index = flatten_index(index, max_length)
    bsz = values.shape[0]
    seq_len = values.shape[1]
    flat_values = values.reshape(bsz * seq_len)
    index_means = scatter(
        src=flat_values,
        index=flat_index.type(torch.long),
        dim=0,
        dim_size=num_index,
        reduce=index_reduce_fn,
    )
    output_values = index_means.view(bsz, -1)
    return output_values
"""


def _index_reduce(values, index, max_length, index_reduce_fn, name):
    index_means = scatter(
        src=values,
        index=index.type(torch.long),
        dim=0,
        dim_size=max_length,
        reduce=index_reduce_fn,
    )
    output_values = index_means.view(-1)
    return output_values


def flatten_index(index, max_length=561, name="index_flatten"):
    batch_size = index.shape[0]
    offset = torch.arange(start=0, end=batch_size, device=index.device) * max_length
    offset = offset.view(batch_size, 1)
    return (index + offset).view(-1), batch_size * max_length