import os
import pickle
import random
from numpy import int32
import torch

PAD_token = 0


def pad_seq(seq, seq_len, max_length):
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq


class GNNBatchGen(object):
    def __init__(self, args, data_mode, out_lang, encoder='roberta', dataset='tat-dqa', model_name='etr', answer_type='all'):
        dpath = f"{model_name}_{encoder}_{dataset}_cached_{data_mode}.pkl"
        self.is_train = data_mode == "train"
        self.args = args
        print(os.getcwd().replace('\\','/'))
        with open(os.path.join(args.data_dir, dpath), 'rb') as f:

            print("Load data from {}.".format(dpath))
            data = pickle.load(f)
        self.lang = out_lang
        all_data = []
        for item in data:
            input_ids = torch.from_numpy(item["input_ids"]).type(torch.int32)
            bbox=torch.from_numpy(item["bbox"]).type(torch.int32)
            attention_mask = torch.from_numpy(item["attention_mask"]).type(torch.int32)
            token_type_ids = torch.from_numpy(item["token_type_ids"]).type(torch.int32) #  torch.zeros_like(input_ids)
            question_mask = torch.from_numpy(item["question_mask"])
            bbox_mask = torch.from_numpy(item["bbox_mask"])
            input_bbox_orders = torch.from_numpy(item["input_bbox_orders"])
            # question_metas = item['question_metas']
            # bbox_metas = item['block_metas']
            # paragraph_metas = item['paragraph_metas']
            answer_dict = item['answer_dict']
            question_id = item["question_id"]
            facts = item["facts"]
            consts = item["consts"]
            answer_mapping=item["answer_mapping"]
            out_seq = item["out_seq"]
            num_nodes = item["num_nodes"]
            num_graph = item["num_graph"]
            date_nodes = item["date_nodes"]
            date_graph = item["date_graph"]
            semantic_nodes = item["semantic_nodes"]
            semantic_range=item["semantic_range"]
            semantic_graph = item["semantic_graph"]
            full_graph = item["full_graph"]
            image=torch.from_numpy(item["image"])

            if answer_type != 'all' and answer_dict['answer_type'] != answer_type:
                continue

            all_data.append((input_ids,bbox, attention_mask, token_type_ids, question_mask,  bbox_mask, answer_dict,
                             question_id,facts,consts,out_seq, num_nodes, num_graph, date_nodes, date_graph, semantic_nodes, semantic_range, semantic_graph, full_graph,image))
        print("Load data size {}.".format(len(all_data)))
        self.data = GNNBatchGen.make_batches(all_data, args.batch_size if self.is_train else args.eval_batch_size,
                                             self.is_train)
        self.offset = 0

    @staticmethod
    def make_batches(data, batch_size=32, is_train=True):
        if is_train:
            random.shuffle(data)
        if is_train:
            return [
                data[i: i + batch_size] if i + batch_size < len(data) else data[i:] + data[
                                                                                      :i + batch_size - len(data)]
                for i in range(0, len(data), batch_size)]
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def tran_seq(self, out_seq):
        res = []
        for word in out_seq:
            if len(word) == 0:
                continue
            if word in self.lang.word2index:
                res.append(self.lang.word2index[word])
            else:
                res.append(self.lang.word2index["UNK"])
        return res

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
            for i in range(len(self.data)):
                random.shuffle(self.data[i])
        self.offset = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]
            self.offset += 1
            input_ids_batch,bbox_batch, attention_mask_batch, token_type_ids_batch, question_mask_batch, bbox_mask_batch,answer_dict_batch, question_id_batch, \
            facts_batch, consts_batch, out_seq_batch, num_nodes_batch, num_graph_batch, date_nodes_batch, date_graph_batch, \
            semantic_nodes_batch, semantic_range_batch, semantic_graph_batch, full_graph_batch,image_batch = zip(*batch)
            max_input_len = self.args.max_input_len
            bsz = len(batch)
            input_ids = torch.LongTensor(bsz, max_input_len ) 
            input_bbox= torch.LongTensor(bsz, max_input_len ,  4)
            attention_mask = torch.LongTensor(bsz, max_input_len ) 
            token_type_ids =  torch.LongTensor(bsz, max_input_len ) .fill_(0)
            image=torch.LongTensor(bsz, 3,224,224)
            question_mask = torch.LongTensor(bsz, max_input_len ) 
            bbox_mask = torch.LongTensor(bsz, max_input_len) 
            position_ids = torch.LongTensor(bsz, max_input_len) 

            # input_bbox_orders = torch.LongTensor(bsz, max_input_len ) 

            out_seq = []

            out_seq_len_batch = [len(seq) for seq in out_seq_batch]
            max_out_seq_len = max(out_seq_len_batch)

            for i in range(bsz):
                input_ids[i] = input_ids_batch[i]
                input_bbox[i]=bbox_batch[i]
                attention_mask[i] = attention_mask_batch[i]
                token_type_ids[i] = (token_type_ids_batch[i]  > 0).int()
                question_mask[i] = question_mask_batch[i]
                bbox_mask[i] = bbox_mask_batch[i]
                image[i]=image_batch[i]
                position_ids[i] = torch.arange(max_input_len).unsqueeze(0)

                out_seq.append(pad_seq(self.tran_seq(out_seq_batch[i]), out_seq_len_batch[i], max_out_seq_len))


            out_batch = {
                "input_ids": input_ids,
                "input_bbox":input_bbox,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "question_mask": question_mask,
                "bbox_mask": bbox_mask,
                # "question_metas": question_metas_batch,
                # "table_metas": table_metas_batch,
                # "paragraph_metas": paragraph_metas_batch,
                "answer_dict": answer_dict_batch,
                "question_ids": question_id_batch,
                "facts": facts_batch,
                "consts": consts_batch,
                "out_len":out_seq_len_batch,
                "out_seq": out_seq,
                "num_nodes": num_nodes_batch,
                "num_graphs": num_graph_batch,
                "date_nodes": date_nodes_batch,
                "date_graphs": date_graph_batch,
                "semantic_nodes": semantic_nodes_batch,
                "semantic_graphs": semantic_graph_batch,
                "full_graphs": full_graph_batch,
                "position_ids": position_ids,
                "image":image
            }

            if self.args.cuda:
                for k in out_batch.keys():
                    if isinstance(out_batch[k], torch.Tensor):
                        out_batch[k] = out_batch[k].cuda()
            yield out_batch