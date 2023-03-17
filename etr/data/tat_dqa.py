import json
from pathlib import Path
from tkinter import Image
from tqdm import tqdm


import sys

from etr.model.tree_model import compute_prefix_expression
from PIL import Image
from .dataset_util import *
from .lang_utils import *
from .tagger import Tagger
np.set_printoptions(threshold=np.inf)
from typing import Dict
import torch
from .file_utils import is_scatter_available
from .data_util import *
from .data_tools import *
from PIL import Image
from etr.model.layout_model import normalize_bbox, extract_feature_from_image
from .graph_construction import *
# soft dependency
if is_scatter_available():
    from torch_scatter import scatter

def get_head_class(answer_type: str, HEAD_CLASSES):
    Head_class = None
    if answer_type == "span":
        Head_class = HEAD_CLASSES["SPAN-TEXT"]
    elif answer_type == "multi-span":
        Head_class = HEAD_CLASSES["MULTI_SPAN"]
    elif answer_type == "count":
        Head_class = HEAD_CLASSES["COUNT"]
    else:
        Head_class = HEAD_CLASSES["ARITHMETIC"]
    return Head_class



"""
instance format:
input_ids: np.array[1, 512]. The input ids.
attention_mask: np.array[1, 512]. The attention_mask to denote whether a id is real or padded.
token_type_ids: np.array[1, 512, 3]. 
    The special tokens needed by tapas within following orders: segment_ids, column_ids, row_ids.
tags_label: np.array[1, 512]. The tag ground truth indicate whether this id is part of the answer.
paragraph_mask: np.array[1, 512]. 1 for ids belongs to paragraph, 0 for others
paragraph_word_piece_mask: np.array[1, 512]. 0 for sub-token, 1 for non-sub-token or the start of sub-token
paragraph_number_value: np.array[1, 512]. nan for no-numerical words and number for numerical words extract from current word. i.e.: $460 -> 460
table_number_value: np.array[1, max_num_columns*max_num_rows]. Definition is the same as above.
paragraph_number_mask: np.array[1, 512]. 0 for non numerical token and 1 for numerical token.
table_number_mask: np.array[1, max_num_columns*max_num_rows]. 0 for non numerical token and 1 for numerical token.
paragraph_index: np.array[1, 512], used to apply token-lv reduce-mean after getting sequence_output
number_order_label: int. The operator calculating order.
operator_label:  int. The operator ground truth.
scale_label: int. The scale ground truth.
answer: str. The answer used to calculate metrics.
"""


def string_tokenizer(string: str, tokenizer) -> List[int]:
    if not string:
        return []
    tokens = []
    prev_is_whitespace = True
    for i, c in enumerate(string):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
        elif c in ["-", "–", "~"]:
            tokens.append(c)
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(c)
            else:
                tokens[-1] += c
            prev_is_whitespace = False  # char_to_word_offset.append(len(tokens) - 1)

    split_tokens = []
    for i, token in enumerate(tokens):
        if i != 0:
            sub_tokens = tokenizer._tokenize(" " + token)
        else:
            sub_tokens = tokenizer._tokenize(token)

        for sub_token in sub_tokens:
            split_tokens.append(sub_token)

    ids = tokenizer.convert_tokens_to_ids(split_tokens)
    return ids


def question_tokenizer(question_text, tokenizer):
    return string_tokenizer(question_text, tokenizer)


class TATDQAReader(object):
    def __init__(self,
                 tokenizer,
                #  passage_length_limit: int = None,
                 question_length_limit: int = None,
                 sep="<s>",
                 mode='train',
                 max_pieces = 1024):
        self.max_pieces = max_pieces
        self.tokenizer = tokenizer
        # self.passage_length_limit = passage_length_limit
        self.question_length_limit = question_length_limit
        self.sep_start = self.tokenizer._convert_token_to_id(sep)
        self.sep_end = self.tokenizer._convert_token_to_id(sep)
        self.tokenizer._tokenize("Feb 2 Nov")
        # print(self.tokenizer._tokenize("1,382,818"))
        self.skip_count = 0
        self.generate_nums = []
        self.generate_nums_dict = {}
        self.copy_num = 0
        self.HEAD_CLASSES = HEAD_CLASSES_
        self.out_lang = Lang()
        self.out_lang.add_operators(['+', '-', '*', '/'])
        self.consts =[str(x)  for x in [-100, -2, -1,0, 1,2,3,4,5,6,7,8,9,10,11, 12, 100,1000,10000, 100000, 1000000, 10000000, 100000000, 1000000000, 2017, 2018, 2019]]
        self.out_lang.add_const(self.consts)
        self.mode = mode 
        self.error = []

    def build_lang(self, generate_nums_bound: int = 10):
            temp_g = []
            for g in self.generate_nums:
                if self.generate_nums_dict[g] >= generate_nums_bound:
                    temp_g.append(g)
            self.out_lang.build_output_lang_for_tree(self.copy_num)

    def _make_instance(self,
                       input_ids,
                       input_bboxes,
                       attention_mask,
                       token_type_ids,
                       question_mask,
                       bbox_mask,
                       input_bbox_orders,
                       question_metas,
                       block_metas,
                       answer_dict,
                       question_id,
                       facts,
                       consts,
                       answer_mapping,
                       out_seq,
                       num_nodes,
                       num_graph,
                       date_nodes,
                       date_graph,
                       semantic_nodes,
                       semantic_range,
                       semantic_graph,
                       full_graph,
                       image_pixels=None):
        return {
            "input_ids": np.array(input_ids),
            "bbox": np.array(input_bboxes),
            "attention_mask": np.array(attention_mask),
            "token_type_ids": np.array(token_type_ids),
            "question_mask": np.array(question_mask),
            "bbox_mask": np.array(bbox_mask),
            "input_bbox_orders": np.array(input_bbox_orders),
            "question_metas": np.array(question_metas),
            "block_metas": np.array(block_metas),
            "answer_dict": answer_dict,
            "question_id": question_id,
            "facts": facts,
            "consts": consts,
            "answer_mapping": answer_mapping,
            "out_seq": out_seq,
            "num_nodes": num_nodes,
            "num_graph": num_graph,
            "date_nodes": date_nodes,
            "date_graph": date_graph,
            'semantic_nodes': semantic_nodes,
            "semantic_range": semantic_range,
            "semantic_graph": semantic_graph,
            "full_graph": full_graph,
            "image": np.array(image_pixels),
        }
    def _match_gold_fact(self, text_val, text_offsets, gold_offsets, method):
        for idx, item in enumerate(gold_offsets):
            fact, one_gold_offset, order = item
            val_match = offset_match = match = False
            if text_val == fact:
                val_match = True
            else:
                text_val_num = to_number(text_val)
                fact_num = to_number(fact)
                if fact_num and text_val_num and abs(text_val_num) == abs(fact_num):
                    val_match = True

            if overlap1d(one_gold_offset, text_offsets):
                offset_match = True

            if method == 'all':
                match = val_match and offset_match
            elif method == 'val':
                match = val_match
            else:
                match = offset_match

            if match:
                return match, order, item

        return False, -1, None

    # 确定该node 是否是target
    def prepare_tag_offsets(self,text, tokens, tag, qid, head_class, token2text_idx, text_tag_offsets, gold_text_offsets=[], source='table' ):
        result = {}
        gold_offsets_copy = copy.deepcopy(gold_text_offsets)
        # to mapping the token with the text

        for tag_name, values in text_tag_offsets.items():
            tag_values = []
            for val, text_offsets in values:
                token_start_idx = len(token2text_idx)
                token_end_idx = -1
                for token_idx, text_idx in enumerate(token2text_idx):
                    if text_idx >= text_offsets[0] and text_idx < text_offsets[1]:
                        if token_start_idx > token_idx:
                            token_start_idx = token_idx
                        if token_end_idx < token_idx:
                            token_end_idx = token_idx

                if self.mode != 'test' and token_end_idx < token_start_idx:
                    raise ValueError(f'cannot find token {val}: {text_offsets} {qid}')

                is_target, order, item = self._match_gold_fact(val, text_offsets, gold_offsets_copy, method='all')
                if is_target:
                    gold_offsets_copy.remove(item)
    
                if not is_target:
                    order = -1

                tag_values.append({'value': val, 'is_target': int(is_target), 'order': order, 'text_offsets': text_offsets, 'token_offsets': [token_start_idx, token_end_idx + 1]})
            result[tag_name] = tag_values

        # to add the rest
        result[TAG_SELF[0]] = {}
        if gold_offsets_copy:
            result[TAG_SELF[0]] = {'is_target': 1, 'self_answer_offsets':[]}
            for fact, one_gold_offset, order in gold_offsets_copy:
                if self.mode != 'test' and head_class == self.HEAD_CLASSES['ARITHMETIC']:
                    raise ValueError(f'cannot find arithmetic {qid}  {source}  gold {fact} : {one_gold_offset}: {order}  ')
                token_start_idx = len(token2text_idx)
                token_end_idx = -1
                for token_idx, text_idx in enumerate(token2text_idx):
                    if text_idx >= one_gold_offset[0] and text_idx < one_gold_offset[1]:
                        if token_start_idx > token_idx:
                            token_start_idx = token_idx
                        if token_end_idx < token_idx:
                            token_end_idx = token_idx

                if self.mode != 'test' and token_end_idx < token_start_idx:
                    raise ValueError(f'cannot find offset in SELF {head_class} {source} {qid} {fact}: {gold_offsets_copy}')
                result[TAG_SELF[0]]['self_answer_offsets'].append((fact, one_gold_offset, [token_start_idx, token_end_idx+1], order)) # 当选中自己为target时与其他普通的node是有差别的

        return result


    def sort_block_map(self, question, block_map, gold_uuids=[]):
        new_bbox_map= {}
        # answer_uuid_idx_mapping = defaultdict(list)
        # for one in gold_uuids:
        #     for uuid, idx in one.items():
        #         answer_uuid_idx_mapping[uuid].append(idx)
        sorted_uuids = get_sorted_block_uuids(question, block_map)
        
        # add gold uuid first if it is training mode 
        # if self.mode == 'train':
        #     for uuid in gold_uuids:
        #         new_bbox_map.update({uuid:block_map[uuid]})

        for uuid in sorted_uuids:
            # if self.mode == 'train' and uuid in gold_uuids:
            #     continue
            new_bbox_map.update({uuid:block_map[uuid]})
        return new_bbox_map

    def _to_instance(self,
                     question: str,
                     block_map: Dict,
                     answer_from: str,
                     answer_type: str,
                     answer,
                     answer_mapping: Dict,
                     scale: str,
                     question_id: str,
                     derivation,
                     facts,
                     image:Image=None):

        ori_facts = copy.deepcopy(facts)
        if question_id=="a792d5b1282e259fec9d9cfaa9aca07e":
            print('1')
        if answer_type == 'span':
            answer_mapping = answer_mapping[:1]
        mapping_facts = []
        try:
            mapping_facts = get_facts(block_map, answer_mapping)
        except ValueError as e:
            if self.mode != 'test':
                raise e
        # print(f"qid:{question_id}, answer_type: {answer_type}, ori facts{ori_facts}, mapping facts: {mapping_facts}")
        
        facts = mapping_facts
        # facts = ori_facts
        question_text = question.strip()
        # get head label
        head_class = get_head_class(answer_type, self.HEAD_CLASSES)

        if self.mode != 'test' and head_class is None:
            raise ValueError(f'exception: no class {question_id}')

        out_seq = ''
        consts=[]
        out_dict={}
        if head_class == self.HEAD_CLASSES['ARITHMETIC']:
            init_out_seq, nums, para_nums, para_pos = get_num_pos_block_outseq(block_map,
                                                                              answer_mapping,
                                                                              derivation)
            # Adjust the out seq based on the [facts]. 
            # to remove % and negative number
            # print(f"init out seq:{init_out_seq}")

            ori_out_seq = []
            for x in init_out_seq:
                if x in ['+','-', '*', '/']:
                    ori_out_seq.append(x)
                else:
                    z = simple_extract_one_num_from_str(x)
                    ori_out_seq.append(str(z))

            ans = compute_prefix_expression(ori_out_seq)
            if ans is None and self.mode != 'test':
                 raise ValueError(f'answer none: {question_id}, facts:{facts}, ori_out_seq:{ori_out_seq}, init_out_seq:{init_out_seq} ')
            
            if round(ans, 2) != answer:
               if round(ans * 100, 2) == answer and scale == 'percent':
                    ori_out_seq = ['*'] + ori_out_seq + ['100']
                    ans = compute_prefix_expression(ori_out_seq) 
                    # assert round(ans, 2) == answer
               elif round(ans, 2) * -1 == answer:
                    ori_out_seq = ['*'] + ori_out_seq + ['-1']
               else:
                    if self.mode != 'test':
                        raise ValueError(f'answer error: {question_id}, facts:{facts}, ori_out_seq:{ori_out_seq}, init_out_seq:{init_out_seq} ')
            
            facts = convert_fact(facts)
            # for fa
         
            out_seq, consts = tag(ori_out_seq, facts)

            for c in consts:
                if self.mode != 'test' and c not in self.consts:
                    raise ValueError(f'cannot find value in facts and consts: {question_id}, facts{facts}, value:{consts} ')

            self.copy_num = max(self.copy_num, len(facts))

            # set consts NOT add consts now
            # self.out_lang.add_const(consts)
        count_label = 0
        if head_class == self.HEAD_CLASSES['COUNT']:
            count_label = int(answer)
            
        # to get golden info for training
        gold_question_offsets = []
        gold_block_offsets = defaultdict(list)
        gold_orders = []
        gold_offsets = defaultdict(list)
        gold_uuids = []
        for order, (fact, one_mapping) in enumerate(list(zip(facts, answer_mapping))):
            key = list(one_mapping.keys())[0]
            gold_offsets = one_mapping[key]
            # if out_dict:
            #     order=out_dict[fact]
            gold_block_offsets[key].append((fact, gold_offsets, order))
            if key not in gold_uuids:
                gold_uuids.append(key)

        question_tag_offsets, question_tag_codes = Tagger.convert(question_text)

        question_ids, question_tokens, question_token_tags, question_token2text_idx, question_token2word_idx, question_words = text_tokenize_dqa(question_text, question_tag_codes, self.tokenizer)

        question_offsets = self.prepare_tag_offsets(question_text,question_tokens, question_tag_codes, question_id, head_class, question_token2text_idx, question_tag_offsets, gold_question_offsets, 'question')

        question_metas = (question_ids, question_tokens, question_token_tags, question_offsets, question_token2word_idx, question_words)

        block_metas=[]
        # new_block_map={}
        new_block_map = self.sort_block_map(question, block_map, gold_uuids)
        word_bboxes=[]
        block_token_bboxes=[]
        for block in new_block_map.items():
            one_gold_block_offsets = []
            block_order=block[1]['order']
            block_id= block[1]['uuid']
            block_text= block[1]['text']
            block_bbox= block[1]['words']['bbox_list']
            ori_block_words =block[1]['words']['word_list']
            if block_id in gold_block_offsets:
                one_gold_block_offsets = gold_block_offsets[block_id]
            block_tag_offsets, block_tag_codes = Tagger.convert(block_text)
            block_ids, block_tokens, block_token_tags, block_token2text_idx, block_token2word_idx,word_bboxes,block_token_bboxes, block_words = block_tokenize_dqa(ori_block_words,block_text, block_tag_codes,block_bbox,self.tokenizer)

            block_offsets = self.prepare_tag_offsets(block_text,block_tokens, block_tag_codes ,question_id, head_class, block_token2text_idx, block_tag_offsets, one_gold_block_offsets, 'bbox')

            block_metas.append((block_order, block_ids, block_tokens, block_token_tags, block_offsets, block_token2word_idx, block_words,word_bboxes,block_token_bboxes,block[1]['bbox']))

        input_ids,input_bboxes, attention_mask, token_type_ids, question_mask, bbox_mask, input_bbox_orders, input_tags, question_metas, block_metas=concat_dqa(question_metas, block_metas,
                                                                                                                       self.sep_start, self.sep_end, self.max_pieces, self.mode)
        if self.mode != 'test':
            if answer_type == 'arithmetic' and len(facts) != (str(question_metas) + str(block_metas)).count("'is_target': 1"):
                raise ValueError(f'The target node is not included in the input sequence in {question_id} for arithmetic type')
            if (str(question_metas) + str(block_metas)).count("'is_target': 1") == 0:
                raise ValueError(f'Cannot find any target node: {question_id}e')
        # To build graph
        '''1 num graph'''
        question_nodes = questionMeta2nodes_dqa(question_metas)
        bbox_nodes, bbox_ranges = bboxMeta2nodes(block_metas)
        if self.mode != 'test' and (str(question_nodes) + str(bbox_nodes)).count("'self_answer_offsets': []"):
            raise ValueError('can not find answer_offsets')

        assert len(bbox_nodes) == len(bbox_ranges)

        # To build
        all_nodes = question_nodes + bbox_nodes
        num_nodes = filter_nodes(all_nodes, TAG_NUMBER[0])
        date_nodes = filter_nodes(all_nodes, TAG_DATE[0])
        semantic_nodes, semantic_range = filter_nodes_ranges(all_nodes, bbox_ranges, TAG_SELF[0])

        num_node_vals = convert_node_vals(num_nodes, to_number)
        date_node_vals = convert_node_vals(date_nodes, to_date)
        semantic_node_vals = convert_node_vals(semantic_nodes)

        # full_nodes = num_nodes + date_nodes + semantic_nodes
        full_node_vals = num_node_vals + date_node_vals + semantic_node_vals

        num_graph = build_greater_graph(num_node_vals)
        date_graph = build_greater_graph(date_node_vals)
        semantic_graph = build_semantic_graph_dqa(semantic_nodes, semantic_range)

        full_graph = build_full_graph_dqa(num_nodes, date_nodes, semantic_nodes, full_node_vals, semantic_graph)
        answer_dict = {"answer_type": answer_type, "answer": answer, "scale": scale, "answer_from": answer_from, "head_class":head_class}
        image_pixels = None if image is None else extract_feature_from_image(image)
       
        return self._make_instance(input_ids, input_bboxes, attention_mask, token_type_ids, question_mask, bbox_mask,
                                   input_bbox_orders, question_metas,
                                   block_metas, answer_dict, question_id,facts,consts,answer_mapping,out_seq,num_nodes,num_graph,date_nodes,date_graph,semantic_nodes,semantic_range,semantic_graph,full_graph, image_pixels)


    def _read(self, file_path: str, doc_folder:Path, use_vision=False):
        print("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        print("Reading the TAT-DQA dataset for training")
        instances = []
        skip_count = 0
        for one in tqdm(dataset):
            uid = one['doc']['uid']
            table_page_no = one['doc']['page']
            tat_doc_path = doc_folder / f'{uid}.json'
            tat_image = None
            tat_doc = json.load(open(tat_doc_path, 'r'))
            tat_blocks=[]
            pages_number = len(tat_doc['pages'])
           
            # default use the first page page
            tat_image_path = doc_folder / f'{uid}_{1}.png'
            tat_page = tat_doc['pages'][0]
            tat_image = Image.open(tat_image_path).convert("RGB")
            
            # use all pages
            if pages_number==1:
                tat_image_path = doc_folder / f'{uid}_{1}.png'
                tat_image = Image.open(tat_image_path).convert("RGB")
                tat_page=tat_doc['pages'][0]

            elif pages_number==2:
                tat_image=image_Splicing(doc_folder / f'{uid}_{1}.png',doc_folder / f'{uid}_{2}.png')
                tat_page = update_blocks(tat_doc['pages'][0],tat_doc['pages'][1])

            elif pages_number==3:
                tat_image=image_Splicing(doc_folder / f'{uid}_{1}.png',doc_folder / f'{uid}_{2}.png',doc_folder / f'{uid}_{3}.png')
                tat_page = update_blocks(tat_doc['pages'][0],tat_doc['pages'][1],tat_doc['pages'][2])
            
            
            page_bbox = tat_page['bbox']
            page_width = page_bbox[2]
            page_height = page_bbox[3]
            page_blocks = tat_page['blocks']

            for block in page_blocks:
                block['bbox'] = normalize_bbox(block['bbox'], page_width, page_height)
                block['words']['bbox_list'] = [normalize_bbox(w_b, page_width, page_height) for w_b in block['words']['bbox_list']]
                
            tat_blocks.extend(page_blocks)

            block_map = {it['uuid']: it for it in tat_blocks}

            questions = one['questions']
            for question_answer in questions:
                try:
                    question = question_answer["question"].strip()
                    answer_type = question_answer["answer_type"]
                    answer = question_answer["answer"]
                    answer_mapping = question_answer["block_mapping"]
                    answer_from = question_answer["order"]##answer_from = question_answer["answer_from"]
                    scale = question_answer["scale"]
                    derivation = question_answer['derivation']
                    facts = question_answer['facts']

                    instance = self._to_instance(question, block_map, answer_from,
                                                    answer_type, answer, answer_mapping, scale,
                                                    question_answer["uid"], derivation, facts, tat_image)
                    if instance is not None:
                        instances.append(instance)
                except RuntimeError as e:
                    print(f"Exception time error: {e}")
                    print(question_answer["uid"])
                    skip_count+=1
                except ValueError as e:
                    print(f"Exception value error: {e}")
                    print(question_answer["uid"])
                    skip_count+=1

        print(f'skip count:{skip_count}')
        return instances

class IndexMap(object):
    """Index grouping entries within a tensor."""

    def __init__(self, indices, num_segments, batch_dims=0):
        """
        Creates an index
        Args:
            indices (:obj:`torch.LongTensor`, same shape as a `values` Tensor to which the indices refer):
                Tensor containing the indices.
            num_segments (:obj:`torch.LongTensor`):
                Scalar tensor, the number of segments. All elements in a batched segmented tensor must have the same
                number of segments (although many segments can be empty).
            batch_dims (:obj:`int`, `optional`, defaults to 0):
                The number of batch dimensions. The first `batch_dims` dimensions of a SegmentedTensor are treated as
                batch dimensions. Segments in different batch elements are always distinct even if they have the same
                index.
        """
        self.indices = torch.as_tensor(indices)
        self.num_segments = torch.as_tensor(num_segments, device=indices.device)
        self.batch_dims = batch_dims

    def batch_shape(self):
        return self.indices.size()[: self.batch_dims]  # returns a torch.Size object
class ProductIndexMap(IndexMap):
    """The product of two indices."""

    def __init__(self, outer_index, inner_index):
        """
        Combines indices i and j into pairs (i, j). The result is an index where each segment (i, j) is the
        intersection of segments i and j. For example if the inputs represent table cells indexed by respectively rows
        and columns the output will be a table indexed by (row, column) pairs, i.e. by cell. The implementation
        combines indices {0, .., n - 1} and {0, .., m - 1} into {0, .., nm - 1}. The output has `num_segments` equal to
        `outer_index.num_segments` * `inner_index.num_segments`
        Args:
            outer_index (:obj:`IndexMap`):
                IndexMap.
            inner_index (:obj:`IndexMap`):
                IndexMap, must have the same shape as `outer_index`.
        """
        if outer_index.batch_dims != inner_index.batch_dims:
            raise ValueError("outer_index.batch_dims and inner_index.batch_dims must be the same.")

        super(ProductIndexMap, self).__init__(
            indices=(inner_index.indices + outer_index.indices * inner_index.num_segments),
            num_segments=inner_index.num_segments * outer_index.num_segments,
            batch_dims=inner_index.batch_dims,
        )
        self.outer_index = outer_index
        self.inner_index = inner_index

    def project_outer(self, index):
        """Projects an index with the same index set onto the outer components."""
        return IndexMap(
            indices=(index.indices // self.inner_index.num_segments).type(torch.float).floor().type(torch.long),
            num_segments=self.outer_index.num_segments,
            batch_dims=index.batch_dims,
        )

    def project_inner(self, index):
        """Projects an index with the same index set onto the inner components."""
        return IndexMap(
            indices=torch.fmod(index.indices, self.inner_index.num_segments)
                .type(torch.float)
                .floor()
                .type(torch.long),
            num_segments=self.inner_index.num_segments,
            batch_dims=index.batch_dims,
                )

def flatten(index, name="segmented_flatten"):
    """
    Flattens a batched index map (which is typically of shape batch_size, seq_length) to a 1d index map. This operation
    relabels the segments to keep batch elements distinct. The k-th batch element will have indices shifted by
    `num_segments` * (k - 1). The result is a tensor with `num_segments` multiplied by the number of elements in the
    batch.
    Args:
        index (:obj:`IndexMap`):
            IndexMap to flatten.
        name (:obj:`str`, `optional`, defaults to 'segmented_flatten'):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): The flattened IndexMap.
    """
    # first, get batch_size as scalar tensor
    batch_size = torch.prod(torch.tensor(list(index.batch_shape())))
    # next, create offset as 1-D tensor of length batch_size,
    # and multiply element-wise by num segments (to offset different elements in the batch) e.g. if batch size is 2: [0, 64]
    offset = torch.arange(start=0, end=batch_size, device=index.num_segments.device) * index.num_segments
    offset = offset.view(index.batch_shape())
    for _ in range(index.batch_dims, len(index.indices.size())):  # typically range(1,2)
        offset = offset.unsqueeze(-1)

    indices = offset + index.indices
    return IndexMap(indices=indices.view(-1), num_segments=index.num_segments * batch_size, batch_dims=0)

def range_index_map(batch_shape, num_segments, name="range_index_map"):
    """
    Constructs an index map equal to range(num_segments).
    Args:
        batch_shape (:obj:`torch.Size`):
            Batch shape
        num_segments (:obj:`int`):
            Number of segments
        name (:obj:`str`, `optional`, defaults to 'range_index_map'):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    batch_shape = torch.as_tensor(
        batch_shape, dtype=torch.long
    )  # create a rank 1 tensor vector containing batch_shape (e.g. [2])
    assert len(batch_shape.size()) == 1
    num_segments = torch.as_tensor(num_segments)  # create a rank 0 tensor (scalar) containing num_segments (e.g. 64)
    assert len(num_segments.size()) == 0

    indices = torch.arange(
        start=0, end=num_segments, device=num_segments.device
    )  # create a rank 1 vector with num_segments elements
    new_tensor = torch.cat(
        [torch.ones_like(batch_shape, dtype=torch.long, device=num_segments.device), num_segments.unsqueeze(dim=0)],
        dim=0,
    )
    # new_tensor is just a vector of [1 64] for example (assuming only 1 batch dimension)
    new_shape = [int(x) for x in new_tensor.tolist()]
    indices = indices.view(new_shape)

    multiples = torch.cat([batch_shape, torch.as_tensor([1])], dim=0)
    indices = indices.repeat(multiples.tolist())
    # equivalent (in Numpy:)
    # indices = torch.as_tensor(np.tile(indices.numpy(), multiples.tolist()))

    return IndexMap(indices=indices, num_segments=num_segments, batch_dims=list(batch_shape.size())[0])


def _segment_reduce(values, index, segment_reduce_fn, name):
    """
    Applies a segment reduction segment-wise.
    Args:
        values (:obj:`torch.Tensor`):
            Tensor with segment values.
        index (:obj:`IndexMap`):
            IndexMap.
        segment_reduce_fn (:obj:`str`):
            Name for the reduce operation. One of "sum", "mean", "max" or "min".
        name (:obj:`str`):
            Name for the operation. Currently not used
    Returns:
        (:obj:`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    # Flatten the batch dimensions, as segments ops (scatter) do not support batching.
    # However if `values` has extra dimensions to the right keep them
    # unflattened. Segmented ops support vector-valued operations.
    flat_index = flatten(index)
    vector_shape = values.size()[len(index.indices.size()):]  # torch.Size object
    flattened_shape = torch.cat(
        [torch.as_tensor([-1], dtype=torch.long), torch.as_tensor(vector_shape, dtype=torch.long)], dim=0
    )
    # changed "view" by "reshape" in the following line
    flat_values = values.reshape(flattened_shape.tolist())

    segment_means = scatter(
        src=flat_values,
        index=flat_index.indices.type(torch.long),
        dim=0,
        dim_size=flat_index.num_segments,
        reduce=segment_reduce_fn,
    )

    # Unflatten the values.
    new_shape = torch.cat(
        [
            torch.as_tensor(index.batch_shape(), dtype=torch.long),
            torch.as_tensor([index.num_segments], dtype=torch.long),
            torch.as_tensor(vector_shape, dtype=torch.long),
        ],
        dim=0,
    )

    output_values = segment_means.view(new_shape.tolist())
    output_index = range_index_map(index.batch_shape(), index.num_segments)
    return output_values, output_index

def convert_fact(fact):
    p = re.compile('[£€ ,\$a-zA-Z]')
    new_fact=[]
    for i in fact:
            new_fact.append(re.sub(p, '', i).replace('–', '-'))
    return new_fact

def image_Splicing(img_1, img_2, img_3=None):
    img1 = Image.open(img_1).convert("RGB")
    img2 = Image.open(img_2).convert("RGB")
    size1, size2 = img1.size, img2.size
    joint = Image.new("RGB", (size1[0], size2[1]+size1[1]))
    loc1, loc2 = (0, 0), (0, size1[1])
    joint.paste(img1, loc1)
    joint.paste(img2, loc2)
    if img_3:
        img3 = Image.open(img_3).convert("RGB")
        size3, size4 = img3.size, joint.size
        joint2 = Image.new("RGB", (size3[0], size3[1]+size4[1]))
        loc1, loc2 = (0, 0), (0, size4[1])
        joint2.paste(joint, loc1)
        joint2.paste(img3, loc2)
        newjoint=joint2.resize((224,224),Image.ANTIALIAS)
        return newjoint
    newjoint=joint.resize((224,224),Image.ANTIALIAS)
    return newjoint

def update_blocks(page1, page2, page3=None):

    page1_width,page1_height=page1['bbox'][-2],page1['bbox'][-1]
    page2_width,page2_height=page2['bbox'][-2],page2['bbox'][-1]
    page1_blocks = page1['blocks']
    page2_blocks = page2['blocks']
    
    for block1 in page1_blocks:
        block1['bbox'] = normalize_bbox(block1['bbox'], page1_width,page1_height)
        block1['words']['bbox_list'] = [normalize_bbox(w_b,page1_width,page1_height) for w_b in block1['words']['bbox_list']]
    
    result_page = copy.deepcopy(page1)
    pre_order = len(result_page['blocks'])
    for block2 in page2_blocks:
        block2['bbox'] = normalize_bbox(block2['bbox'], page2_width, page2_height)
        block2['bbox'] = update_bbox(block2['bbox'], 1000)
       
        block2['words']['bbox_list'] = [normalize_bbox(w_b,page2_width,page2_height) for w_b in block2['words']['bbox_list']] 
        block2['words']['bbox_list'] = [update_bbox(w_b,1000) for w_b in block2['words']['bbox_list']]
        block2['order']= block2['order']+pre_order

    result_page['blocks'].extend(page2_blocks)
    result_page['bbox']= [0, 0, 1000, 2000]
        
    if page3:
        pre_order = len(result_page['blocks'])
        page3_width,page3_height=page3['bbox'][-2],page3['bbox'][-1]
        page3_blocks = page3['blocks']
        for block3 in page3_blocks:
            block3['bbox'] = normalize_bbox(block3['bbox'], page3_width,page3_height)
            block3['bbox'] = update_bbox(block3['bbox'], 2000)

            block3['words']['bbox_list'] = [normalize_bbox(w_b,page3_width,page3_height) for w_b in block3['words']['bbox_list']]
            block3['words']['bbox_list'] = [update_bbox(w_b,2000) for w_b in block3['words']['bbox_list']]
            block3['order']= block3['order']+ pre_order
     
        result_page['blocks'].extend(page3_blocks)
        result_page['bbox']= [0, 0, 1000, 3000]
    
    return result_page

       

def update_bbox(bbox,height):
    return[
        bbox[0],
        bbox[1]+height,
        bbox[2],
        bbox[3]+height
    ]

def get_facts(blocks,answer_mapping):
    facts=[]
    if not answer_mapping:
        raise ValueError(f'exception: answer_mapping is blank')
    for mapping in answer_mapping:
        for key,value in mapping.items():
            if key in blocks.keys():
                fact=blocks[key]['text'][value[0]:value[1]]
            else:
                raise ValueError(f'exception: can not find answer_mapping')
        facts.append(fact)
    return facts

###################################################
