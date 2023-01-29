import numpy as np
import datetime
import re
from tatqa_utils import *
from data.tag_constant import *
import copy

def tagOffsets2nodes(tag_offsets, tokens, token2word_idx, words, signature=None, source='table'):
    nodes = []
    valid_tag_token_offsets = []
    add_self = True
    for tag, tag_values in tag_offsets.items():
        if tag == TAG_SELF[0]:
            continue
        for idx, tag_val in enumerate(tag_values):
            sign = f"{tag}:{source}:{signature}:{idx}"
            val = tag_val['value']
            offset = tag_val['seq_offsets']
            token_offset = tag_val['token_offsets']
            is_target = tag_val['is_target']
            order = tag_val['order']
            nodes.append((sign, tag, val, source, offset, offset, [], [], order, is_target))
            valid_tag_token_offsets.append(token_offset)

    if len(valid_tag_token_offsets) == 1:
        tmp_tokens = copy.deepcopy(tokens)
        s = valid_tag_token_offsets[0][0]
        e = valid_tag_token_offsets[0][1]
        tmp_tokens[s:e] = [''] * (e - s)
        if not re.findall('[a-zA-Z]', " ".join(tmp_tokens)):
            add_self = False

    if add_self:
        _self = tag_offsets[TAG_SELF[0]]
        val = _self['value'] if 'value' in _self else ""
        seq_offsets = _self['seq_offsets']
        is_target = _self['is_target'] if 'is_target' in _self else 0  # self cell will be always 0 for FinQA
        self_answer_offsets = _self['self_answer_offsets'] if 'self_answer_offsets' in _self else []
        answer_offsets = [one[2] for one in self_answer_offsets]
        sign = f"{TAG_SELF[0]}:{source}:{signature}:0"
        # token to text
        nodes.append((sign, TAG_SELF[0], val, source, seq_offsets, answer_offsets, token2word_idx, words, -1, is_target))
    return nodes


def tagOffsets2nodes_dqa(tag_offsets, tokens, token2word_idx, words, signature=None, source='table'):
    nodes = []
    valid_tag_token_offsets = []
    add_self = True
    for tag, tag_values in tag_offsets.items():
        if tag == TAG_SELF[0]:
            continue
        for idx, tag_val in enumerate(tag_values):
            sign = f"{tag}:{source}:{signature}:{idx}"
            val = tag_val['value']
            offset = tag_val['seq_offsets']
            token_offset = tag_val['token_offsets']
            is_target = tag_val['is_target']
            order = tag_val['order']
            tag_token2word_idx = token2word_idx[token_offset[0]:token_offset[1]]
            unique_word_idxs = list(np.unique(np.array(tag_token2word_idx)))
            tag_words = [words[idx] for idx in unique_word_idxs]
            nodes.append((sign, tag, val, source, offset, offset, tag_token2word_idx, tag_words, order, is_target))
            valid_tag_token_offsets.append(token_offset)

    if len(valid_tag_token_offsets) == 1:
        tmp_tokens = copy.deepcopy(tokens)
        s = valid_tag_token_offsets[0][0]
        e = valid_tag_token_offsets[0][1]
        tmp_tokens[s:e] = [''] * (e - s)
        if not re.findall('[a-zA-Z]', " ".join(tmp_tokens)):
            if 'is_target' in tag_offsets[TAG_SELF[0]].keys():
                if tag_offsets[TAG_SELF[0]]['is_target']:
                    nodes = []
                else:
                    add_self= False

    if add_self:
        _self = tag_offsets[TAG_SELF[0]]
        # if 'value'  in _self:
        #     print("")
        val = _self['value'] if 'value' in _self else ""
        seq_offsets = _self['seq_offsets']
        is_target = _self['is_target'] if 'is_target' in _self else 0  # self cell will be always 0 for FinQA
        # if 'self_answer_offsets'  in _self:
        #     print("")
        self_answer_offsets = _self['self_answer_offsets'] if 'self_answer_offsets' in _self else []
        answer_offsets = [one[2] for one in self_answer_offsets]
        sign = f"{TAG_SELF[0]}:{source}:{signature}:0"
        # if self_answer_offsets==[] and is_target==1:
        #     print('')
        # token to text
        nodes.append((sign, TAG_SELF[0], val, source, seq_offsets, answer_offsets, token2word_idx, words, -1, is_target))
    return nodes


def questionMeta2nodes(question_metas):
    question_ids, question_tokens, question_token_tags, question_offsets, question_token2word_idx, question_words = question_metas
    return tagOffsets2nodes(question_offsets, question_tokens,  question_token2word_idx, question_words, signature='Q', source='question')

def questionMeta2nodes_dqa(question_metas):
    question_ids, question_tokens, question_token_tags, question_offsets, question_token2word_idx, question_words = question_metas
    return tagOffsets2nodes_dqa(question_offsets, question_tokens,  question_token2word_idx, question_words, signature='Q', source='question')

def tableMeta2nodes(table_metas):
    table_nodes = []
    for cell_row_ids, cell_col_ids, cell_ids, cell_tokens, cell_token_tags, cell_offsets, cell_token2word_idx, cell_words in table_metas:
        signature = f"{cell_row_ids[0]}_{cell_col_ids[0]}"
        one_cell_nodes = tagOffsets2nodes(cell_offsets, cell_tokens, cell_token2word_idx, cell_words, signature=signature, source='table')
        table_nodes.extend(one_cell_nodes)
    return table_nodes


def paragraphMeta2nodes(paragraphs_metas):
    para_nodes = []
    for idx, (para_order, paragraph_ids, paragraph_tokens, paragraph_token_tags, paragraph_offsets, paragraph_token2word_idx, paragraph_words) in enumerate(paragraphs_metas):
        one_para_nodes = tagOffsets2nodes(paragraph_offsets,paragraph_tokens, paragraph_token2word_idx, paragraph_words, signature=idx, source='paragraph')
        para_nodes.extend(one_para_nodes)
    return para_nodes

def bboxMeta2nodes(bbox_metas: object) -> object:
    bbox_nodes = []
    bbox_ranges = []
    for idx,(bbox_order, bbox_ids, bbox_tokens, bbox_token_tags, bbox_offsets, bbox_token2word_idx, bbox_words,word_bboxes,block_token_bboxes, bbox_range) in enumerate(bbox_metas):
        one_para_nodes = tagOffsets2nodes_dqa(bbox_offsets,bbox_tokens, bbox_token2word_idx, bbox_words, signature=idx, source='bbox')
        bbox_nodes.extend(one_para_nodes)
        bbox_ranges.extend(add_range(one_para_nodes, bbox_range))
    return bbox_nodes,bbox_ranges

def filter_nodes(all_nodes, tag=TAG_SELF[0]):
    return [node for node in all_nodes if node[1] == tag]

def filter_nodes_ranges(all_nodes,all_ranges,tag=TAG_SELF[0]):

    return [node for node in all_nodes if node[1] == tag],[bbox_range for bbox_range in all_ranges if bbox_range[1]==tag]


def convert_node_vals(nodes, converter=lambda x: x):
    return [converter(it[2]) for it in nodes]

def convert_node_vals_dqa(nodes, converter=lambda x: x):
    return [converter(it[1][2]) for it in nodes]

def get_nodes(batch_nodes, tag=TAG_SELF[0]):
    result = []
    for one_seq_nodes in batch_nodes:
        tag_nodes = [node for node in one_seq_nodes if node[1] == tag]
        result.append(tag_nodes)
    return result

def build_greater_graph(node_list):
    '''
    To build a directional graph by node comparison greater -> smaller
    :param max_len:
    :param node_list:
    :param node_masks:
    :return:
    '''
    diag_ele = np.zeros(len(node_list))
    vals = node_list
    graph = np.diag(diag_ele)
    for i in range(len(vals)):
        # if vals[i] is None: # not a number # invalid date
        #     continue
        for j in range(len(vals)):
            # if vals[j] is None:  #  not a number # invalid date
            #     continue
            if vals[i] > vals[j]:
                graph[i][j] = 1
            else:
                graph[j][i] = 1
    return graph


def _update_semantic_graph(i, j, i_node, j_node, graph, mapping):
    i_sign, i_tag, i_val, i_source, _, _, _, _, i_ord, i_is_target = i_node
    j_sign, j_tag, j_val, j_source, _, _, _, _, j_ord, j_is_target = j_node
    if i_source == 'question' and j_source != 'question':  # question -> table cell and question -> paragraph
        graph[i][j] = 1
        graph[j][i] = 1
    elif i_source == 'paragraph':
        if j_source == 'table':  # paragraph -> table cell
            graph[i][j] = 1
            graph[j][i] = 1
        elif j_source == 'paragraph':  # paragraph -> next paragraph
            i_para_idx = int(i_sign.split(':')[-2])
            j_para_idx = int(j_sign.split(':')[-2])
            if i_para_idx < j_para_idx and i_para_idx not in mapping:
                graph[i][j] = 1
                graph[j][i] = 1
                mapping[i_para_idx] = j_sign
    elif i_source == 'table' and j_source == 'table':  # table cell <> cell edges
        i_cell_sign = i_sign.split(':')[-2]
        j_cell_sign = j_sign.split(':')[-2]
        i_row, i_col = [int(it) for it in i_cell_sign.split('_')]
        j_row, j_col = [int(it) for it in j_cell_sign.split('_')]
        row_key = f"{i_cell_sign}_row"
        col_key = f"{i_cell_sign}_col"
        if i_row == j_row and i_col < j_col and row_key not in mapping:  # same row edge neighbour cell
            graph[i][j] = 1
            graph[j][i] = 1
            mapping[row_key] = j_sign
        elif i_row < j_row and i_col == j_col and col_key not in mapping:  # # same col edge neighbour cell
            graph[i][j] = 1
            graph[j][i] = 1
            mapping[col_key] = j_sign


def build_semantic_graph(semantic_nodes):

    diag_ele = np.zeros(len(semantic_nodes))
    # seq * (sign, tag, val, emb, source, -1, is_target)
    nodes = semantic_nodes
    graph = np.diag(diag_ele)
    for i, i_node in enumerate(nodes):
        if nodes[i] is None:  # mask or not a number # invalid date
            continue
        mapping = {}
        for j, j_node in enumerate(nodes):
            # sign = f"{tag}:{source}:{signature}:{idx}" signature (order  or row_id_col_id
            _update_semantic_graph(i, j, i_node, j_node, graph, mapping)
    return graph

def build_semantic_graph_dqa(semantic_nodes, bbox_range):
    diag_ele = np.zeros(len(semantic_nodes))
    nodes = semantic_nodes
    locations= get_location(bbox_range)
    graph = np.diag(diag_ele)
    orientation_dicts = make_orientation_dict(locations)
    for i in range(len(nodes)-1):
            graph = update_semantic_graph_dqa(i,orientation_dicts,graph)
        #add question-bbox
    for i in range(len(graph)):
        graph[0][i]=1
        graph[i][0]=1
    return graph

def make_orientation_dict(locations):
    orientation_dicts=[]
    for i in locations:
        orientation_dict={'up':[],'right':[],'left':[],'down':[],'self':[]}
        for m in range(len(locations)):
            location = locations[m]
            orientation_dict[calculate_orientation(i,location)].append((calculate_distance(i,location),m))
        min_ori_dict  = {}
        for key,value in orientation_dict.items():
            if value:
                min_ori_dict[key] = min(orientation_dict[key])
        orientation_dicts.append(min_ori_dict)

    return orientation_dicts

def update_semantic_graph_dqa(i,orientation_dicts,graph):
    graph[i][0:i]=1
    graph[i][i+1:len(graph[i]+1)]=1

    # for key,value in orientation_dicts[i].items():
    #     if value:
    #         ##print(i+1,key,min(value))
    #         graph[i + 1][value[1] + 1] = 1
    #         graph[value[1] + 1][i + 1] = 1

    return graph

def calculate_orientation(a,b):
    x1,y1=a
    x2,y2=b
    if x2>x1:
        slope=(y2-y1)/(x2-x1)
        if slope>=1:
            return 'up'
        elif -1<slope<1:
            return 'right'
        elif slope<=-1:
            return 'down'
    elif x2==x1:
        if y2>y1:
            return 'up'
        elif y2<y1:
            return 'down'
        elif y2==y1:
            return 'self'
    elif x2<x1:
        slope=(y2-y1)/(x2-x1)
        if slope>=1:
            return 'down'
        elif -1<slope<1:
            return 'left'
        elif slope<=-1:
            return 'up'

def calculate_distance(a,b):
    x1,y1=a
    x2,y2=b
    distance=pow(pow(x2-x1,2)+pow(y2-y1,2),0.5)
    return distance

def get_location(bbox_ranges):
    locations=[]
    for i in bbox_ranges:
        x1,y1,x2,y2=i[0]
        locations.append([(x1+x2)/2,(y1+y2)/2])
    return locations

def get_node(i,
             num_nodes,
             date_nodes,
             semantic_nodes):
    num_len = len(num_nodes)
    date_len = len(date_nodes)
    if i < num_len:  # num
        return num_nodes[i]
    elif num_len <= i < num_len + date_len:  # date
        return date_nodes[i - num_len]
    return semantic_nodes[i - num_len - date_len]


def get_semantic_node_sign_set(bz_semantic_nodes):
    return set([get_semantic_sign(node[0]) for node in bz_semantic_nodes])


def get_semantic_sign(node_sign):
    return "_".join(node_sign.split(':')[1:3])

def build_full_graph(num_nodes,
                     date_nodes,
                     semantic_nodes,
                     full_node_vals):

    diag_ele = np.zeros(len(num_nodes) + len(date_nodes) + len(semantic_nodes))
    # seq * (sign, tag, val, emb, source, -1, is_target)
    semantic_node_set = get_semantic_node_sign_set(semantic_nodes)
    vals = full_node_vals
    graph = np.diag(diag_ele)
    for i, val in enumerate(vals):
        i_node = get_node(i, num_nodes, date_nodes, semantic_nodes)
        i_sign, i_tag, i_val, i_source, _, _, _, _, i_ord, i_is_target = i_node
        i_self_sign = get_semantic_sign(i_sign)
        i_has_self = True if i_self_sign in semantic_node_set else False

        mapping = {}
        for j, val in enumerate(vals):
            j_node = get_node(j, num_nodes, date_nodes, semantic_nodes)
            j_sign, j_tag, j_val, j_source, _, _, _, _, j_ord, j_is_target = j_node
            j_self_sign = get_semantic_sign(j_sign)
            j_has_self = True if j_self_sign in semantic_node_set else False

            # i -> j
            # sign = f"{tag}:{source}:{signature}:{idx}" signature (para idx  or row_id_col_id
            if i_tag in [TAG_NUMBER[0], TAG_DATE[0]]: # num/date -> self
                if i_has_self: # num/date -> self
                    if j_tag == TAG_SELF[0] and i_self_sign == j_self_sign:
                        graph[i][j] = 1
                        # graph[j][i] = 1
                else:
                    # 1.1 num/date -> other semantic cells with same row/col in the table
                    # 1.2 num -> date cells with same row/col in the table
                    if i_source == 'table' and j_source == 'table':
                        if (i_tag == TAG_NUMBER[0] and j_tag == TAG_DATE[0]) or j_tag == TAG_SELF[0]:
                            i_cell_sign = i_sign.split(':')[-2]
                            j_cell_sign = j_sign.split(':')[-2]
                            i_row, i_col = [int(it) for it in i_cell_sign.split('_')]
                            j_row, j_col = [int(it) for it in j_cell_sign.split('_')]
                            if i_row == j_row or i_col == j_col:
                                graph[i][j] = 1
                                # graph[j][i] = 1
            # j -> i
            if j_tag in [TAG_NUMBER[0], TAG_DATE[0]]: # num/
                if j_has_self:  # num/date -> self
                    if i_tag == TAG_SELF[0] and i_self_sign == j_self_sign:
                        graph[j][i] = 1
                else:
                    # 1.1 num/date -> other semantic cells with same row/col in the table
                    # 1.2 num -> date cells with same row/col in the table
                    if i_source == 'table' and j_source == 'table':
                        if (j_tag == TAG_NUMBER[0] and i_tag == TAG_DATE[0]) or i_tag == TAG_SELF[0]:
                            i_cell_sign = i_sign.split(':')[-2]
                            j_cell_sign = j_sign.split(':')[-2]
                            i_row, i_col = [int(it) for it in i_cell_sign.split('_')]
                            j_row, j_col = [int(it) for it in j_cell_sign.split('_')]
                            if i_row == j_row or i_col == j_col:
                                graph[j][i] = 1

            # num
            if i_tag == TAG_NUMBER[0] and j_tag == TAG_NUMBER[0]:
                if i_val >= j_val:
                    graph[i][j] = 1
                else:
                    graph[j][i] = 1

            # date
            if i_tag == TAG_DATE[0] and j_tag == TAG_DATE[0]:
                if i_val >= j_val:
                    graph[i][j] = 1
                else:
                    graph[j][i] = 1

            # semantic
            if i_tag == j_tag == TAG_SELF[0]:  # repeat the semantic graph
                _update_semantic_graph(i, j, i_node, j_node, graph, mapping)
    return graph

def build_full_graph_dqa(num_nodes,
                     date_nodes,
                     semantic_nodes,
                     full_node_vals,
                     semantic_graph):

    diag_ele = np.zeros(len(num_nodes) + len(date_nodes) + len(semantic_nodes))
    semantic_begin=len(num_nodes)+len(date_nodes)
    semantic_end=semantic_begin+len(semantic_nodes)
    graph_cursor=0
    # seq * (sign, tag, val, emb, source, -1, is_target)
    semantic_node_set = get_semantic_node_sign_set(semantic_nodes)
    vals = full_node_vals
    graph = np.diag(diag_ele)

    for i, val in enumerate(vals):
        i_node = get_node(i, num_nodes, date_nodes, semantic_nodes)
        i_sign, i_tag, i_val, i_source, _, _, _, _, i_ord, i_is_target = i_node
        i_self_sign = get_semantic_sign(i_sign)
        i_has_self = True if i_self_sign in semantic_node_set else False

        mapping = {}
        for j, val in enumerate(vals):
            j_node = get_node(j, num_nodes, date_nodes, semantic_nodes)
            j_sign, j_tag, j_val, j_source, _, _, _, _, j_ord, j_is_target = j_node
            j_self_sign = get_semantic_sign(j_sign)
            j_has_self = True if j_self_sign in semantic_node_set else False
            # sign = f"{tag}:{source}:{signature}:{idx}" signature (para idx  or row_id_col_id
            
            # i -> j
            if i_tag in [TAG_NUMBER[0], TAG_DATE[0]]: # num/date -> self
                if i_has_self and j_tag == TAG_SELF[0] and i_self_sign == j_self_sign : # num/date -> self
                    graph[i][j] = 1

            # j -> i 
            if j_tag in [TAG_NUMBER[0], TAG_DATE[0]]:
                if j_has_self and i_tag == TAG_SELF[0] and i_self_sign == j_self_sign:
                    graph[j][i] = 1

             # num comparison
            if i_tag == TAG_NUMBER[0] and j_tag == TAG_NUMBER[0]:
                if i_val >= j_val:
                    graph[i][j] = 1
                else:
                    graph[j][i] = 1

            # date comparison
            if i_tag == TAG_DATE[0] and j_tag == TAG_DATE[0]:
                if i_val >= j_val:
                    graph[i][j] = 1
                else:
                    graph[j][i] = 1
                       
    for i in graph[semantic_begin:semantic_end]:
        i[semantic_begin:semantic_end] = semantic_graph[graph_cursor]
        graph_cursor += 1

    return graph

def add_range(bbox_nodes,bbox_range):
    bbox_ranges=[]
    for i in range(len(bbox_nodes)):
        bbox_ranges.append((bbox_range,bbox_nodes[i][1]))
    return bbox_ranges