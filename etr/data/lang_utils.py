import pickle
import re


class Lang:
    """
    class to save the vocab and two dict: the word->index and index->word
    """

    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0
        self.gen_num = 0
        self.max_copy_num = 0
        self.copy_num_start = 0
        self.consts = []

    def load(self, load_path):
        with open(load_path, 'rb') as f:
            dic = pickle.load(f)
        self.word2index = dic['word2index']
        self.word2count = dic['word2count']
        self.index2word = dic['index2word']
        self.n_words = dic['n_words']  # Count word tokens
        self.num_start = dic['num_start']
        self.gen_num = dic['gen_num']
        self.max_copy_num = dic['max_copy_num']
        self.copy_num_start = dic['copy_num_start']

    def save(self, save_path):
        dic = {}
        dic['word2index'] = self.word2index
        dic['word2count'] = self.word2count
        dic['index2word'] = self.index2word
        dic['n_words'] = self.n_words  # Count word tokens
        dic['num_start'] = self.num_start
        dic['gen_num'] = self.gen_num
        dic['max_copy_num'] = self.max_copy_num
        dic['copy_num_start'] = self.copy_num_start
        with open(save_path, 'wb') as f:
            pickle.dump(dic, f)

    def add_const(self, consts):
        self.consts.extend(consts)
        self.consts = list(set(self.consts))

    def add_operators(self, operators):  # add operators of sentence to vocab
        for op in operators:
            op = str(op)
            if op not in self.index2word:
                self.word2index[op] = self.n_words
                self.word2count[op] = 1
                self.index2word.append(op)
                self.n_words += 1
            else:
                self.word2count[op] += 1

    def trim(self, min_count):  # trim words below a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_output_lang_for_tree(self, max_copy_num):  # build the output lang vocab and dict
        # max num of copy numbers
        self.max_copy_num = max_copy_num
        self.num_start = len(self.index2word)
        self.gen_num = len(self.consts)
        self.index2word = self.index2word + self.consts + ["N" + str(i) for i in range(max_copy_num)] + ["UNK"]
        self.n_words = len(self.index2word)
        # the start position of copy num
        self.copy_num_start = self.num_start + self.gen_num

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i
