import argparse
import os
import pickle
from pathlib import Path


from transformers import BertTokenizer
from transformers import RobertaTokenizer
from transformers import LayoutLMTokenizer
from transformers import LayoutLMv2Tokenizer

from etr.data.tat_dqa import TATDQAReader


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="./dataset_tatdqa")
parser.add_argument("--output_dir", type=str, default="./dataset_tatdqa")
parser.add_argument("--model_path", type=str, default='.')
# parser.add_argument("--passage_length_limit", type=int, default=463)
parser.add_argument("--question_length_limit", type=int, default=46)
parser.add_argument("--encoder", type=str, default="layoutlm_v2")
parser.add_argument("--mode", type=str, default='test')
parser.add_argument("--generate_limit", type=int, default=20)
parser.add_argument("--doc_dir", type=str, default="tat_docs")
parser.add_argument("--max_pieces" , type=int, default=1024)

args = parser.parse_args()

tokenizer = LayoutLMv2Tokenizer.from_pretrained('microsoft/layoutlmv2-large-uncased')
sep = '[SEP]'

if args.mode == 'test':
    data_reader = TATDQAReader(tokenizer, args.question_length_limit, sep=sep, max_pieces=args.max_pieces, mode=args.mode)
    data_mode = ["test"]
# elif args.mode == 'dev':
#     data_reader = TATDQAReader(tokenizer, args.question_length_limit, sep=sep,max_pieces=args.max_pieces, mode=args.mode)
#     data_mode = ["dev", "train"]
else:
    data_reader = TATDQAReader(tokenizer, args.question_length_limit, sep=sep,max_pieces=args.max_pieces, mode=args.mode)
    data_mode = ["train","dev"]


data_format = "tatdqa_dataset_{}.json"
print(f'==== NOTE ====: encoder:{args.encoder}, mode:{args.mode}')

for dm in data_mode:
    dpath = Path(args.input_path) / data_format.format(dm)
    doc_folder = Path(args.input_path) / args.doc_dir / dm
    print(f'==== NOTE ====: read file:{dpath}, TAT-DQA doc folder:{doc_folder}')
    data = data_reader._read(dpath, doc_folder, args.encoder=='layoutlm_v2')
    print(data_reader.skip_count)
    data_reader.skip_count = 0
    file_name = f'etr_{args.encoder}_tatdqa_cached_{dm}.pkl'
    if args.mode == 'test':
        file_name = f'etr_{args.encoder}_tatdqa_cached_{dm}_pred.pkl'
    out_lang_file_name = f'{args.encoder}_out_lang.pkl'
    print("Save data to {}.".format(os.path.join(args.output_dir, file_name)))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if dm == 'dev':
        data_reader.build_lang(args.generate_limit)
        data_reader.out_lang.save(os.path.join(args.output_dir, out_lang_file_name))
        print("Save lang to {} with limit {}.".format(os.path.join(args.output_dir, out_lang_file_name), args.generate_limit))
    with open(os.path.join(args.output_dir, file_name), "wb") as f:
        pickle.dump(data, f)
        f.flush()
        f.close()
