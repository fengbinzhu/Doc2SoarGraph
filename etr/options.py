import torch
from argparse import ArgumentParser


def add_data_args(parser: ArgumentParser):
    parser.add_argument("--gpu_num",  default=torch.cuda.device_count(), type=int, help="training gpu num.")
    parser.add_argument("--data_dir", default="./dataset_1203", type=str, help="data dir.")
    parser.add_argument("--save_dir", default="./checkpoint/tatdqa_b64_lr5e05_20221203/", type=str,  help="save dir.")
    parser.add_argument("--log_file", default="train.log", type=str, help="train log file.")

def add_train_args(parser: ArgumentParser):

    parser.add_argument("--max_seq_len", type=int, default=561, help="The maximum sequence length after layoutlm_v2")
    parser.add_argument("--max_input_len", type=int, default=512, help="The maximum input length")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--log_per_updates", default=50, type=int, help="log pre update size.")
    parser.add_argument("--max_epoch", default=100, type=int, help="max epoch.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="learning rate.")
    parser.add_argument("--grad_clipping", default=1.0, type=float, help="gradient clip.")
    parser.add_argument('--warmup', type=float, default=0.06,
                        help="Proportion of training to perform linear learning rate warm up for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_schedule", default="warmup_linear", type=str, help="warmup schedule.")
    parser.add_argument("--optimizer", default="adam", type=str, help="train optimizer.")
    parser.add_argument('--seed', type=int, default=2018, help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--pre_path', type=str, default=None, help="Load from pre trained.")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout.")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size.")
    parser.add_argument('--eval_batch_size', type=int, default=32, help="eval batch size.")
    parser.add_argument("--eps", type=float,default=1e-5, help="ema gamma.")
    parser.add_argument("--tree_learning_rate", type=float, default=5e-4, help="tree learning rate.")
    parser.add_argument("--tree_weight_decay", type=float, default=0.01, help="tree weight decay.")


def add_bert_args(parser: ArgumentParser):
    parser.add_argument("--bert_learning_rate", type=float, default=1.5e-5,help="bert learning rate.")
    parser.add_argument("--bert_weight_decay", type=float, default=0.01, help="bert weight decay.")
    #parser.add_argument("--roberta_model", type=str, help="robert model path.", default="./roberta.large")
    parser.add_argument("--roberta_model", type=str, help="robert model path.", default="./roberta.large")

def add_inference_args(parser: ArgumentParser):
    parser.add_argument("--pre_path", type=str, help="Prepath")
    parser.add_argument("--data_mode", type=str, help="inference data mode")
    parser.add_argument("--inf_path", type=str, help="inference data path.")
    parser.add_argument("--dump_path", type=str, help="inference data path.")
    parser.add_argument('--eval_batch_size', type=int, default=32, help="eval batch size.")
