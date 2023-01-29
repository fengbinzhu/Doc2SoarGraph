import os
import json
import argparse
import pickle
from datetime import datetime
from etr.model.model import ETRPredictModel
from etr.data.lang_utils import Lang
from etr import options
from pprint import pprint
import pandas as pd
from etr.model.sys_util import create_logger, set_environment
from etr.data.etr_batch_gen import ETRTestBatchGen
from transformers import RobertaModel, BertModel
from etr.model.modeling_etr import MutiHeadModel
import torch.nn as nn

parser = argparse.ArgumentParser("ETR Prediction task.")
options.add_data_args(parser)
options.add_train_args(parser)
options.add_bert_args(parser)
parser.add_argument("--encoder", type=str, default='roberta')
parser.add_argument("--op_mode", type=int, default=0)
parser.add_argument("--ablation_mode", type=int, default=0)
parser.add_argument("--test_data_dir", type=str, default="./dataset_tagdqa")
parser.add_argument("--model_path", type=str, default='./tatdqa_etr_b128_32_lr_5e04_tr5e04')
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

args.cuda = args.gpu_num > 0

logger = create_logger("Roberta Predicting")

pprint(args)
set_environment(2018, args.cuda)


def main():
    logger.info("Loading lang..")
    lang = Lang()
    lang.load(os.path.join(args.data_dir, "out_lang.pkl"))
    logger.info("Loading data...")

    test_itr = ETRTestBatchGen(args, data_mode="test", encoder=args.encoder)

    logger.info("TAT data: {}!".format(len(test_itr)))

    logger.info(f"Build {args.encoder} model.")
    if args.encoder == 'bert':
        bert_model = BertModel.from_pretrained('bert-large-uncased')
    elif args.encoder == 'roberta':
        bert_model = RobertaModel.from_pretrained(args.roberta_model)

    network = MutiHeadModel(bert=bert_model,
                            config=bert_model.config,
                            out_lang=lang,
                            scale_criterion=nn.CrossEntropyLoss(reduction='sum'),
                            scale_classes=5)

    model = ETRPredictModel(args, network)
    load_path = os.path.join(args.model_path, "checkpoint_best.pt")
    model.load(load_path)
    train_start = datetime.now()

    model.reset()
    pred_json = model.predict(test_itr)
    raw_detail, detail_em, detail_f1, score_dict = model.get_metrics(logger)

    logger.info('time total:{}'.format((datetime.now() - train_start)))
    with open('count.pkl', 'wb') as f:
        pickle.dump(model.network.count, f)

    output_metric_path = os.path.join(args.save_dir, 'metrics.xlsx')
    with pd.ExcelWriter(output_metric_path, engine='xlsxwriter') as writer:
        raw_detail.to_excel(writer, sheet_name='raw')
        detail_em.to_excel(writer, sheet_name='em')
        detail_f1.to_excel(writer, sheet_name='f1')

    with open(os.path.join(args.save_dir, 'pred.json'), 'w') as f:
        json.dump(pred_json, f, indent=2)
    with open(os.path.join(args.save_dir, 'score.json'), 'w') as fl:
        json.dump(score_dict, fl, indent=2)


if __name__ == "__main__":
    main()
