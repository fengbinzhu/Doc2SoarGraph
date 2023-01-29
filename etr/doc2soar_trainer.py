import argparse
import json
import os
# os.environ['CUDA_VISIBLE_DEVICES']='3'

import pickle
from datetime import datetime
from pathlib import Path
from pprint import pprint
import torch
import torch.nn as nn
from transformers import RobertaModel, BertModel, LayoutLMModel, LayoutLMv2Model
import options
from etr.data.batch_gen import GNNBatchGen
from data.lang_utils import Lang
from model.model import ETRFineTuningModel, ETRPredictModel
from etr.model.model_doc2soar import Doc2SoarGraph
from model.sys_util import create_logger, set_environment

parser = argparse.ArgumentParser("GNN Tree training and evaluating task.")

options.add_data_args(parser)
options.add_bert_args(parser)
options.add_train_args(parser)

parser.add_argument("--encoder", type=str, default='layoutlm_v2')
# parser.add_argument("--test_data_dir", type=str, default="./dataset_tatdqa_gnn/dev_/")
parser.add_argument("--model_path", type=str, default='./')
parser.add_argument("--pretrained", type=bool, default=True)
parser.add_argument("--dataset", type=str, default='tatdqa')
parser.add_argument("--mode", type=str, default='eval')
parser.add_argument("--max_copy_num", type=int, default=12)
parser.add_argument("--answer_type", type=str, default='count')

args = parser.parse_args()

if not Path(args.save_dir).exists():
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

if not Path(args.model_path).exists():
    Path(args.model_path).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args.cuda = args.gpu_num > 0

args_path = os.path.join(args.save_dir, "args.json")
with open(args_path, "w") as f:
    json.dump((vars(args)), f)

args.batch_size = args.batch_size // args.gradient_accumulation_steps

logger = create_logger("ETR Training", log_file=os.path.join(args.save_dir, args.log_file))

pprint(args)
set_environment(args.seed, args.cuda)


def main():
    best_result = float("-inf")
    lang_path = os.path.join(args.data_dir, f"{args.encoder}_out_lang.pkl")
    logger.info(f"Loading lang {lang_path} ...")
    lang = Lang()
    lang.load(lang_path)
    logger.info("Loading data...")

    train_itr = GNNBatchGen(args, data_mode="train", out_lang=lang, encoder=args.encoder, dataset=args.dataset, answer_type=args.answer_type)
    dev_itr = GNNBatchGen(args, data_mode="dev", out_lang=lang, encoder=args.encoder, dataset=args.dataset, answer_type=args.answer_type)

    num_train_steps = int(args.max_epoch * len(train_itr) / args.gradient_accumulation_steps)
 
    logger.info("GNN Tree update steps {}!".format(num_train_steps))
    logger.info(f"Build GNN Tree {args.encoder} model.")
    if args.encoder == 'bert':
        bert_model = BertModel.from_pretrained('bert-large-uncased')
    elif args.encoder == 'roberta':
        bert_model = RobertaModel.from_pretrained(args.roberta_model)
    elif args.encoder == 'layoutlm':
        bert_model = LayoutLMModel.from_pretrained("microsoft/layoutlm-large-uncased")
    elif args.encoder == 'layoutlm_v2':
        bert_model = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-large-uncased")
    

    network = Doc2SoarGraph(bert=bert_model,
                               config=bert_model.config,
                               out_lang=lang,
                               scale_criterion=nn.CrossEntropyLoss(reduction='sum'),
                               encoder=args.encoder,
                               scale_classes=5,
                               head_count=5,
                               dataset=args.dataset,
                               max_copy_num=args.max_copy_num,
                               max_seq_len=args.max_seq_len,
                               max_input_len=args.max_input_len,
                               mode=args.mode)

    model = ETRFineTuningModel(args, network, num_train_steps=num_train_steps) if args.mode != 'test' else ETRPredictModel(args, network)
    epoch_pre = 0
    if args.pretrained:
        best_model_prefix = Path(args.save_dir) / "checkpoint_best"
        best_model_prefix = best_model_prefix.as_posix()
        if Path(best_model_prefix + '.pt').exists():
            logger.info(f'loading previous best model: {best_model_prefix}')
            model.load(best_model_prefix)
            other_path = best_model_prefix + '.ot'
            epoch_pre = torch.load(other_path)['epoch']
    train_start = datetime.now()
    first = True

    skip_batch = 0
    if args.mode == 'train':
        logger.info("====== The mode is [Train ]=======")
        for epoch in range(1 + epoch_pre, args.max_epoch + epoch_pre + 1):
            model.reset()
            if not first:
                train_itr.reset()
            first = False

            logger.info('At epoch {}'.format(epoch))
            for step, batch in enumerate(train_itr):
                # try:
                if model.step % 10 == 0:
                    logger.info(f'=====complete steps: {model.step}')
                model.update(batch)
                if model.step % (args.log_per_updates * args.gradient_accumulation_steps) == 0 or model.step == 1:
                    logger.info(
                        "Updates[{0:6}] train loss[{1:.5f}] head acc[{2:.5f}] skip_batch: [{3:6}] remaining[{4}].\r\n ".format(
                            model.updates, model.train_loss.avg, model.head_acc.avg, model.network.skip_batch,
                            str((datetime.now() - train_start) / (step + 1) * (num_train_steps - step - 1)).split('.')[0])
                    )
                    logger.info("Loss_dict:{0}, tree_acc:{1:.5f} \r\n".format(model.loss_dic,
                                                                              model.network.acc[0] / max(model.network.acc[1], 1e-5)))

                    model.avg_reset()
                # except ValueError as e:
                #     logger.info(f'Training exception, skip it:{e}')
                #     skip_batch += 1
            # with open('etr/met_count.pkl', 'wb') as f:
            #     pickle.dump(model.network.count, f)
            train_metric = model.get_metrics(logger)
            logger.info(f"Train Metric:{train_metric}")

            model.reset()
            model.avg_reset()
            logger.info("====== Begin to Evaluate on Dev set...")
            model.evaluate(dev_itr)
            logger.info("Evaluate epoch:[{0:6}] eval loss[{1:.5f}] head acc[{2:.5f}].\r\n".format(epoch, model.dev_loss.avg,
                                                                                                  model.head_acc.avg))
            eval_metrics = model.get_metrics(logger)

            logger.info(f"Eval Metric:{eval_metrics}")

            if eval_metrics["f1"] >= best_result:
                save_prefix = os.path.join(args.save_dir, "checkpoint_best")
                model.save(save_prefix, epoch)
                best_result = eval_metrics["f1"]
                logger.info("Best eval F1 {} at epoch {}.\r\n".format(best_result, epoch))

    elif args.mode == 'eval':
        logger.info("====== The mode is [Evaluation]=======")

        model.reset()
        model.avg_reset()
        logger.info("====== Begin to Evaluate on Dev set")
        model.evaluate(dev_itr)

        df_dev = model.get_df()
        output_raw_path_dev = os.path.join(args.save_dir, f'{args.dataset}_{args.encoder}_{args.answer_type}_dev_raw.xlsx')
        import pandas as pd
        with pd.ExcelWriter(output_raw_path_dev) as writer:
            df_dev.to_excel(writer, sheet_name='details')

        eval_metrics = model.get_metrics(logger)

        logger.info(f"Eval Metric on Dev:{eval_metrics}")

    else:
        pred_dev_itr = GNNBatchGen(args, data_mode="dev_pred", out_lang=lang, encoder=args.encoder, dataset=args.dataset, answer_type=args.answer_type)
        pred_test_itr = GNNBatchGen(args, data_mode="test_pred", out_lang=lang, encoder=args.encoder, dataset=args.dataset, answer_type=args.answer_type)

        model.avg_reset()
        model.reset()
        logger.info("====== Begin to Prediction on Dev set...")
        pred_result = model.predict(pred_dev_itr)
        output_raw_path_dev_est = os.path.join(args.save_dir, f'{args.dataset}_{args.encoder}_dev_pred.json')
        json.dump(pred_result, open(output_raw_path_dev_est, 'w'), indent=2)

        pred_dev_metrics = model.get_metrics(logger)

        logger.info(f"Pred Dev Metric: {pred_dev_metrics}")

        model.avg_reset()
        model.reset()
        logger.info("====== Begin to Prediction on Test set...")
        pred_result = model.predict(pred_test_itr)
        output_raw_path_test = os.path.join(args.save_dir, f'{args.dataset}_{args.encoder}_test_pred.json')
        json.dump(pred_result, open(output_raw_path_test, 'w'), indent=2)

        test_metrics = model.get_metrics(logger)

        logger.info(f"Pred Test Metric: {test_metrics}")
        

if __name__ == "__main__":
    main()
