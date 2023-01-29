import torch
import torch.nn as nn
from .optimizer import BertAdam as Adam
from .sys_util import AverageMeter
from tqdm import tqdm


class ETRPredictModel():
    def __init__(self, args, network):
        self.args = args
        self.train_loss = AverageMeter()
        self.dev_loss = AverageMeter()
        self.step = 0
        self.updates = 0
        self.network = network

        self.mnetwork = nn.DataParallel(self.network) if args.gpu_num > 1 else self.network

        if self.args.gpu_num > 0:
            self.network.cuda()

    def avg_reset(self):
        self.train_loss.reset()
        self.dev_loss.reset()

    @torch.no_grad()
    def evaluate(self, dev_data_list, epoch):
        dev_data_list.reset()
        self.network.eval()
        for batch in tqdm(dev_data_list):
            output_dict = self.network(**batch, mode="eval", epoch=epoch)
            loss = output_dict["loss"].item()
            self.dev_loss.update(loss, 1)
        self.network.train()

    @torch.no_grad()
    def predict(self, test_data_list):
        test_data_list.reset()
        self.network.eval()
        pred_json = {}
        for batch in tqdm(test_data_list):
            output_dict = self.network.predict(**batch, mode="eval")
            pred_answer = output_dict["prediction"]
            pred_scale = output_dict["scale"]
            question_id = output_dict["question_id"]
            for i in range(len(question_id)):
                pred_json[question_id[i]] = (pred_answer[i], pred_scale[i])
        return pred_json

    def reset(self):
        self.mnetwork.reset()

    def get_df(self):
        return self.mnetwork.get_df()

    def get_metrics(self, logger=None):
        return self.mnetwork.get_metrics_predict(logger, True)

    def load(self, state_path):
        state_path = state_path + ".pt"
        self.network.load_state_dict(torch.load(state_path))


class ETRFineTuningModel():
    def __init__(self, args, network, state_dict=None, num_train_steps=1):
        self.args = args
        self.train_loss = AverageMeter()
        self.dev_loss = AverageMeter()
        self.head_acc = AverageMeter()
        self.step = 0
        self.updates = 0
        self.network = network
        self.loss_dic = {}
        if state_dict is not None:
            print("Load Model!")
            self.network.load_state_dict(state_dict["state"])
        self.mnetwork = nn.DataParallel(self.network) if args.gpu_num > 1 else self.network

        self.total_param = sum([p.nelement() for p in self.network.parameters() if p.requires_grad])
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.network.lm.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.bert_weight_decay, 'lr': args.bert_learning_rate},
            {'params': [p for n, p in self.network.lm.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.bert_learning_rate},
            {'params': [p for n, p in self.network.train_tree.named_parameters()],
             'weight_decay': args.tree_weight_decay, 'lr': args.tree_learning_rate},
            {'params': [p for n, p in self.network.named_parameters() if not n.startswith("lm.") and not n.startswith("train_tree.")],
             "weight_decay": args.weight_decay, "lr": args.learning_rate}
        ]
        self.optimizer = Adam(optimizer_parameters,
                              lr=args.learning_rate,
                              warmup=args.warmup,
                              t_total=num_train_steps,
                              max_grad_norm=args.grad_clipping,
                              schedule=args.warmup_schedule)
        if self.args.gpu_num > 0:
            self.network.cuda()

    def avg_reset(self):
        self.train_loss.reset()
        self.dev_loss.reset()
        self.head_acc.reset()

    def update(self, tasks):
        self.network.train()
        output_dict = self.mnetwork(**tasks)
        loss = output_dict["loss"]
        self.train_loss.update(loss.item(), 1)
        acc = output_dict["head_acc"]
        loss_d = output_dict['loss_sp']
        self.loss_dic.update(loss_d)
        self.head_acc.update(acc, 1)
        if self.args.gradient_accumulation_steps > 1:
            loss /= self.args.gradient_accumulation_steps
        loss.backward()
        if (self.step + 1) % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.updates += 1
        self.step += 1

    @torch.no_grad()
    def evaluate(self, dev_data_list):
        dev_data_list.reset()
        self.network.eval()
        with torch.no_grad():
            for batch in tqdm(dev_data_list):
                output_dict = self.network(**batch, mode='eval')
                loss = output_dict["loss"].item()
                self.dev_loss.update(loss, 1)
                acc = output_dict["head_acc"]
                self.head_acc.update(acc, 1)
        # self.network.train()

    # @torch.no_grad()
    # def predict(self, test_data_list):
    #     test_data_list.reset()
    #     self.network.eval()
    #     # pred_json = {}
    #     for batch in tqdm(test_data_list):
    #         self.network.predict(**batch, mode="eval")

    def reset(self):
        self.mnetwork.reset()

    def get_df(self):
        return self.mnetwork.get_df()

    def get_metrics(self, logger=None):
        return self.mnetwork.get_metrics(logger, True)

    def get_metrics_predict(self, logger=None):
        return self.mnetwork.get_metrics_predict(logger, True)
    
    def save(self, prefix, epoch):
        network_state = dict([(k, v.cpu()) for k, v in self.network.state_dict().items()])
        other_params = {
            'optimizer': self.optimizer.state_dict(),
            'config': self.args,
            'epoch': epoch
        }
        state_path = prefix + ".pt"
        other_path = prefix + ".ot"
        torch.save(other_params, other_path)
        torch.save(network_state, state_path)
        print('model saved to {}'.format(prefix))

    def load(self, prefix): 
        state_path = prefix + ".pt"
        other_path = prefix + ".ot"
        self.network.load_state_dict(torch.load(state_path))
        self.optimizer.load_state_dict(torch.load(other_path)['optimizer'])