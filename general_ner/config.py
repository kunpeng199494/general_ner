# -*- coding:utf-8 -*-

import os
import sys
import hao
root_dir = hao.paths.project_root_path()
sys.path.append(root_dir)
import torch


class Config(object):
    def __init__(self):
        self.label_file = os.path.join(root_dir, "data/process_data/training_data/labels.txt")
        self.train_file = os.path.join(root_dir, "data/process_data/training_data/train.txt")
        self.dev_file = os.path.join(root_dir, "data/process_data/training_data/dev.txt")
        self.test_file = os.path.join(root_dir, "data/process_data/training_data/test.txt")
        self.model_name = "2021_01_25_BERT_LSTM_CRF"
        self.log_path = os.path.join(root_dir, f"data/logs/tensorboard_logs/2021_01_25_{self.model_name}")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path, exist_ok=True)
        self.bert_path = os.path.join(root_dir, "data/bert")
        self.vocab = os.path.join(root_dir, "data/bert/vocab.txt")
        self.bert_embedding = 768
        self.bert_layers_num = 1
        self.num_attention_heads = 4
        self.attention_dim = 768
        self.freeze = True
        self.use_cuda = True
        self.output_attentions = False
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.layer_norm_eps = 1e-12
        self.bert_mode = "weighted"
        self.head_mask = torch.LongTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # self.head_mask = None
        self.gpu = 1
        self.batch_size = 16
        self.max_length = 128
        self.rnn_hidden = 128
        self.dropout1 = 0.5
        self.rnn_layer = 1
        self.dropout_ratio = 0.5 if self.rnn_layer > 1 else 0
        self.lr = 0.0001
        self.lr_decay = 0.00001
        self.weight_decay = 0.00005
        self.optim = "Adam"
        self.save_path = os.path.join(root_dir, f"data/save_model/{self.model_name}.pt")
        self.base_epoch = 100
        self.early_stop = 1000
        self.warmup_ratio = 0.1
        self.device = torch.device("cuda") if self.use_cuda and torch.cuda.is_available() else torch.device("cpu")
        self.map_location = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return "\n".join(["%s:%s" % item for item in self.__dict__.items()])


if __name__ == "__main__":
    con = Config()
    con.update(gpu=0)
    print(con.gpu)
    print(con)
