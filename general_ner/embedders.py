# -*- coding:utf-8 -*-

import torch
import os
import codecs
import json
from torch import nn
from hao.logs import get_logger
from transformers import BertConfig
from general_ner.bert_crop import BertCropModel
from general_ner.config import Config

model_config = Config()

logger = get_logger(__name__)


class BertEmbedder(nn.Module):

    def __init__(self, model: BertCropModel):
        super().__init__()
        self.model = model
        if model_config.bert_mode == "weighted":
            self.bert_weights = nn.Parameter(torch.FloatTensor(model_config.bert_layers_num, 1))
            self.bert_gamma = nn.Parameter(torch.FloatTensor(1, 1))

        if model_config.use_cuda:
            self.cuda()

        self.init_weights()

    def init_weights(self):
        if model_config.bert_mode == "weighted":
            nn.init.xavier_normal_(self.bert_gamma)
            nn.init.xavier_normal_(self.bert_weights)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        encoder_output = self.model(input_ids, attention_mask=attention_mask, head_mask=head_mask)
        if model_config.bert_mode == "last":
            return encoder_output[0]
        elif model_config.bert_mode == "weighted":
            all_encoder_layers = torch.stack(
                [a * b for a, b in zip(encoder_output[1][:model_config.bert_layers_num], self.bert_weights)])
            return self.bert_gamma * torch.sum(all_encoder_layers, dim=0)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    # 冻结后面的层
    def freeze_to(self, to=-1):
        idx = 0
        if to < 0:
            to = len(self.model.encoder.layer) + to + 1
        for idx in range(to):
            for param in self.model.encoder.layer[idx].parameters():
                param.requires_grad = False
        logger.info("Embeddings freezed to {}".format(to))
        to = len(self.model.encoder.layer)
        for idx in range(idx, to):
            for param in self.model.encoder.layer[idx].parameters():
                param.requires_grad = True

    def get_n_trainable_params(self):
        pp = 0
        for p in list(self.parameters()):
            if p.requires_grad:
                num = 1
                for s in list(p.size()):
                    num = num * s
                pp += num
        return pp

    @classmethod
    def create(cls):

        logger.info('Loading pretrained bert model!')
        bert_config_file = os.path.join(model_config.bert_path, "config.json")
        init_checkpoint_pt = os.path.join(model_config.bert_path, "pytorch_model.bin")

        bert_config = BertConfig.from_json_file(bert_config_file)
        logger.info("Model config {}".format(bert_config))
        model = BertCropModel(bert_config, model_config.bert_layers_num)

        state_dict = torch.load(init_checkpoint_pt, model_config.map_location)
        model.init_from_pretrained(state_dict)

        model = model.to(model_config.device)
        model = cls(model)
        if model_config.freeze:
            model.freeze()
        if not model_config.freeze:
            model.unfreeze()
        return model
