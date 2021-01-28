# -*- coding: utf-8 -*-

import re
import os
import sys
import hao
root_dir = hao.paths.project_root_path()
sys.path.append(root_dir)
import torch
from torch.utils.data import DataLoader, TensorDataset
from general_ner.utils import load_vocab, read_corpus, process_line


def data_load(config):
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.label_file)

    train_data = read_corpus(config.train_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    dev_data = read_corpus(config.dev_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    test_data = read_corpus(config.test_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)

    train_ids = torch.LongTensor([temp.input_id for temp in train_data]).to(config.device)
    train_masks = torch.LongTensor([temp.input_mask for temp in train_data]).to(config.device)
    train_tags = torch.LongTensor([temp.label_id for temp in train_data]).to(config.device)

    train_dataset = TensorDataset(train_ids, train_masks, train_tags)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)

    dev_ids = torch.LongTensor([temp.input_id for temp in dev_data]).to(config.device)
    dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data]).to(config.device)
    dev_tags = torch.LongTensor([temp.label_id for temp in dev_data]).to(config.device)

    dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=config.batch_size)

    test_ids = torch.LongTensor([temp.input_id for temp in test_data]).to(config.device)
    test_masks = torch.LongTensor([temp.input_mask for temp in test_data]).to(config.device)
    test_tags = torch.LongTensor([temp.label_id for temp in test_data]).to(config.device)

    test_dataset = TensorDataset(test_ids, test_masks, test_tags)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size)

    return train_loader, dev_loader, test_loader


def batch_predict_loader(config):
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.label_file)

    test_data = read_corpus(config.test_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)

    test_ids = torch.LongTensor([temp.input_id for temp in test_data]).to(config.device)
    test_masks = torch.LongTensor([temp.input_mask for temp in test_data]).to(config.device)
    test_tags = torch.LongTensor([temp.label_id for temp in test_data]).to(config.device)

    batch_predict_dataset = TensorDataset(test_ids, test_masks)
    batch_predict_load = DataLoader(batch_predict_dataset, shuffle=True, batch_size=config.batch_size)

    test_dataset = TensorDataset(test_ids, test_masks, test_tags)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size)

    return batch_predict_load, test_loader
