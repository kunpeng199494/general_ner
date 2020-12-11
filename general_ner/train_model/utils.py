# -*- coding:utf-8 -*-

import torch
import os
import tqdm
import datetime
import unicodedata


class InputFeatures(object):
    def __init__(self, input_id, label_id, input_mask):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask


class TestFeatures(object):
    def __init__(self, input_id, input_mask):
        self.input_id = input_id
        self.input_mask = input_mask


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def read_corpus(path, max_length, label_dic, vocab):
    """
    :param path:数据文件路径
    :param max_length: 最大长度
    :param label_dic: 标签字典
    :return:
    """
    file = open(path, encoding="utf-8")
    content = file.readlines()
    file.close()
    result = []
    tokens = []
    label = []
    for line in tqdm.tqdm(content, total=len(content)):
        line = line.strip()
        if len(line) > 0:
            if len(line) == 1:
                continue
            tokens.append(line.split()[0])
            label.append(line.split()[1])
            continue
        else:
            if len(tokens) > max_length-2:
                tokens = tokens[0:(max_length-2)]
                label = label[0:(max_length-2)]
            tokens_f = ["[CLS]"] + tokens + ["[SEP]"]
            label_f = ["<start>"] + label + ["<eos>"]
            input_ids = [int(vocab[i]) if i in vocab else int(vocab["[UNK]"]) for i in tokens_f]
            label_ids = [label_dic[i] for i in label_f]
            input_mask = [1] * len(tokens_f)
            while len(input_ids) < max_length:
                input_ids.append(int(vocab["[PAD]"]))
                input_mask.append(0)
                label_ids.append(label_dic["<pad>"])
            assert len(input_ids) == max_length
            assert len(input_mask) == max_length
            assert len(label_ids) == max_length
            feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=label_ids)
            result.append(feature)
            tokens = []
            label = []
    return result


def process_line(line, max_length, vocab):
    tokens = list(line)
    if len(tokens) > max_length - 2:
        tokens = tokens[0:(max_length - 2)]
    tokens_f = ["[CLS]"] + tokens + ["[SEP]"]
    input_ids = [int(vocab[i]) if i in vocab else int(vocab["[UNK]"]) for i in tokens_f]
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_length:
        input_ids.append(int(vocab["[PAD]"]))
        input_mask.append(0)
    assert len(input_ids) == max_length
    assert len(input_mask) == max_length
    feature = TestFeatures(input_id=input_ids, input_mask=input_mask)
    return feature, tokens_f


def ids_labels(config):
    label_dic = load_vocab(config.label_file)

    ids2labels = {}
    for k, v in label_dic.items():
        ids2labels[v] = k

    return ids2labels
