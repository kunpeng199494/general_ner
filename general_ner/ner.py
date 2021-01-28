# -*- coding:utf-8 -*-

import os
import sys
import hao
root_dir = hao.paths.project_root_path()
sys.path.append(root_dir)
import time
import torch
import torch.nn as nn
from genaral_ner.config import Config
from genaral_ner.ner_models import *
import torch.optim as optim
from genaral_ner.utils import load_vocab, read_corpus, process_line, ids_labels
from genaral_ner.data_loader import data_load, batch_predict_loader
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn import metrics
from transformers import get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from genaral_ner.metrics import span_classification_report

LOGGER = hao.logs.get_logger(__name__)


def init_model(config):
    model = BERT_LSTM_CRF(config,
                          config.bert_embedding,
                          config.rnn_hidden,
                          config.rnn_layer,
                          dropout_ratio=config.dropout_ratio,
                          dropout1=config.dropout1)

    # model = BERT_CRF(config,
    #                  config.bert_embedding,
    #                  config.dropout1)

    # model = BERT_LSTM_ATTN_CRF(config,
    #                            config.bert_embedding,
    #                            config.rnn_hidden,
    #                            config.rnn_layer,
    #                            config.dropout_ratio,
    #                            config.dropout1)

    # model = BERT_ATTN_CRF(config,
    #                       config.bert_embedding,
    #                       config.dropout1)
    head_mask = None if config.head_mask is None else config.head_mask.to(config.device)
    return model, head_mask


def init_optimizer(config, model, optimizer):
    # 给不同的网络层设置不同的学习率，主要是要注意CRF层
    # 自行测试rnn输出层得分和crf中转移得分的数据相差大概300~700倍
    # https://zhuanlan.zhihu.com/p/106654565
    # param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #      'weight_decay': config.weight_decay},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # optimizer = optimizer(optimizer_grouped_parameters, config.lr)

    weight_decay_and_no_crf_parameters = []
    no_weight_decay_and_no_crf_parameters = []
    weight_decay_and_crf_parameters = []
    no_weight_decay_and_crf_parameters = []

    crf_keyword = 'crf'
    for n, p in model.named_parameters():
        if crf_keyword not in n:
            if not any(nd in n for nd in no_decay):
                weight_decay_and_no_crf_parameters.append(p)
            else:
                no_weight_decay_and_no_crf_parameters.append(p)
        else:
            if not any(nd in n for nd in no_decay):
                weight_decay_and_crf_parameters.append(p)
            else:
                no_weight_decay_and_crf_parameters.append(p)

    optimizer_grouped_parameters = []

    if len(weight_decay_and_no_crf_parameters) != 0:
        optimizer_grouped_parameters.append({
            "params": weight_decay_and_no_crf_parameters,
            "weight_decay": config.weight_decay
        })

    if len(no_weight_decay_and_no_crf_parameters) != 0:
        optimizer_grouped_parameters.append({
            "params": no_weight_decay_and_no_crf_parameters,
            "weight_decay": 0.0
        })

    crf_lr = min(5e-2, config.lr * 1000)
    if len(weight_decay_and_crf_parameters) != 0:
        optimizer_grouped_parameters.append({
            "params": weight_decay_and_crf_parameters,
            "weight_decay": config.weight_decay,
            "lr": crf_lr
        })

    if len(no_weight_decay_and_crf_parameters) != 0:
        optimizer_grouped_parameters.append({
            "params": no_weight_decay_and_crf_parameters,
            "weight_decay": 0.0,
            "lr": crf_lr
        })

    optimizer = optimizer(optimizer_grouped_parameters, config.lr)

    return optimizer


def init_scheduler(config, optimizer, train_loader):
    # 这个地方的设置不算特别合理，因为不一定预设的epoch全部都能跑完，因此可能学习率并未降到预定的水准，看到具体项目可以进行更精细的调参
    t_total = len(train_loader) * config.base_epoch
    warmup_steps = t_total * config.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    return scheduler


def train_from_blank(config):
    # 初始加载训练数据，模型，并将模型送入训练设备
    LOGGER.info("Loading train_dev data ...")
    train_loader, dev_loader, test_loader = data_load(config)
    labels = list(load_vocab(config.label_file).keys())

    LOGGER.info("Initialization model ...")
    model, head_mask = init_model(config)

    LOGGER.info("Training model ...")
    model.to(config.device)
    model.train()

    # 初始化优化器和线性权重衰减器
    optimizer = getattr(optim, config.optim)
    optimizer = init_optimizer(config, model, optimizer)
    scheduler = init_scheduler(config, optimizer, train_loader)

    # 一些用于记录训练过程的标记，包括了eval_loss，指标有无提升，是否提前结束训练等
    eval_loss = float("inf")
    total_step = 0
    total_batch = 0
    last_improve = 0
    improve = "*"
    flag = False

    # 初始化summary_writer
    writer = SummaryWriter(log_dir=config.log_path)

    # 写循环开始训练过程
    for epoch in range(config.base_epoch):
        step = 0
        for i, batch in enumerate(train_loader):
            step += 1
            total_step += 1
            optimizer.zero_grad()
            inputs, masks, tags = batch
            feats = model(inputs, masks, head_mask)
            loss = model.loss(feats, masks, tags)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 100 == 0:
                path_score, best_path = model.crf(feats, masks.byte())
                pred = []
                true = []
                for b, t in zip(best_path, tags):
                    length = len(b)
                    pred.extend(b)
                    true.extend(t.cpu().tolist()[:length])
                train_acc = metrics.accuracy_score(true, pred)
                loss_temp, dev_acc = dev(model, dev_loader, head_mask)

                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", loss_temp, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)

                train_acc = str(train_acc * 100.)[:5] + "%"
                dev_acc = str(dev_acc * 100.)[:5] + "%"
                LOGGER.info("epoch: {}  step: {}  train_loss: {}  dev_loss: {}  train_acc: {}  dev_acc: {}  {}"
                            .format(epoch, step, str(loss.item())[:6], str(loss_temp)[:6], train_acc, dev_acc, improve))

                if loss_temp < eval_loss:
                    eval_loss = loss_temp
                    torch.save(model.state_dict(), config.save_path)
                    improve = "*"
                    last_improve = total_step
                else:
                    improve = ""
            total_batch += 1
            if total_step - last_improve > config.early_stop:
                LOGGER.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test_loss, test_acc, report, span_report = dynamic_test(config, model, test_loader, labels, head_mask)
    test_acc = str(test_acc.item() * 100.)[:5] + "%"
    LOGGER.info("test_loss: {}".format(str(test_loss)[:6]))
    LOGGER.info("test_acc: {}".format(test_acc))
    LOGGER.info(report)
    LOGGER.info(span_report)


def train_from_saved(config):
    # 初始加载训练数据、初始化模型
    LOGGER.info("Loading train_dev data ...")
    train_loader, dev_loader, test_loader = data_load(config)
    labels = list(load_vocab(config.label_file).keys())

    LOGGER.info("Initialization model ...")
    model, head_mask = init_model(config)

    # 从硬盘中加载之前训练到某个程度的模型
    assert config.save_path is not None
    if config.use_cuda:
        model.load_state_dict(torch.load(config.save_path))
    else:
        model.load_state_dict(torch.load(config.save_path, map_location=torch.device("cpu")))
    LOGGER.info("Load model {} successfully".format(os.path.split(config.save_path)[1]))

    # 将模型送入训练设备，并打开训练开关
    LOGGER.info("Training model ...")
    model.to(config.device)
    model.train()

    # 初始化优化器和线性权重衰减器
    optimizer = getattr(optim, config.optim)
    optimizer = init_optimizer(config, model, optimizer)
    scheduler = init_scheduler(config, optimizer, train_loader)

    # 一些用于记录训练过程的标记，包括了eval_loss，指标有无提升，是否提前结束训练等
    eval_loss = float("inf")
    total_step = 0
    total_batch = 0
    last_improve = 0
    improve = "*"
    flag = False

    # 初始化summary_writer
    writer = SummaryWriter(log_dir=config.log_path)

    # 写循环开始训练过程
    for epoch in range(config.base_epoch):
        step = 0
        for i, batch in enumerate(train_loader):
            step += 1
            total_step += 1
            optimizer.zero_grad()
            inputs, masks, tags = batch
            feats = model(inputs, masks, head_mask)
            loss = model.loss(feats, masks, tags)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 100 == 0:
                path_score, best_path = model.crf(feats, masks.byte())
                pred = []
                true = []
                for b, t in zip(best_path, tags):
                    length = len(b)
                    pred.extend(b)
                    true.extend(t.cpu().tolist()[:length])
                train_acc = metrics.accuracy_score(true, pred)
                loss_temp, dev_acc = dev(model, dev_loader, head_mask)

                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", loss_temp, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)

                train_acc = str(train_acc * 100.)[:5] + "%"
                dev_acc = str(dev_acc * 100.)[:5] + "%"
                LOGGER.info("epoch: {}  step: {}  train_loss: {}  dev_loss: {}  train_acc: {}  dev_acc: {}  {}"
                            .format(epoch, step, str(loss.item())[:6], str(loss_temp)[:6], train_acc, dev_acc, improve))

                if loss_temp < eval_loss:
                    eval_loss = loss_temp
                    torch.save(model.state_dict(), config.save_path)
                    improve = "*"
                    last_improve = total_step
                else:
                    improve = ""
            total_batch += 1
            if total_step - last_improve > config.early_stop:
                LOGGER.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test_loss, test_acc, report, span_report = dynamic_test(model, test_loader, labels, head_mask)
    test_acc = str(test_acc.item() * 100.)[:5] + "%"
    LOGGER.info("test_loss: {}".format(str(test_loss)[:6]))
    LOGGER.info("test_acc: {}".format(test_acc))
    LOGGER.info(report)
    LOGGER.info(span_report)


def dev(model, dev_loader, head_mask):
    model.eval()
    eval_loss = 0
    batches = 0
    true = []
    pred = []
    for i, batch in enumerate(dev_loader):
        inputs, masks, tags = batch
        batches += 1
        feats = model(inputs, masks, head_mask)
        path_score, best_path = model.crf(feats, masks)
        loss = model.loss(feats, masks, tags)
        eval_loss += loss.item()
        for b, t in zip(best_path, tags):
            length = len(b)
            pred.extend(b)
            true.extend(t.cpu().tolist()[:length])
    model.train()
    dev_acc = metrics.accuracy_score(true, pred)
    return eval_loss / batches, dev_acc


def dynamic_test(config, model, test_loader, labels, head_mask):
    labels = labels[1:]

    ids2labels = ids_labels(config)

    model.eval()

    eval_loss = 0
    batches = 0
    true = []
    pred = []
    span_true = []
    span_pred = []
    for i, batch in enumerate(test_loader):
        inputs, masks, tags = batch
        batches += 1
        feats = model(inputs, masks, head_mask)
        path_score, best_path = model.crf(feats, masks.byte())
        loss = model.loss(feats, masks, tags)
        eval_loss += loss.item()
        for b, t in zip(best_path, tags):
            length = len(b)
            pred.extend(b)
            true.extend(t.cpu().tolist()[:length])
            mid_true = [ids2labels[t] for t in t.cpu().tolist()[:length]]
            mid_pred = [ids2labels[b] for b in b]
            span_true.append(mid_true)
            span_pred.append(mid_pred)
    test_acc = metrics.accuracy_score(true, pred)
    report = metrics.classification_report(true, pred, target_names=labels)
    span_report = span_classification_report(span_true, span_pred, labels=["sentiment"])
    return eval_loss / batches, test_acc, report, span_report


def static_test(config):
    labels = list(load_vocab(config.label_file).keys())

    ids2labels = ids_labels(config)

    model, head_mask = init_model(config)

    if config.use_cuda:
        model.load_state_dict(torch.load(config.save_path))

    else:
        model.load_state_dict(torch.load(config.save_path, map_location=torch.device("cpu")))

    model.to(config.device)
    model.eval()

    test_loader = batch_predict_loader(config)[1]

    eval_loss = 0
    batches = 0
    true = []
    pred = []
    span_true = []
    span_pred = []
    for i, batch in enumerate(test_loader):
        inputs, masks, tags = batch
        batches += 1
        feats = model(inputs, masks, head_mask)
        path_score, best_path = model.crf(feats, masks.byte())
        loss = model.loss(feats, masks, tags)
        eval_loss += loss.item()
        for b, t in zip(best_path, tags):
            length = len(b)
            pred.extend(b)
            true.extend(t.cpu().tolist()[:length])
            mid_true = [ids2labels[t] for t in t.cpu().tolist()[:length]]
            mid_pred = [ids2labels[b] for b in b]
            span_true.append(mid_true)
            span_pred.append(mid_pred)
    test_acc = metrics.accuracy_score(true, pred)
    report = metrics.classification_report(true, pred, target_names=labels)
    print(eval_loss / batches, test_acc, report)
    # print(span_classification_report(span_true, span_pred, labels=["sentiment"]))
    print(span_classification_report(span_true, span_pred))


def load_model_for_predict(config):

    model, head_mask = init_model(config)

    if config.use_cuda:
        model.load_state_dict(torch.load(config.save_path))

    else:
        model.load_state_dict(torch.load(config.save_path, map_location=torch.device("cpu")))

    model.to(config.device)
    model.eval()

    return model, head_mask


def predict_line(config, model, line, head_mask):

    vocab = load_vocab(config.vocab)

    ids2labels = ids_labels(config)

    start = time.time()

    predict_data, tokens = process_line(line, max_length=config.max_length, vocab=vocab)

    inputs = torch.LongTensor([predict_data.input_id]).to(config.device)
    masks = torch.LongTensor([predict_data.input_mask]).to(config.device)

    feats = model(inputs, masks, head_mask)
    path_score, best_path = model.crf(feats, masks.bool())
    labels = [ids2labels[t] for t in best_path[0]]

    words = []
    for i in range(len(labels)):
        s = ""
        if labels[i] == "<eos>":
            break
        if i == len(tokens):
            break
        if labels[i] == "B-sentiment":
            s += tokens[i]
            for j in range(i+1, len(labels)):
                if labels[j] == "I-sentiment":
                    s += tokens[j]
                else:
                    words.append(s)
                    break

    diff = time.time() - start

    return words, diff


def batch_predict(config):
    model, head_mask = init_model(config)

    if config.use_cuda:
        model.load_state_dict(torch.load(config.save_path))

    else:
        model.load_state_dict(torch.load(config.save_path, map_location=torch.device("cpu")))

    model.to(device)
    model.eval()

    test_loader = batch_predict_loader(config)[0]

    pred = []
    for i, batch in enumerate(test_loader):
        inputs, masks = batch
        feats = model(inputs, masks, head_mask)
        path_score, best_path = model.crf(feats, masks.bool())
        for line in best_path.tolist():
            pred.append([ids2labels[t] for t in line])
    return pred


if __name__ == "__main__":
    # Load training config ...
    model_config = Config()
    LOGGER.info("Config is :".format(model_config))
    if model_config.use_cuda:
        torch.cuda.set_device(model_config.gpu)

    train_from_blank(model_config)

    # train_from_saved(model_config)

    model_, head_mask_ = load_model_for_predict(model_config)

    # static_test(model_config)

    # test_loss, test_acc, report = static_test(model_config)
    # test_acc = str(test_acc * 100.)[:5] + "%"
    # print("test_loss: {}".format(str(test_loss)[:6]))
    # print("test_acc: {}".format(test_acc))
    # print(report)

    texts = ["台北市防疫中心這段防疫期間，可以規定外籍移工假日，不要聚集北車，怕群聚感染漫延開來。(可用網路聚會，用手機開群組聊天)",
             "Jilly Lai 確診前因未知而趴趴走事後才確診,這是隱憂,而另種居家隔離雖罰,還趴趴走,這也是隱憂,所以有點恐怖啦但新北市府的防疫方式能穩定民心",
             "第32例移工外勞染疫新聞事件，造成人人都恐慌害怕!但我在捷運上還是看到很多人搭捷運不戴口罩的……非常時期實在好可怕。",
             "還好板橋車站沒有印尼移工的群聚∼真是太好了。第一次看到他們群聚在台北車站時，我還以為發生什麼大事，而且他們還直接坐在地上，同行的國外朋友也問我發生何事，當時我楞住了一下只能回說不知道。現在發生第32例的事,政府也應該好好處理非法外勞與外勞群聚的地點，替他們找個開放空間又不影響交通與觀感處。"]
    for text in texts:
        word, spend = predict_line(model_config, model_, text, head_mask_)
        print(word)
        print(spend)
