# -*- coding:utf-8 -*-

import torch.nn as nn
from general_ner.embedders import BertEmbedder
# from general_ner.NCRF import CRF
from general_ner.CRF import CRF
from general_ner.attention import Attn
import torch
from general_ner.utils import load_vocab


class BERT_LSTM_CRF(nn.Module):
    def __init__(self,
                 config,
                 embedding_dim,
                 hidden_dim,
                 rnn_layers,
                 dropout_ratio,
                 dropout1):

        super(BERT_LSTM_CRF, self).__init__()
        self.device = config.device
        self.num_tags = len(load_vocab(config.label_file))
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.encoder = BertEmbedder.create()

        self.lstm = nn.LSTM(self.embedding_dim,
                            self.hidden_dim,
                            num_layers=rnn_layers,
                            bidirectional=True,
                            dropout=dropout_ratio,
                            batch_first=True)
        self.rnn_layers = rnn_layers

        self.dropout1 = nn.Dropout(p=dropout1)
        self.linear = nn.Linear(hidden_dim * 2, self.num_tags)

        self.crf = CRF(num_tags=self.num_tags, device=self.device)

        self.init_weights()

    def init_weights(self):
        for p in self.lstm.parameters():
            if len(p.shape) >= 2:
                nn.init.xavier_normal_(p)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        embeds = self.encoder(input_ids, attention_mask=attention_mask, head_mask=head_mask)

        lens = attention_mask.sum(1)
        sorted_lengths, sorted_idx = torch.sort(lens, descending=True)
        embeds = embeds[sorted_idx]

        embeds = nn.utils.rnn.pack_padded_sequence(embeds, sorted_lengths.data.tolist(), batch_first=True)
        lstm_out, hidden = self.lstm(embeds)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        lstm_out = lstm_out[reversed_idx]

        batch_size, seq_length, input_dim = lstm_out.size()
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim * 2)
        d_lstm_out = self.dropout1(lstm_out)
        l_out = self.linear(d_lstm_out)
        lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)
        return lstm_feats

    def loss(self, feats, mask, tags):
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tags)
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value


class BERT_CRF(nn.Module):
    def __init__(self, config, embed_dim, dropout1):
        super(BERT_CRF, self).__init__()
        self.device = config.device
        self.num_tags = len(load_vocab(config.label_file))
        self.embed_dim = embed_dim
        self.encoder = BertEmbedder.create()
        self.dropout1 = nn.Dropout(p=dropout1)
        self.linear = nn.Linear(self.embed_dim, self.num_tags)
        self.crf = CRF(num_tags=self.num_tags, device=self.device)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        embeds = self.encoder(input_ids, attention_mask=attention_mask, head_mask=head_mask)
        # embeds: [batch_size, sequence_length, embedding_dim]
        batch_size, sequence_length, embed_dim = embeds.size()
        embeds = embeds.contiguous().view(-1, self.embed_dim)
        embeds_dropout = self.dropout1(embeds)
        linear_output = self.linear(embeds_dropout)
        output = linear_output.contiguous().view(batch_size, sequence_length, -1)
        return output

    def loss(self, feats, mask, tags):
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tags)
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value


class BERT_LSTM_ATTN_CRF(nn.Module):
    def __init__(self,
                 config,
                 embedding_dim,
                 hidden_dim,
                 rnn_layers,
                 dropout_ratio,
                 dropout1):
        super(BERT_LSTM_ATTN_CRF, self).__init__()
        self.device = config.device
        self.hidden_dim = hidden_dim
        self.num_tags = len(load_vocab(config.label_file))
        self.encoder = BertEmbedder.create()
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=rnn_layers,
                            batch_first=True,
                            dropout=dropout_ratio,
                            bidirectional=True)
        config.update(attention_dim=self.hidden_dim * 2)
        self.attention = Attn(config)
        self.linear = nn.Linear(self.hidden_dim * 2, self.num_tags)
        self.dropout = nn.Dropout(dropout1)
        self.crf = CRF(num_tags=self.num_tags, device=self.device)
        self.init_weights()

    def init_weights(self):
        for p in self.lstm.parameters():
            if len(p.shape) >= 2:
                nn.init.xavier_normal_(p)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        embeds = self.encoder(input_ids, attention_mask=attention_mask, head_mask=head_mask)

        lens = attention_mask.sum(1)
        sorted_lengths, sorted_idx = torch.sort(lens, descending=True)
        embeds = embeds[sorted_idx]

        embeds = nn.utils.rnn.pack_padded_sequence(embeds, sorted_lengths.data.tolist(), batch_first=True)
        lstm_out, hidden = self.lstm(embeds)
        lstm_out, hidden = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx, descending=True)
        lstm_out = lstm_out[reversed_idx]

        attn_out = self.attention.forward(lstm_out, attention_mask=attention_mask[:, :lstm_out.size(1)])[0]

        batch_size, sequence_length, hidden_dim = attn_out.size()
        attn_out = attn_out.contiguous().view(-1, self.hidden_dim * 2)
        attn_out = self.dropout(attn_out)
        linear_output = self.linear(attn_out)
        output = linear_output.contiguous().view(batch_size, sequence_length, -1)
        return output

    def loss(self, feats, mask, tags):
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tags)
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value


class BERT_ATTN_CRF(nn.Module):
    def __init__(self,
                 config,
                 embedding_dim,
                 dropout1):
        super(BERT_ATTN_CRF, self).__init__()
        self.device = config.device
        self.num_tags = len(load_vocab(config.label_file))
        self.encoder = BertEmbedder.create()
        self.attention = Attn(config)
        self.dropout = nn.Dropout(dropout1)
        config.update(attention_dim=embedding_dim)
        self.linear = nn.Linear(embedding_dim, self.num_tags)
        self.crf = CRF(num_tags=self.num_tags, device=self.device)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        embeds = self.encoder(input_ids, attention_mask=attention_mask, head_mask=head_mask)
        # embeds: [batch_size, sequence_length, embedding_dim]
        attn_out = self.attention.forward(embeds, attention_mask=attention_mask[:, :embeds.size(1)])[0]
        batch_size, sequence_length, embedding_dim = attn_out.size()
        attn_out = attn_out.contiguous().view(-1, embedding_dim)
        attn_out = self.dropout(attn_out)
        linear_output = self.linear(attn_out)
        output = linear_output.contiguous().view(batch_size, sequence_length, -1)
        return output

    def loss(self, feats, mask, tags):
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tags)
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value
