# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertLayer, BertEmbeddings
import copy
from transformers import BertConfig
from hao.logs import get_logger
import re

BertLayerNorm = torch.nn.LayerNorm


class BertCropModel(nn.Module):

    def __init__(self, config: BertConfig, num_hidden_layers=None):
        super().__init__()
        self.logger = get_logger(__name__)
        config.output_hidden_states = True
        self.embeddings = BertEmbeddings(config)
        num_hidden_layers = config.num_hidden_layers if num_hidden_layers is None else num_hidden_layers
        assert num_hidden_layers > 0, 'bert_layers must > 0'

        # 需要注意的是和原始transformer的BERT_Encoder的输出不一样
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_hidden_layers)])
        self.config = config
        self.num_hidden_layers = num_hidden_layers
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            attention_mask=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            past_key_values=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None
    ):
        global encoder_extended_attention_mask
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape,
                                                                                                        attention_mask.shape))

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)

            if next(self.parameters()).dtype == torch.float16:
                encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
            elif next(self.parameters()).dtype == torch.float32:
                encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9

            else:
                raise ValueError(
                    "{} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`".format(
                        next(self.parameters()).dtype
                    )
                )

        else:
            encoder_extended_attention_mask = None

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids,
                                           token_type_ids=token_type_ids, inputs_embeds=inputs_embeds,
                                           past_key_values_length=past_key_values_length)

        hidden_states = embedding_output

        all_hidden_states = ()
        all_attentions = ()

        attention_mask = extended_attention_mask
        encoder_attention_mask = encoder_extended_attention_mask
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states,
                                         encoder_attention_mask, past_key_value, self.output_attentions,)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        # outputs meaning: # last-layer hidden state, (all hidden states), (all attentions)
        return outputs

    def init_from_pretrained(
            self,
            pretrained_state_dict
    ):
        state_dict = pretrained_state_dict

        unused_key_p = re.compile('encoder\.layer\.(\d{1,2}).')
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []

        del_keys = []
        for key in state_dict.keys():
            do_pop = False
            new_key = None

            if 'pooler' in key:
                del_keys.append(key)
                do_pop = True

            m = unused_key_p.findall(key)
            for num in m:
                if int(num) >= self.num_hidden_layers:
                    del_keys.append(key)
                    do_pop = True
                else:
                    # 删除encoder属性
                    new_key = key.replace('encoder.', '')

            if do_pop:
                continue

            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for key in del_keys:
            state_dict.pop(key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module: nn.Module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(self, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(self, prefix=start_prefix)

        if len(missing_keys) > 0:
            self.logger.info("Weights of {} not initialized from pretrained model: {}".format(
                self.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            self.logger.info("Weights from pretrained model not used in {}: {}".format(
                self.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                self.__class__.__name__, "\n\t".join(error_msgs)))
