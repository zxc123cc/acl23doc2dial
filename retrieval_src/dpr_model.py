# coding:utf-8


import os

import torch

import torch.nn as nn

from transformers import XLMRobertaModel, AutoConfig
from torch.utils.checkpoint import checkpoint


class Wrapper(nn.Module):
    def __init__(self, encoder):
        super(Wrapper, self).__init__()
        self.encoder = encoder

    def forward(
            self,
            input_ids,
            attention_mask,
            dummy_tensor=None,
            return_type='mean_pooling',
            norm=False
    ):
        output = self.encoder(input_ids, attention_mask)

        if return_type == 'mean_pooling':
            output_vector = self.mean_pooling(output.last_hidden_state, attention_mask)
        elif return_type == 'max_pooling':
            output_vector = self.max_pooling(output.last_hidden_state, attention_mask)
        elif return_type == 'cls':
            output_vector = output.last_hidden_state[:, 0, :]
        else:
            output_vector = output.pooler_output

        if norm:
            output_vector = nn.functional.normalize(output_vector, dim=1)

        return output_vector

    @staticmethod
    def mean_pooling(sequence_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
        return output_vector

    @staticmethod
    def max_pooling(sequence_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).long()
        sequence_output[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(sequence_output, 1)[0]
        return output_vector


class DPRModel(nn.Module):
    def __init__(self, model_dir, t=0.05, finetune_t=True):
        super().__init__()

        qry_encoder = XLMRobertaModel.from_pretrained(
            config=AutoConfig.from_pretrained(model_dir),
            pretrained_model_name_or_path=model_dir
        )
        self.qry_encoder = Wrapper(qry_encoder)

        ctx_encoder = XLMRobertaModel.from_pretrained(
            config=AutoConfig.from_pretrained(model_dir),
            pretrained_model_name_or_path=model_dir
        )
        self.ctx_encoder = Wrapper(ctx_encoder)

        # temperature
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / t).log(), requires_grad=finetune_t)

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(
            self,
            query_input_ids,
            query_attention_mask,
            context_input_ids,
            context_attention_mask,
            labels,
            negative_context_input_ids=None,
            negative_context_attention_mask=None,
            gck_segment=32,
            return_type='mean_pooling',
            norm=True,
            training=False,
            pre_negative_context_input_ids1=None,
            pre_negative_context_attention_mask1=None,
            pre_negative_context_input_ids2=None,
            pre_negative_context_attention_mask2=None,
            aft_negative_context_input_ids1=None,
            aft_negative_context_attention_mask1=None,
            aft_negative_context_input_ids2=None,
            aft_negative_context_attention_mask2=None
    ):

        query_vector = self.encode(
            self.qry_encoder,
            query_input_ids,
            query_attention_mask,
            gck_segment,
            return_type=return_type,
            norm=norm
        )

        context_vector = self.encode(
            self.ctx_encoder,
            context_input_ids,
            context_attention_mask,
            gck_segment,
            return_type=return_type,
            norm=norm
        )

        if negative_context_input_ids is not None:

            negative_context_vector = self.encode(
                self.ctx_encoder,
                negative_context_input_ids,
                negative_context_attention_mask,
                gck_segment,
                return_type=return_type,
                norm=norm
            )

            context_vector = torch.cat([context_vector, negative_context_vector], dim=0)

        logits = torch.matmul(query_vector, context_vector.T)

        if training:
            add_margin = 1.0
            logits -= torch.zeros(logits.size()).fill_diagonal_(add_margin).to(logits.device)

        # add temperature
        logits *= self.log_inv_t.exp()

        """
        ADD pre and after neg
        """
        if pre_negative_context_input_ids1 is not None:
            pre_negative_context_vector1 = self.encode(
                self.ctx_encoder,
                pre_negative_context_input_ids1,
                pre_negative_context_attention_mask1,
                gck_segment,
                return_type=return_type,
                norm=norm
            )

            pre_negative_context_vector2 = self.encode(
                self.ctx_encoder,
                pre_negative_context_input_ids2,
                pre_negative_context_attention_mask2,
                gck_segment,
                return_type=return_type,
                norm=norm
            )

            neg_pre_context_vectors = torch.cat([pre_negative_context_vector1, pre_negative_context_vector2], dim=0)

            neg_logits = torch.matmul(query_vector, neg_pre_context_vectors.T)
            neg_logits *= self.log_inv_t.exp()
            neg_logits *= 0.5

            logits = torch.cat([logits, neg_logits], dim=-1)

        if aft_negative_context_input_ids1 is not None:
            aft_negative_context_vector1 = self.encode(
                self.ctx_encoder,
                aft_negative_context_input_ids1,
                aft_negative_context_attention_mask1,
                gck_segment,
                return_type=return_type,
                norm=norm
            )

            aft_negative_context_vector2 = self.encode(
                self.ctx_encoder,
                aft_negative_context_input_ids2,
                aft_negative_context_attention_mask2,
                gck_segment,
                return_type=return_type,
                norm=norm
            )

            neg_aft_context_vectors = torch.cat([aft_negative_context_vector1, aft_negative_context_vector2], dim=0)

            neg_logits = torch.matmul(query_vector, neg_aft_context_vectors.T)
            neg_logits *= self.log_inv_t.exp()
            neg_logits *= 0.5

            logits = torch.cat([logits, neg_logits], dim=-1)

        loss = self.loss_fct(logits, labels)
        return loss, logits

    def encode(
            self,
            context_model,
            input_ids,
            attention_mask,
            gck_segment=32,
            norm=False,
            return_type='mean_pooling'
    ):

        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        encoder_outputs = []
        for mini_batch in range(0, input_ids.shape[0], gck_segment):
            mini_batch_input_ids = input_ids[mini_batch:mini_batch + gck_segment]
            mini_batch_attention_mask = attention_mask[mini_batch:mini_batch + gck_segment]
            mini_batch_output = checkpoint(
                context_model, mini_batch_input_ids,
                mini_batch_attention_mask,
                dummy_tensor,
                return_type,
                norm
            )
            encoder_outputs.append(mini_batch_output)

        output_vector = torch.cat(encoder_outputs, dim=0)

        return output_vector

