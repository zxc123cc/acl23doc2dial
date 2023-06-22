# coding:utf-8


import os
from typing import Dict
from modelscope.models.base import Tensor, TorchModel
from retrieval_src.dpr_model import DPRModel


class DocumentGroundedDialogRetrievalModel(TorchModel):
    def __init__(self, model_dir, t=0.05, finetune_t=True):
        super().__init__(model_dir)
        self.model = DPRModel(model_dir, t, finetune_t)

    def encode_query(
            self,
            input: Dict[str, Tensor],
            return_type='mean_pooling',
            norm=False
    ):

        query_input_ids = input['query_input_ids']
        query_attention_mask = input['query_attention_mask']

        query_vector = self.model.qry_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            return_type=return_type,
            norm=norm
        )

        return query_vector

    def encode_context(
            self,
            input: Dict[str, Tensor],
            return_type='mean_pooling',
            norm=False
    ):

        context_input_ids = input['context_input_ids']
        context_attention_mask = input['context_attention_mask']

        context_vector = self.model.ctx_encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            return_type=return_type,
            norm=norm
        )

        return context_vector

    def forward(
            self,
            input: Dict[str, Tensor],
            pre_input: Dict[str, Tensor]=None,
            aft_input: Dict[str, Tensor]=None,
            gck_segment=32,
            norm=True,
            return_type='mean_pooling',
            training=False
    ):

        query_input_ids = input['query_input_ids']
        query_attention_mask = input['query_attention_mask']
        context_input_ids = input['context_input_ids']
        context_attention_mask = input['context_attention_mask']
        labels = input['labels']

        if 'negative_context_input_ids' in input.keys():
            negative_context_input_ids = input['negative_context_input_ids']
            negative_context_attention_mask = input['negative_context_attention_mask']
        else:
            negative_context_input_ids = None,
            negative_context_attention_mask = None

        """
        pre and after batch for neg cons
        """
        pre_negative_context_input_ids1, pre_negative_context_attention_mask1 = None, None
        pre_negative_context_input_ids2, pre_negative_context_attention_mask2 = None, None
        if pre_input is not None:
            pre_negative_context_input_ids1, pre_negative_context_attention_mask1 = pre_input['context_input_ids'], \
                                                                                    pre_input['context_attention_mask']
            if 'negative_context_input_ids' in pre_input.keys():
                pre_negative_context_input_ids2, pre_negative_context_attention_mask2 = pre_input['negative_context_input_ids'], \
                                                                                        pre_input['negative_context_attention_mask']

        aft_negative_context_input_ids1, aft_negative_context_attention_mask1 = None, None
        aft_negative_context_input_ids2, aft_negative_context_attention_mask2 = None, None
        if aft_input is not None:
            aft_negative_context_input_ids1, aft_negative_context_attention_mask1 = aft_input['context_input_ids'], \
                                                                                    aft_input['context_attention_mask']
            if 'negative_context_input_ids' in aft_input.keys():
                aft_negative_context_input_ids2, aft_negative_context_attention_mask2 = aft_input['negative_context_input_ids'], \
                                                                                        aft_input['negative_context_attention_mask']

        outputs = self.model(
            query_input_ids, query_attention_mask,
            context_input_ids, context_attention_mask, labels,
            gck_segment=gck_segment, norm=norm, return_type=return_type,
            negative_context_input_ids=negative_context_input_ids,
            negative_context_attention_mask=negative_context_attention_mask,
            training=training,
            pre_negative_context_input_ids1=pre_negative_context_input_ids1,
            pre_negative_context_attention_mask1=pre_negative_context_attention_mask1,
            pre_negative_context_input_ids2=pre_negative_context_input_ids2,
            pre_negative_context_attention_mask2=pre_negative_context_attention_mask2,
            aft_negative_context_input_ids1=aft_negative_context_input_ids1,
            aft_negative_context_attention_mask1=aft_negative_context_attention_mask1,
            aft_negative_context_input_ids2=aft_negative_context_input_ids2,
            aft_negative_context_attention_mask2=aft_negative_context_attention_mask2
        )

        return outputs
