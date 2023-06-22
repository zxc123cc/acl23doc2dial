# coding:utf-8


import os
import torch
from typing import Any, Dict
from transformers import XLMRobertaTokenizer

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.utils.type_assert import type_assert
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.preprocessors.builder import PREPROCESSORS

from retrieval_src.config_retrieval import Config


user_args = Config()


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.document_grounded_dialog_retrieval)
class DocumentGroundedDialogRetrievalPreprocessor(Preprocessor):
    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_dir = model_dir
        self.device = user_args.device
        self.query_sequence_length = user_args.query_sequence_length
        self.context_sequence_length = user_args.context_sequence_length
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(os.path.join(self.model_dir))

    @type_assert(object, Dict)
    def __call__(self,
                 data: Dict[str, Any],
                 invoke_mode=ModeKeys.INFERENCE,
                 input_type='query',
                 **preprocessor_param) -> Dict[str, Any]:

        if invoke_mode in (ModeKeys.TRAIN, ModeKeys.EVAL) and \
                invoke_mode != ModeKeys.INFERENCE:

            query, positive, negative = data['query'],\
                                        data['positive'], \
                                        data['negative']

            query_tokenizer_outputs = self.tokenizer.batch_encode_plus(
                query,
                padding=True,
                return_tensors='pt',
                max_length=self.query_sequence_length,
                truncation=True
            )

            positive_context_tokenizer_outputs = self.tokenizer.batch_encode_plus(
                positive,
                padding=True,
                return_tensors='pt',
                max_length=self.context_sequence_length,
                truncation=True
            )

            negative_context_tokenizer_outputs = self.tokenizer.batch_encode_plus(
                negative,
                padding=True,
                return_tensors='pt',
                max_length=self.context_sequence_length,
                truncation=True
            )

            result = {
                'query_input_ids': query_tokenizer_outputs.input_ids,
                'query_attention_mask': query_tokenizer_outputs.attention_mask,
                'context_input_ids': positive_context_tokenizer_outputs.input_ids,
                'context_attention_mask': positive_context_tokenizer_outputs.attention_mask,
                'negative_context_input_ids': negative_context_tokenizer_outputs.input_ids,
                'negative_context_attention_mask': negative_context_tokenizer_outputs.attention_mask,
                'labels': torch.tensor(list(range(len(query))), dtype=torch.long)
            }

        elif input_type == 'query':
            query = data['query']
            query_tokenizer_outputs = self.tokenizer.batch_encode_plus(
                query,
                padding=True,
                return_tensors='pt',
                max_length=self.query_sequence_length,
                truncation=True
            )
            result = {
                'query_input_ids': query_tokenizer_outputs.input_ids,
                'query_attention_mask': query_tokenizer_outputs.attention_mask,
            }

        else:
            context = data['context']
            context_tokenizer_outputs = self.tokenizer.batch_encode_plus(
                context,
                padding=True,
                return_tensors='pt',
                max_length=self.context_sequence_length,
                truncation=True
            )

            result = {
                'context_input_ids': context_tokenizer_outputs.input_ids,
                'context_attention_mask': context_tokenizer_outputs.attention_mask,
            }

        for k, v in result.items():
            result[k] = v.to(self.device)

        return result
