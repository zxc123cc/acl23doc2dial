# coding:utf-8


import os

import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaForSequenceClassification, MT5ForConditionalGeneration, AutoConfig, \
    RagTokenForGeneration, DPRQuestionEncoder, DPRConfig


class Rerank(nn.Module):

    def __init__(self, encoder, top_k):
        super().__init__()
        self.encoder = encoder
        self.top_k = top_k

    def forward(self, inputs):
        model = self.encoder
        logits = F.log_softmax(model(**inputs)[0], dim=-1)[:, 1]
        logits = logits.view(-1, self.top_k)
        logprobs = F.log_softmax(logits, dim=-1)
        return logprobs


class Re2GModel(nn.Module):

    def __init__(self, model_dir, config):
        super(Re2GModel, self).__init__()
        self.config = config
        self.top_k = self.config['top_k']

        encoder = XLMRobertaForSequenceClassification(
            config=AutoConfig.from_pretrained(
                os.path.join(model_dir, 'rerank')))
        generator = MT5ForConditionalGeneration(
            config=AutoConfig.from_pretrained(
                os.path.join(model_dir, 'generation')))

        self.rerank = Rerank(encoder, self.top_k)

        dpr_config = DPRConfig()
        dpr_config.vocab_size = encoder.config.vocab_size
        rag_model = RagTokenForGeneration(
            question_encoder=DPRQuestionEncoder(dpr_config),
            generator=generator)
        rag_model.rag.question_encoder = None
        self.generator = rag_model

    def forward(self, rerank_input_ids, input_ids, attention_mask, label_ids):
        doc_scores = self.rerank(rerank_input_ids)

        outputs = self.generator(
            labels=label_ids,
            context_input_ids=input_ids,
            context_attention_mask=attention_mask,
            doc_scores=doc_scores,
            n_docs=self.top_k)
        return outputs

    def generate(self, rerank_input_ids, input_ids, attention_mask):
        doc_scores = self.rerank(rerank_input_ids)

        beam_search_output = self.generator.generate(
            n_docs=self.top_k,
            encoder_input_ids=input_ids,
            context_input_ids=input_ids,
            context_attention_mask=attention_mask,
            doc_scores=doc_scores,
            num_beams=self.config['num_beams'],
            max_length=self.config['target_sequence_length'],
            early_stopping=True,
            no_repeat_ngram_size=self.config['no_repeat_ngram_size'],
            return_dict_in_generate=True,
            output_scores=True
        )
        generated_ids = beam_search_output.detach().cpu().numpy()

        return generated_ids

