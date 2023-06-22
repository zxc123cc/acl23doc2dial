# coding:utf-8

import os
import torch
from typing import Dict
from modelscope.models.base import Tensor, TorchModel
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile

from gen_src.re2gmodel import Re2GModel


class DocumentGroundedDialogGenerateModel(TorchModel):
    _backbone_prefix = ''

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)

        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION)
        )
        self.model = Re2GModel(model_dir, self.config)
        state_dict = torch.load(
            os.path.join(self.model_dir, 're2g', ModelFile.TORCH_MODEL_BIN_FILE),
            map_location='cpu'
        )
        self.model.load_state_dict(state_dict)

    def forward(self, input: Dict[str, Tensor]):
        rerank_input_ids = input['rerank_input_ids']
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        label_ids = input['label_ids']

        outputs = self.model(rerank_input_ids, input_ids, attention_mask,
                             label_ids)
        return outputs

    def generate(self, input: Dict[str, Tensor]):
        rerank_input_ids = input['rerank_input_ids']
        input_ids = input['input_ids']
        attention_mask = input['attention_mask']
        outputs = self.model.generate(rerank_input_ids, input_ids,
                                      attention_mask)
        return outputs