# coding:utf-8


import json
from modelscope.trainers import EpochBasedTrainer
from modelscope.utils.logger import get_logger

from gen_src.gen_model import DocumentGroundedDialogGenerateModel
from gen_src.gen_processor import DocumentGroundedDialogGeneratePreprocessor

logger = get_logger()


def collate(batch):
    query = [item['query'] for item in batch]
    context = [json.loads(item['rerank']) for item in batch]
    label = [item['response'] for item in batch]
    return query, context, label


class DocumentGroundedDialogGenerateTrainer(EpochBasedTrainer):
    def __init__(self, model, **kwargs):
        self.model = DocumentGroundedDialogGenerateModel(model)
        self.preprocessor = DocumentGroundedDialogGeneratePreprocessor(model_dir=self.model.model_dir)
        self.device = self.preprocessor.device
        self.model.model.to(self.device)
        self.train_dataset = kwargs['train_dataset']
        self.eval_dataset = kwargs['eval_dataset']
