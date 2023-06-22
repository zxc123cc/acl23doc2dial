# coding:utf-8

from modelscope.metainfo import Trainers
from modelscope.utils.logger import get_logger
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers import EpochBasedTrainer

from retrieval_src.retrieval_model import DocumentGroundedDialogRetrievalModel
from retrieval_src.retrieval_preprocessor import DocumentGroundedDialogRetrievalPreprocessor

logger = get_logger()


@TRAINERS.register_module(
    module_name=Trainers.document_grounded_dialog_retrieval_trainer)
class DocumentGroundedDialogRetrievalTrainer(EpochBasedTrainer):
    def __init__(self, model, **kwargs):

        self.model = DocumentGroundedDialogRetrievalModel(model_dir=model)
        self.preprocessor = DocumentGroundedDialogRetrievalPreprocessor(
            model_dir=self.model.model_dir)
        self.device = self.preprocessor.device
        self.model.model.to(self.device)
        self.train_dataset = kwargs['train_dataset']
        self.eval_dataset = kwargs['eval_dataset']
        self.all_passages = kwargs['all_passages']
