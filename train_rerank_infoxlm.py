# -*- coding: utf-8 -*-

# author:Administrator
# contact: test@test.com
# datetime:2023/3/2 14:49
# software: PyCharm

"""
文件说明：
    
"""
import sys
sys.path.append('./rerank_src')
from modelscope.msdatasets import MsDataset
from rerank_src.utils import DocumentGroundedDialogRerankTrainer,set_seed
from modelscope.utils.constant import DownloadMode
from modelscope.utils.logger import get_logger


logger = get_logger()
def main():
    args = {
        'device': 'gpu',
        'tokenizer_name': '',
        'cache_dir': '',
        'instances_size': 1,
        'output_dir': './model_storage/rerank_output_infoxlm',
        'max_num_seq_pairs_per_device': 32,
        'full_train_batch_size': 32,
        'gradient_accumulation_steps': 32,
        'per_gpu_train_batch_size': 1,
        'num_train_epochs': 10,
        'train_instances': -1,
        'learning_rate': 2e-5,
        'max_seq_length': 512,
        'num_labels': 2,
        'fold': '',  # IofN
        'doc_match_weight': 0.0,
        'query_length': 195,
        'resume_from': '',  # to resume training from a checkpoint
        'config_name': '',
        'do_lower_case': True,
        'weight_decay': 0.01,  # previous default was 0.01
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'warmup_instances': 0.0,  # previous default was 0.1 of total
        'warmup_fraction': 0.1,  # only applies if warmup_instances <= 0
        'no_cuda': False,
        'n_gpu': 1,
        'seed': 42,
        'fp16': False,
        'fp16_opt_level': 'O1',  # previous default was O2
        'per_gpu_eval_batch_size': 8,
        'log_on_all_nodes': False,
        'world_size': 1,
        'global_rank': 0,
        'local_rank': -1,
        'tokenizer_resize': True,
        'model_resize': True,
        'ema':False,
        'adv':True,
        'add_neg':False
    }
    args['gradient_accumulation_steps'] = args['full_train_batch_size'] // (
            args['per_gpu_train_batch_size'] * args['world_size'])

    fr_train_dataset = MsDataset.load(
        'DAMO_ConvAI/FrDoc2BotRerank',
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
    vi_train_dataset = MsDataset.load(
        'DAMO_ConvAI/ViDoc2BotRerank',
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
    train_dataset = [x for dataset in [fr_train_dataset, vi_train_dataset] for x in dataset]

    trainer = DocumentGroundedDialogRerankTrainer(
        model_path='./pretrain_storage/infoxlm-large', dataset=train_dataset[:], args=args)
    trainer.train()


if __name__ == '__main__':
    set_seed(42)
    main()
