# coding:utf-8
import sys
sys.path.append('./retrieval_src')
import os
import json
import faiss
import torch
import numpy as np
from modelscope.msdatasets import MsDataset

from tqdm import tqdm
from torch.utils.data import DataLoader
from modelscope.utils.constant import ModeKeys, DownloadMode

from retrieval_src.config_retrieval import Config
from retrieval_src.trainer import DocumentGroundedDialogRetrievalTrainer
from retrieval_src.retrieval_preprocessor import DocumentGroundedDialogRetrievalPreprocessor

user_args = Config()

with open('DAMO_ConvAI/test.json', encoding='utf-8') as f_in:
    with open('./results/input_test.jsonl', 'w', encoding='utf-8') as f_out:
        for line in f_in.readlines():
            sample = json.loads(line)
            sample['positive'] = ''
            sample['negative'] = ''
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

with open('./results/input_test.jsonl', encoding='utf-8') as f:
    eval_dataset = [json.loads(line) for line in f.readlines()]


all_passages = []
for file_name in ['fr', 'vi']:
    with open(f'all_passages/{file_name}.json', encoding='utf-8') as f:
        all_passages += json.load(f)


model_path = user_args.pretrain_model_dir

preprocessor = DocumentGroundedDialogRetrievalPreprocessor(model_dir=user_args.pretrain_model_dir)


# fr_train_dataset = MsDataset.load(
#     'DAMO_ConvAI/FrDoc2BotRetrieval',
#     download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
#
# vi_train_dataset = MsDataset.load(
#     'DAMO_ConvAI/ViDoc2BotRetrieval',
#     download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
#
# test_eval_dataset = [x for dataset in [fr_train_dataset, vi_train_dataset] for x in dataset]

trainer = DocumentGroundedDialogRetrievalTrainer(
    model=model_path,
    train_dataset=None,
    eval_dataset=eval_dataset,
    all_passages=all_passages
)


def collate(batch):
    query = [item['query'] for item in batch]
    positive = [item['positive'] for item in batch]
    negative = [item['negative'] for item in batch]
    return query, positive, negative


def measure_result(result_dict):
    recall_k = [1, 5, 10, 20]
    meters = {f'R@{k}': [] for k in recall_k}

    for output, target in zip(result_dict['outputs'], result_dict['targets']):
        for k in recall_k:
            if target in output[:k]:
                meters[f'R@{k}'].append(1)
            else:
                meters[f'R@{k}'].append(0)
    for k, v in meters.items():
        meters[k] = sum(v) / len(v)
    return meters


def evaluate(
        trainer,
        return_type='mean_pooling',
        norm=False,
        top_k=20,
        per_gpu_batch_size=32,
        checkpoint_path=None
):
    """
    Evaluate test dataset
    """
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path)
        trainer.model.model.load_state_dict(state_dict)

    valid_loader = DataLoader(
        dataset=trainer.eval_dataset,
        batch_size=per_gpu_batch_size,
        collate_fn=collate
    )
    trainer.model.model.eval()

    valid_iterator = tqdm(valid_loader, total=len(valid_loader),
                          desc='Validation')

    with torch.no_grad():
        all_ctx_vector = []
        for mini_batch in tqdm(
                range(0, len(all_passages), per_gpu_batch_size)
        ):
            context = all_passages[mini_batch: mini_batch + per_gpu_batch_size]

            processed = preprocessor(
                {'context': context},
                invoke_mode=ModeKeys.INFERENCE,
                input_type='context'
            )

            sub_ctx_vector = trainer.model.encode_context(
                processed,
                return_type=return_type,
                norm=norm
            ).detach().cpu().numpy()

            all_ctx_vector.append(sub_ctx_vector)

        all_ctx_vector = np.concatenate(all_ctx_vector, axis=0)
        all_ctx_vector = np.array(all_ctx_vector).astype('float32')
        faiss_index = faiss.IndexFlatIP(all_ctx_vector.shape[-1])
        faiss_index.add(all_ctx_vector)

        results = {'outputs': [], 'targets': []}

        for index, payload in enumerate(valid_iterator):

            query, positive, negative = payload

            processed = preprocessor(
                {'query': query},
                invoke_mode=ModeKeys.INFERENCE
            )

            # mean pooling, cls, pooled output
            query_vector = trainer.model.encode_query(
                processed,
                return_type=return_type,
                norm=norm
            ).detach().cpu().numpy().astype('float32')

            D, Index = faiss_index.search(query_vector, top_k)

            results['outputs'] += [
                [all_passages[x] for x in retrieved_ids] for retrieved_ids in Index.tolist()
            ]
            results['targets'] += positive

        meters = measure_result(results)

    result_path = os.path.join('./results','evaluate_result_retrieval_test_100.json')

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(meters)
    return meters


evaluate(
    trainer=trainer,
    return_type=user_args.return_type,
    norm=user_args.norm,
    top_k=user_args.top_k,
    per_gpu_batch_size=user_args.val_batch_size,
    checkpoint_path=os.path.join('./model_storage/retrieval_storage', 'finetuned_model.bin')
)
