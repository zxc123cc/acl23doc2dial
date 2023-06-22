# coding:utf-8
import sys
sys.path.append('./retrieval_src')
import os
import json
import faiss
import torch
import warnings
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from modelscope.msdatasets import MsDataset
from transformers import AdamW, get_scheduler
from modelscope.utils.logger import get_logger
from modelscope.utils.constant import DownloadMode, ModeKeys
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from retrieval_src.tricks.opt import Lookahead
from retrieval_src.config_retrieval import Config
from retrieval_src.retrieval_preprocessor import DocumentGroundedDialogRetrievalPreprocessor
from retrieval_src.trainer import DocumentGroundedDialogRetrievalTrainer
from retrieval_src.tricks.adv import FGM
from retrieval_src.tricks.ema import EMA

user_args = Config()
logger = get_logger()
warnings.filterwarnings('ignore')

import os
import random
import numpy as np
import torch


def seed_everything(seed=None):
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if (seed is None) or not (min_seed_value <= seed <= max_seed_value):
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    # print(f"Global seed set to {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


seed_everything(user_args.seed)


def collate(batch):
    query = [item['query'] for item in batch]
    positive = [item['positive'] for item in batch]
    negative = [item['negative'] for item in batch]
    return query, positive, negative


def prepare_optimizer(model, lr, weight_decay, eps):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
            weight_decay,
    }, {
        'params': [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay':
            0.0,
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
    # optimizer = Lookahead(optimizer, 0.5, 5)
    return optimizer


def prepare_scheduler(optimizer, epochs, steps_per_epoch, warmup_rate):
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_rate)
    scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)

    # scheduler = CosineAnnealingWarmRestarts(optimizer, total_steps // user_args.total_epoches * 1, 1, eta_min=5e-6, last_epoch=-1)
    return scheduler


def train(
        trainer,
        return_type='mean_pooling',
        norm=False,
        total_epoches=20,
        batch_size=128,
        per_gpu_batch_size=32,
        accumulation_steps=1,
        clip_grad_norm=1.0,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.1,
        eps=1e-06,
        loss_log_freq=100,
        ema=False,
        adv=False,
        adv_eps=1.0
):
    """
    Fine-tuning trainsets
    """

    # obtain train loader
    train_loader = DataLoader(
        dataset=trainer.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0
    )

    optimizer = prepare_optimizer(trainer.model.model,
                                  learning_rate,
                                  weight_decay, eps)
    steps_per_epoch = len(train_loader) // accumulation_steps
    scheduler = prepare_scheduler(optimizer, total_epoches,
                                  steps_per_epoch, warmup_ratio)

    if ema:
        ema = EMA(trainer.model.parameters(), decay=0.999)

    if adv:
        fgm = FGM(trainer.model, eps=adv_eps)


    """
    saving pre and aft batch
    """
    train_iterator = tqdm(train_loader, total=len(train_loader),
                          desc=f'Preparing pre and aft batch')
    pre_inputs, aft_inputs = [], []
    all_inputs = []
    for index, payload in enumerate(train_iterator):
        all_inputs.append(payload)

    for i in range(len(all_inputs)):
        if i == 0:
            pre_inputs.append(None)
        elif i == len(all_inputs) - 1:
            aft_inputs.append(None)
        else:
            pre_query, pre_positive, pre_negative = all_inputs[i - 1]
            aft_query, aft_positive, aft_negative = all_inputs[i + 1]
            pre_input = preprocessor(
                {
                    'query': pre_query,
                    'positive': pre_query,
                    'negative': pre_negative
                },
                invoke_mode=ModeKeys.TRAIN
            )
            aft_input = preprocessor(
                {
                    'query': aft_query,
                    'positive': aft_query,
                    'negative': aft_negative
                },
                invoke_mode=ModeKeys.TRAIN
            )
            pre_inputs.append(pre_input)
            aft_inputs.append(aft_input)

    pre_inputs.append(None)
    aft_inputs.append(None)
    aft_inputs = [None for i in range(len(pre_inputs))]

    global_step = 0
    best_score = 0.0
    for epoch in range(total_epoches):

        trainer.model.model.train()

        losses = []

        train_iterator = tqdm(train_loader, total=len(train_loader),
                              desc=f'Training epoch : {epoch + 1}')

        for index, payload in enumerate(train_iterator):

            global_step += 1

            if user_args.debug and global_step == 50:
                _ = evaluate(trainer, per_gpu_batch_size=per_gpu_batch_size)

            query, positive, negative = payload
            print('query: ',len(query))
            print('positive: ',len(positive))
            print('negative: ',len(negative))
            processed = preprocessor(
                {
                    'query': query,
                    'positive': positive,
                    'negative': negative
                },
                invoke_mode=ModeKeys.TRAIN
            )

            loss, logits = trainer.model(
                input=processed,
                pre_input=pre_inputs[index],
                aft_input=aft_inputs[index],
                norm=norm,
                return_type=return_type,
                training=True
            )

            if accumulation_steps > 1:
                loss = loss / accumulation_steps

            loss.backward()

            if adv:
                fgm.attack()
                adv_loss, _ = trainer.model(
                    processed,
                    norm=norm,
                    return_type=return_type
                )

                if accumulation_steps > 1:
                    adv_loss = adv_loss / accumulation_steps

                adv_loss.backward()
                fgm.restore()

            train_iterator.set_postfix(loss=loss.item(), global_step=global_step)

            if (index + 1) % accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), clip_grad_norm)

                optimizer.step()

                if ema:
                    ema.update(trainer.model.parameters())

                scheduler.step()
                optimizer.zero_grad()
            losses.append(loss.item())

            if (index + 1) % loss_log_freq == 0:
                logger.info(
                    f'\n>>> batch: {batch_size * index} \t loss: {sum(losses) / len(losses)}'
                )
                losses = []

        if losses:
            logger.info(
                f'\nEpoch: {epoch + 1} \t batch: last \t loss: {sum(losses) / len(losses)}'
            )

        if ema:
            ema.store(trainer.model.parameters())
            ema.copy_to(trainer.model.parameters())

        meters = evaluate(trainer, per_gpu_batch_size=per_gpu_batch_size, top_k=user_args.top_k)
        # total_score = sum([x for x in meters.values()])
        # total_score = meters[f'R@{user_args.topk}']
        # if total_score >= best_score:
        #     best_score = total_score
        #     model_path = os.path.join(trainer.model.model_dir,
        #                               'finetuned_model.bin')
        #     state_dict = trainer.model.model.state_dict()
        #     torch.save(state_dict, model_path)

        model_path = os.path.join('model_storage/retrieval_storage', 'finetuned_model.bin')
        state_dict = trainer.model.model.state_dict()
        torch.save(state_dict, model_path)

        if ema:
            ema.restore(trainer.model.parameters())


def measure_result(result_dict):
    recall_k = [1, 10, 20, 30, 40]
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
        collate_fn=collate,
        num_workers=16
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

    result_path = os.path.join(trainer.model.model_dir,
                               'evaluate_result.json')

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    trainer.model.model.train()

    logger.info(meters)

    return meters


fr_train_dataset = MsDataset.load(
    'DAMO_ConvAI/FrDoc2BotRetrieval',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

vi_train_dataset = MsDataset.load(
    'DAMO_ConvAI/ViDoc2BotRetrieval',
    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

train_dataset = [x for dataset in [fr_train_dataset, vi_train_dataset] for x in dataset]

to_train_dataset = [x for i, x in enumerate(fr_train_dataset) if i < 3000] + \
                   [x for i, x in enumerate(vi_train_dataset) if i < 3000]
to_valid_dataset = [x for i, x in enumerate(fr_train_dataset) if i >= 3000] + \
                   [x for i, x in enumerate(vi_train_dataset) if i >= 3000]

all_passages = []
for file_name in ['fr', 'vi']:
    with open(f'all_passages/{file_name}.json', encoding='utf-8') as f:
        all_passages += json.load(f)

model_path = user_args.pretrain_model_dir

preprocessor = DocumentGroundedDialogRetrievalPreprocessor(model_dir=user_args.pretrain_model_dir)


if user_args.valid_only:
    print(f'Total spilt train samples is : {len(to_train_dataset)}, '
          f'total split valid samples is : {len(to_valid_dataset)}  !!!')

    trainer = DocumentGroundedDialogRetrievalTrainer(
        model=model_path,
        train_dataset=to_train_dataset,
        eval_dataset=to_valid_dataset,
        all_passages=all_passages)
else:
    print(f'>>> Total num of samples is : {len(train_dataset)} !!!')

    trainer = DocumentGroundedDialogRetrievalTrainer(
        model=model_path,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        all_passages=all_passages)


train(
    trainer=trainer,
    norm=user_args.norm,
    batch_size=user_args.batch_size,
    per_gpu_batch_size=user_args.val_batch_size,
    total_epoches=user_args.total_epoches,
    weight_decay=user_args.weight_decay,
    warmup_ratio=user_args.warmup_ratio,
    learning_rate=user_args.lr,
    eps=user_args.eps,
    accumulation_steps=user_args.accumulation_steps,
    clip_grad_norm=user_args.clip_grad_norm,
    loss_log_freq=user_args.log_freq,
    ema=user_args.ema,
    adv=user_args.adv,
    adv_eps=user_args.adv_eps
)


evaluate(
    trainer,
    return_type=user_args.return_type,
    norm=user_args.norm,
    top_k=user_args.top_k,
    per_gpu_batch_size=user_args.val_batch_size,
    checkpoint_path=os.path.join('model_storage/retrieval_storage', 'finetuned_model.bin')
)
