from functools import partial
from transformers import MT5Tokenizer
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import os
import json
import torch
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode


def collate(batch):
    query = [item['query'] for item in batch]
    # context = [json.loads(item['rerank']) for item in batch]
    context = []
    for item in batch:
        try:
            context.append(json.loads(item['rerank']))
        except:
            context.append(json.loads(item['passages']))
    label = [item['response'] for item in batch]
    return query, context, label

def collate_inference(batch):
    query = [item['query'] for item in batch]
    # context = [json.loads(item['rerank']) for item in batch]
    context = []
    for item in batch:
        try:
            context.append(json.loads(item['rerank']))
        except:
            context.append(json.loads(item['passages']))
    label = [item['response'] for item in batch]
    scored_pids = [item['scored_pids'] for item in batch]
    return query, context, label,scored_pids

def get_single_turn(query):
    tmp = query.replace('<user>','<agent>')
    text_list = tmp.split('<agent>')
    query_text = text_list[0]
    if len(text_list)>=2:
        query_text = query_text+'<agent>'+text_list[1]
    if len(text_list)>=3:
        query_text = query_text+'<user>'+text_list[2]
    return query_text


def collate_single_turn(batch):
    query = [get_single_turn(item['query']) for item in batch]
    context = [json.loads(item['rerank']) for item in batch]
    label = [item['response'] for item in batch]
    return query, context, label


def remove_duplication_data(dataset):
    response_dict ={}
    removed_dataset = []
    for data in dataset:
        if data['response'] not in response_dict:
            try:
                response_dict[data['response']] = json.dumps(data['rerank'])[0]
            except:
                response_dict[data['response']] = json.dumps(data['passages'])[0]
            removed_dataset.append(data)
    return removed_dataset

def get_translated_dataset(args, fromLang='zh',toLang='fr'):
    translated_dataset = []
    if fromLang=='zh' and toLang == 'fr':
        file_path = 'DAMO_ConvAI/translate_zh2fr.json'
    if fromLang=='zh' and toLang == 'vi':
        file_path = 'DAMO_ConvAI/translate_zh2vi.json'
    if fromLang=='en' and toLang == 'fr':
        file_path = 'DAMO_ConvAI/translate_en2fr.json'
    if fromLang=='en' and toLang == 'vi':
        file_path = 'DAMO_ConvAI/translate_en2vi.json'

    with open(file_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            translated_dataset.append(json.loads(line))
    # translated_dataset = remove_duplication_data(translated_dataset)
    #是否添加prompt
    if args.add_prompt:
        translated_dataset_prompt = []
        for data in translated_dataset:
            if toLang == 'fr':
                data['response'] = "La réponse française qui en résulte est " + data['response']
            else:
                data['response'] = "Câu trả lời tiếng Việt được tạo ra là " + data['response']
            translated_dataset_prompt.append(data)
        translated_dataset = translated_dataset_prompt

    return translated_dataset


def get_train_val_dataset(args):
    fr_train_dataset = MsDataset.load(
        'DAMO_ConvAI/FrDoc2BotGeneration',
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
    vi_train_dataset = MsDataset.load(
        'DAMO_ConvAI/ViDoc2BotGeneration',
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
    # fr_train_dataset = remove_duplication_data(fr_train_dataset)
    # vi_train_dataset = remove_duplication_data(vi_train_dataset)
    train_dataset = [x for dataset in [fr_train_dataset, vi_train_dataset] for x in dataset]

    #是否添加prompt
    if args.add_prompt:
        fr_train_dataset_prompt = []
        vi_train_dataset_prompt = []

        # 法语回答prompt：La réponse française qui en résulte est
        for data in fr_train_dataset:
            data['response'] = "La réponse française qui en résulte est " + data['response']
            fr_train_dataset_prompt.append(data)

        #越南语回答prompt：Câu trả lời tiếng Việt được tạo ra là
        for data in vi_train_dataset:
            data['response'] = "Câu trả lời tiếng Việt được tạo ra là " + data['response']
            vi_train_dataset_prompt.append(data)
        train_dataset_prompt = [x for dataset in [fr_train_dataset_prompt, vi_train_dataset_prompt] for x in dataset]

        fr_train_dataset = fr_train_dataset_prompt
        vi_train_dataset = vi_train_dataset_prompt
        train_dataset = train_dataset_prompt

    if args.valid_only:
        to_train_dataset = [x for i, x in enumerate(fr_train_dataset[:]) if i < 3000] + \
                           [x for i, x in enumerate(vi_train_dataset[:]) if i < 3000]
        to_valid_dataset = [x for i, x in enumerate(fr_train_dataset[:]) if i >= 3000] + \
                           [x for i, x in enumerate(vi_train_dataset[:]) if i >= 3000]
    else:
        to_train_dataset = train_dataset
        to_valid_dataset = train_dataset

    if args.add_pseudo:
        pseudo_dataset = []
        with open(args.pseudo_path,'r',encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                pseudo_dataset.append(json.loads(line))
        to_train_dataset = to_train_dataset + pseudo_dataset

    return to_train_dataset,to_valid_dataset


def create_dataloaders_base(args):
    train_dataset, val_dataset = get_train_val_dataset(args)
    if args.is_mul_gpu:
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                collate_fn=collate,
                                num_workers=16,
                                pin_memory=True,
                                sampler=train_sampler
                                )
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        valid_loader = DataLoader(dataset=val_dataset,
                                  batch_size=args.val_batch_size,
                                  collate_fn=collate,
                                  num_workers=16,
                                  pin_memory=True,
                                  sampler=val_sampler
                                  )
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate)
        valid_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.val_batch_size,
            pin_memory=True,
            collate_fn=collate)

    return train_loader, valid_loader
