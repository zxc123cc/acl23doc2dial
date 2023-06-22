# coding:utf-8
import sys
sys.path.append('./gen_src')
import os
import re
import json
import tqdm
import torch
import string
import warnings
import sacrebleu
import transformers

from rouge import Rouge
from collections import Counter

from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
import random
import numpy as np
from modelscope.msdatasets import MsDataset
from modelscope.utils.logger import get_logger
from modelscope.utils.constant import DownloadMode

from gen_src.tricks.ema import EMA
from gen_src.tricks.adv import FGM, PGD,AWP
from gen_src.tricks.opt import Lookahead
from gen_src.gen_trainer import DocumentGroundedDialogGenerateTrainer

from gen_src.config_infer import Config

from gen_src.data_helpter_gen import collate,collate_single_turn,get_translated_dataset,get_train_val_dataset


user_args = Config()
logger = get_logger()
transformers.logging.set_verbosity_error()

warnings.filterwarnings('ignore')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def prepare_optimizer(
        model,
        encoder_lr, decoder_lr, other_lr, opt_lr,
        weight_decay, eps, lk=False
):
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(model.named_parameters())

    encoder_param_optimizer = []
    decoder_param_optimizer = []
    other_param_optimizer = []

    for name, param in model_param:
        if 'encoder' in str(name):
            encoder_param_optimizer.append((name, param))
        elif 'decoder' in str(name):
            decoder_param_optimizer.append((name, param))
        else:
            other_param_optimizer.append((name, param))

    optimizer_grouped_parameters = [
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay, 'lr': other_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': other_lr},

        {"params": [p for n, p in encoder_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay, 'lr': encoder_lr},
        {"params": [p for n, p in encoder_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': encoder_lr},

        {"params": [p for n, p in decoder_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay, 'lr': decoder_lr},
        {"params": [p for n, p in decoder_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': decoder_lr}

    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opt_lr, eps=eps)

    if lk:
        optimizer = Lookahead(optimizer, 5, 1)

    return optimizer


def prepare_scheduler(optimizer, epochs, steps_per_epoch, warmup_rate):
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_rate)
    scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    return scheduler


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def matching_evaluate(references, predictions):
    f1 = em = total = 0
    for ref_text, prediction in zip(references, predictions):
        total += 1
        ground_truths = [ref_text]
        f1 += metric_max_over_ground_truths(f1_score, prediction,
                                            ground_truths)
        em += metric_max_over_ground_truths(exact_match_score, prediction,
                                            ground_truths)
    f1 = 100.0 * f1 / total
    em = 100.0 * em / total

    return f1, em


def measure_result(result_dict):
    meters = dict()

    hypothesis_list = [
        x.replace('<extra_id_0>', '') for x in result_dict['outputs']
    ]
    hypothesis_list = [x if len(x) > 10 else 'placeholder' for x in hypothesis_list]
    if user_args.add_prompt:
        reference_list = [
            x.replace('<extra_id_0> ', '').split('<response>')[1].strip() for x in result_dict['targets']
        ]
    else:
        reference_list = [
            x.replace('<response>', '') for x in result_dict['targets']
        ]
    instance_num = len(reference_list)

    # F1
    f1, em = matching_evaluate(reference_list, hypothesis_list)
    meters['f1'] = f1

    # SacreBleu
    bleu_score = [
        sacrebleu.sentence_bleu(hypothesis, [reference]).score
        for hypothesis, reference in zip(hypothesis_list, reference_list)
    ]
    bleu_score = sum(bleu_score) / instance_num
    meters['bleu'] = bleu_score

    # Rouge-L
    rouge_func = Rouge()
    rouge_score = [
        x['rouge-l']['f']
        for x in rouge_func.get_scores(hypothesis_list, reference_list)
    ]
    rouge_score = (sum(rouge_score) / instance_num) * 100
    meters['rouge'] = rouge_score

    return meters


def train(
        trainer,
        total_epoches=10,
        batch_size=16,
        accumulation_steps=1,
        encoder_lr=2e-5,
        decoder_lr=1e-4,
        other_lr=2e-5,
        opt_lr=4e-5,
        warmup_ratio=0.1,
        weight_decay=0.1,
        eps=1e-06,
        loss_log_freq=40,
        clip_grad_norm=1.0,
        ema=True,
        adv=True
):
    model = trainer.model.model.generator.generator
    if user_args.warmup_checkpoint_path is not None:
        state_dict = torch.load(user_args.warmup_checkpoint_path)
        trainer.model.model.load_state_dict(state_dict)
    if ema:
        ema = EMA(model.parameters(), decay=0.999)

    if adv:
        fgm = FGM(model)
        awp = AWP(model, adv_lr=0.001, adv_eps=0.0001)

    tokenizer = trainer.preprocessor.generation_tokenizer
    device = trainer.preprocessor.device

    train_loader = DataLoader(
        dataset=trainer.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate)

    optimizer = prepare_optimizer(trainer.model.model,
                                  encoder_lr, decoder_lr, other_lr, opt_lr,
                                  weight_decay, eps, False)
    steps_per_epoch = len(train_loader) // accumulation_steps

    """
    BUILD SCHEDULER
    """
    # t_total = len(train_loader) * total_epoches
    # scheduler = CosineAnnealingWarmRestarts(optimizer, t_total // total_epoches * 1,
    #                                         1, eta_min=5e-6, last_epoch=-1)

    scheduler = prepare_scheduler(optimizer, total_epoches,
                                  steps_per_epoch, warmup_ratio)

    best_score = 0.0
    global_step = 0

    for epoch in range(total_epoches):
        trainer.model.model.train()

        losses = []

        train_iterator = tqdm.tqdm(train_loader, total=len(train_loader),
                                   desc=f'Training epoch : {epoch + 1}')

        for index, payload in enumerate(train_iterator):

            global_step += 1

            query, context, label = payload
            query = [
                tokenizer.decode(
                    tokenizer([x], add_special_tokens=False, return_tensors='pt')['input_ids'][0][:user_args.query_max_length])
                for x in query
            ]

            generator_inputs = [
                ' '.join([query[i], '<passage>', context[i][0]])
                for i in range(len(query))
            ]

            input_ids = tokenizer.batch_encode_plus(
                list(generator_inputs), padding=True, return_tensors='pt').input_ids.to(device)
            label_ids = tokenizer.batch_encode_plus(
                list(label), padding=True, return_tensors='pt').input_ids.to(device)

            loss = model(input_ids=input_ids, labels=label_ids)[0]

            if accumulation_steps > 1:
                loss = loss / accumulation_steps


            loss.backward()

            if adv and epoch >= user_args.awp_start:
                awp._save()
                awp._attack_step()
                adv_loss = model(input_ids=input_ids, labels=label_ids)[0]
                optimizer.zero_grad()
                adv_loss.backward()
                awp._restore()

            if adv:
                fgm.attack()
                adv_loss = model(input_ids=input_ids, labels=label_ids)[0]
                if accumulation_steps > 1:
                    adv_loss = adv_loss / accumulation_steps
                adv_loss.backward()
                fgm.restore()

            if (index + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()

                if ema:
                    ema.update(model.parameters())

                scheduler.step()
                optimizer.zero_grad()

            train_iterator.set_postfix(loss=loss.item(), global_step=global_step)

        if ema:
            ema.store(model.parameters())
            ema.copy_to(model.parameters())

        if user_args.valid_only:
            meters = evaluate(trainer, batch_size=batch_size)
            total_score = sum([x for x in meters.values()])
            logger.info('epoch %d score: %.4f' %(epoch, total_score))
            if total_score >= best_score:
                best_score = total_score
                model_path = os.path.join(user_args.model_storage_dir,f'finetuned_model_best.bin')
                state_dict = trainer.model.model.state_dict()
                torch.save(state_dict, model_path)
                logger.info('saving model to %s' %(model_path))

        if epoch == user_args.save_epoch:
            model_path = os.path.join(user_args.model_storage_dir,f'finetuned_model_epoch{epoch}.bin')
            state_dict = trainer.model.model.state_dict()
            torch.save(state_dict, model_path)
            logger.info('saving model to %s' %(model_path))
            break

        if ema:
            ema.restore(model.parameters())


def evaluate(
        trainer,
        batch_size=16,
        checkpoint_path=None
):
    model = trainer.model.model.generator.generator
    tokenizer = trainer.preprocessor.generation_tokenizer
    device = trainer.preprocessor.device

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path)
        trainer.model.model.load_state_dict(state_dict)

    valid_loader = DataLoader(
        dataset=trainer.eval_dataset,
        batch_size=batch_size,
        collate_fn=collate)

    valid_iterator = tqdm.tqdm(valid_loader, total=len(valid_loader), desc='Evaluation')

    trainer.model.model.eval()
    with torch.no_grad():
        results = {'outputs': [], 'targets': []}
        for index, payload in enumerate(valid_iterator):
            query, context, label = payload
            query = [
                tokenizer.decode(
                    tokenizer([x], add_special_tokens=False,
                              return_tensors='pt')['input_ids'][0][:user_args.query_max_length]
                )
                for x in query
            ]

            generator_inputs = [
                ' '.join([query[i], '<passage>', context[i][0]])
                for i in range(len(query))
            ]

            input_ids = tokenizer.batch_encode_plus(
                list(generator_inputs), padding=True, return_tensors='pt').input_ids.to(device)

            outputs = model.generate(input_ids, num_beams=user_args.num_beams,
                                     max_length=user_args.max_length, early_stopping=True,
                                     no_repeat_ngram_size=user_args.no_repeat_ngram_size)

            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)

            label = trainer.preprocessor.generation_tokenizer.batch_decode(
                trainer.preprocessor.generation_tokenizer.batch_encode_plus(
                    label, add_special_tokens=False).input_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)

            results['outputs'] += predictions
            results['targets'] += label

        meters = measure_result(results)
        result_path = os.path.join(trainer.model.model_dir,
                                   'evaluate_result.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    trainer.model.model.train()

    logger.info(meters)
    return meters

def warmup_model():
    zh2fr_dataset = get_translated_dataset(user_args,fromLang='zh',toLang='fr')
    zh2vi_dataset = get_translated_dataset(user_args,fromLang='zh',toLang='vi')
    en2fr_dataset = get_translated_dataset(user_args,fromLang='en',toLang='fr')
    en2vi_dataset = get_translated_dataset(user_args,fromLang='en',toLang='vi')

    train_dataset = [x for dataset in [zh2fr_dataset, zh2vi_dataset,en2fr_dataset,en2vi_dataset] for x in dataset]
    print(f'Total spilt train samples is : {len(train_dataset)}, '
          f'total split valid samples is : {len(train_dataset)}  !!!')
    trainer = DocumentGroundedDialogGenerateTrainer(
        model=user_args.pretrain_model_dir,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
    )
    user_args.model_storage_dir = './model_storage/translate_lang_warmup'
    user_args.total_epoches = 5
    user_args.adv = False
    user_args.save_epoch = 0
    if not os.path.exists(user_args.model_storage_dir):
        os.makedirs(user_args.model_storage_dir, exist_ok=True)
    train(
        trainer,
        batch_size=user_args.batch_size,
        accumulation_steps=user_args.accumulation_steps,
        total_epoches=user_args.total_epoches,
        encoder_lr=user_args.encoder_lr,
        decoder_lr=user_args.decoder_lr,
        other_lr=user_args.other_lr,
        opt_lr=user_args.opt_lr,
        warmup_ratio=user_args.warmup_ratio,
        weight_decay=user_args.weight_decay,
        eps=user_args.eps,
        loss_log_freq=user_args.loss_log_freq,
        ema=user_args.ema,
        adv=user_args.adv
    )

if __name__ == '__main__':
    setup_seed(user_args.seed)
    # warmup_model()
    to_train_dataset,to_valid_dataset = get_train_val_dataset(user_args)
    print(f'Total spilt train samples is : {len(to_train_dataset)}, '
          f'total split valid samples is : {len(to_valid_dataset)}  !!!')
    trainer = DocumentGroundedDialogGenerateTrainer(
        model=user_args.pretrain_model_dir,
        train_dataset=to_train_dataset,
        eval_dataset=to_valid_dataset,
    )

    if not os.path.exists(user_args.model_storage_dir):
        os.makedirs(user_args.model_storage_dir, exist_ok=True)
    train(
        trainer,
        batch_size=user_args.batch_size,
        accumulation_steps=user_args.accumulation_steps,
        total_epoches=user_args.total_epoches,
        encoder_lr=user_args.encoder_lr,
        decoder_lr=user_args.decoder_lr,
        other_lr=user_args.other_lr,
        opt_lr=user_args.opt_lr,
        warmup_ratio=user_args.warmup_ratio,
        weight_decay=user_args.weight_decay,
        eps=user_args.eps,
        loss_log_freq=user_args.loss_log_freq,
        ema=user_args.ema,
        adv=user_args.adv
    )
