import sys
sys.path.append('./gen_src')
import os
import json
import re
import string
from collections import Counter
import sacrebleu
import torch
from modelscope.trainers.nlp.document_grounded_dialog_generate_trainer import logger
from rouge import Rouge
from torch.utils.data import DataLoader
from tqdm import tqdm

from gen_src.gen_trainer import DocumentGroundedDialogGenerateTrainer

from gen_src.config_pseudo import Config
from gen_src.data_helpter_gen import collate,collate_single_turn,collate_inference

user_args = Config()


with open('all_passages/id_to_passage.json', encoding='utf-8') as f:
    id_to_passage = json.load(f)

eval_dataset = []
with open('./results/rerank_output_ensemble.jsonl', encoding='utf-8') as f:
    for line in f.readlines():
        sample = json.loads(line)
        eval_dataset.append({
            'query': sample['input'],
            'rerank': json.dumps([id_to_passage[x['wikipedia_id']] for x in sample['output'][0]['provenance']],
                                 ensure_ascii=False),
            'response': '<response> @',
            'scored_pids' :sample['scored_pids']
        })

cache_path = user_args.pretrain_model_dir
trainer = DocumentGroundedDialogGenerateTrainer(
    model=cache_path,
    train_dataset=None,
    eval_dataset=eval_dataset,
)

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

def evaluate(trainer, batch_size=16, checkpoint_path=None):

    model = trainer.model.model.generator.generator
    tokenizer = trainer.preprocessor.generation_tokenizer
    device = trainer.preprocessor.device

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path)
        trainer.model.model.load_state_dict(state_dict)

    valid_loader = DataLoader(
        dataset=trainer.eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_inference)

    valid_iterator = tqdm(valid_loader, total=len(valid_loader), desc='Evaluation')

    trainer.model.model.eval()

    def _get_predictions(query,context,select_ids=0):
        generator_inputs = [
            ' '.join([query[i], '<passage>', context[i][select_ids]])
            for i in range(len(query))
        ]
        input_ids = tokenizer.batch_encode_plus(
            list(generator_inputs), padding=True, return_tensors='pt').input_ids.to(device)

        outputs = model.generate(input_ids,
                                 num_beams=user_args.infer_num_beams,
                                 max_length=user_args.infer_max_length,
                                 early_stopping=True,
                                 no_repeat_ngram_size=user_args.infer_no_repeat_ngram_size,
                                 )

        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
        return predictions

    def _get_score(output,target):
        meters = dict()

        hypothesis_list = [output.replace('<extra_id_0>', '')]
        hypothesis_list = [x if len(x) > 10 else 'placeholder' for x in hypothesis_list]
        if user_args.add_prompt:
            reference_list = [target.replace('<extra_id_0> ', '').split('<response>')[1].strip()]
        else:
            reference_list = [target.replace('<response>', '')]
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

        return sum([x for x in meters.values()])

    def _select(now_prediction_list):
        max_score = 0
        prediction = ''
        for i in range(len(now_prediction_list)):
            now_prediction = now_prediction_list[i]
            # score = len(now_prediction)
            score = 0
            for tmp in now_prediction_list:
                if tmp != now_prediction:
                    score += _get_score(now_prediction,tmp)
            print('now score: ',score)
            if max_score < score:
                max_score = score
                prediction = now_prediction
        return prediction

    def _select_output(predictions_list,scored_pids):

        predictions = []
        num = len(predictions_list)
        for i in range(len(scored_pids)):
            max_score = scored_pids[i][0][1]
            min_score = scored_pids[i][num-1][1]
            # top1 passage可信度很低且其他passage与top1 passage可信度没有相差太多，则认为这些passage都有机会成为候选passage，而不是只使用top1 passage
            if max_score < 0.01 and max_score/10 <= min_score:
                now_prediction_list = [predictions[i] for predictions in predictions_list]
                now_prediction = _select(now_prediction_list)
            else:
                now_prediction = predictions_list[0][i]
            predictions.append(now_prediction)
            if now_prediction != predictions_list[0][i]:
                print(predictions_list[0][i])
                print(now_prediction)

        return predictions


    with torch.no_grad():
        results = {'outputs': [], 'targets': []}
        for index, payload in enumerate(valid_iterator):
            query, context, label ,scored_pids= payload
            query = [
                tokenizer.decode(
                    tokenizer([x], add_special_tokens=False,return_tensors='pt')['input_ids'][0][:user_args.infer_query_max_length]
                ) for x in query
            ]
            # query = [
            #     tokenizer.decode(
            #         tokenizer([query[i]], add_special_tokens=False,return_tensors='pt')['input_ids'][0][:user_args.infer_query_max_length]
            #     ) if scored_pids[i][0][1] >=0.01 else
            #     tokenizer.decode(
            #         tokenizer([query[i]], add_special_tokens=False,return_tensors='pt')['input_ids'][0][:512]
            #     )
            #     for i in range(len(query))
            # ]

            # predictions = _get_predictions(query=query,context=context,select_ids=0)

            # predictions0 = _get_predictions(query=query,context=context,select_ids=0)
            # predictions1 = _get_predictions(query=query,context=context,select_ids=1)
            # predictions2 = _get_predictions(query=query,context=context,select_ids=2)
            # predictions = _select_output(
            #     predictions_list = [predictions0, predictions1, predictions2],scored_pids=scored_pids
            # )

            generator_inputs = [
                ' '.join([query[i], '<passage>', context[i][0]])
                for i in range(len(query))
            ]
            input_ids = tokenizer.batch_encode_plus(
                list(generator_inputs), padding=True, return_tensors='pt').input_ids.to(device)

            outputs = model.generate(input_ids,
                                     num_beams=user_args.infer_num_beams,
                                     min_length=10,
                                     max_length=user_args.infer_max_length,
                                     early_stopping=True,
                                     no_repeat_ngram_size=user_args.no_repeat_ngram_size,
                                     )

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
        result_path = './results/evaluate_result_gen.json'
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info(meters)
    return meters

print(f'loading model {user_args.checkpoint_path} ...')
evaluate(trainer, checkpoint_path=user_args.checkpoint_path)


with open(f'./results/evaluate_result_gen.json', encoding='utf-8') as f:
    predictions = json.load(f)['outputs']

if user_args.inference_mode == 'pseudo':
    with open('./results/test_pseudo.json', 'w', encoding='utf-8') as f:
        for query, prediction in zip(eval_dataset, predictions):
            f.write(json.dumps({
                'query': query['query'],
                'rerank':query['rerank'],
                'response': prediction.replace('<extra_id_0> ', '')
            }, ensure_ascii=False) + '\n')
else:
    with open('./results/test_result.json', 'w', encoding='utf-8') as f:
        for query, prediction in zip(eval_dataset, predictions):
            if user_args.add_prompt:
                f.write(json.dumps({
                    'query': query['query'],
                    'response': prediction.replace('<extra_id_0> ', '').split('<response>')[1].strip()
                }, ensure_ascii=False) + '\n')
            else:
                f.write(json.dumps({
                    'query': query['query'],
                    'response': prediction.replace('<response>','').strip()
                }, ensure_ascii=False) + '\n')
