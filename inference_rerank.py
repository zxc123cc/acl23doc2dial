# -*- coding: utf-8 -*-

# author:Administrator
# contact: test@test.com
# datetime:2023/3/2 15:04
# software: PyCharm

"""
文件说明：
    
"""

from typing import Any, Dict, Iterable, List, Union

import torch
import torch.nn.functional as F

import sys
sys.path.append('./rerank_src')
import json
from tqdm import tqdm
from modelscope.pipelines.nlp import DocumentGroundedDialogRerankPipeline
from typing import Union
from rerank_src.rerank_model import RerankModel
from modelscope.preprocessors.nlp.document_grounded_dialog_rerank_preprocessor import \
    DocumentGroundedDialogRerankPreprocessor
# from rerank_src.document_grounded_dialog_rerank_preprocessor import DocumentGroundedDialogRerankPreprocessor


def to_distinct_doc_ids(passage_ids):
    doc_ids = []
    for pid in passage_ids:
        # MARK
        doc_id = pid
        if doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids

def merge_rerank(pids1,pids2):
    pid_rank = {}
    pid_score = {}
    def handle(pid,rank,score):
        if pid not in pid_rank:
            pid_rank[pid] = rank
            pid_score[pid] = score
        else:
            pid_rank[pid] += rank

    for i,(tmp1,tmp2) in enumerate(zip(pids1,pids2)):
        pid1,score1 = tmp1
        pid2,score2 = tmp2
        handle(pid1,i,score1)
        handle(pid2,i,score2)

    results = []
    for pid,rank in pid_rank.items():
        results.append((pid,rank,pid_score[pid]))
    # results.sort(key=lambda x: x[1], reverse=False)
    results.sort(key=lambda x: (x[1], -1 * x[2]), reverse=False)
    return results


class myDocumentGroundedDialogRerankPipeline(DocumentGroundedDialogRerankPipeline):
    def __init__(self,
                 model: Union[RerankModel, str],
                 model2:Union[RerankModel, str],
                 preprocessor: DocumentGroundedDialogRerankPreprocessor = None,
                 config_file: str = None,
                 device: str = 'cuda',
                 auto_collate=True,
                 seed: int = 88,
                 **kwarg):
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate,
            seed=seed,
            **kwarg
        )
        self.model2 = model2
        self.model2.to(self.device)
        self.model2.eval()

    def save(self, addr):
        file_out = open(addr, 'w')
        for every_dict in self.guess:
            file_out.write(json.dumps(every_dict) + '\n')
    def one_instance(self, input_ids, attention_mask):
        all_probs = []
        for start_ndx in range(0, len(input_ids), self.args['max_batch_size']):
            logits1 = self.model({
                'input_ids':input_ids[start_ndx:start_ndx + self.args['max_batch_size']],
                'attention_mask':attention_mask[start_ndx:start_ndx + self.args['max_batch_size']]
            }).logits.detach().cpu()

            logits2 = self.model2({
                'input_ids':input_ids[start_ndx:start_ndx + self.args['max_batch_size']],
                'attention_mask':attention_mask[start_ndx:start_ndx + self.args['max_batch_size']]
            }).logits.detach().cpu()

            logits =  (logits1 + logits2) / 2.0
            probs = F.softmax(logits,dim=-1)[:, 1].numpy().tolist()
            all_probs.extend(probs)
        # max_score_tmp = max(all_probs[self.args['max_batch_size']:])
        # if max_score_tmp >= 0.1:
        #     print(max_score_tmp)
        #     return all_probs, len(all_probs)
        # else:
        #     return all_probs, self.args['max_batch_size']
        return all_probs, len(all_probs)

    def forward(self, dataset: Union[list, Dict[str, Any]],
                **forward_params) -> Dict[str, Any]:
        self.guess = []
        with torch.no_grad():
            for jobj in dataset:
                inst_id = jobj['id']
                probs ,len_probs= self.one_instance(jobj['input_ids'],
                                          jobj['attention_mask'])
                passages = jobj['passages']
                query = jobj['query']
                scored_pids = [(p['pid'], prob)
                               for p, prob in zip(passages[:len_probs], probs[:len_probs])]
                scored_pids.sort(key=lambda x: x[1], reverse=True)
                wids = to_distinct_doc_ids([
                    pid for pid, _ in scored_pids
                ])  # convert to Wikipedia document ids
                # ori_scored_pids = [(p['pid'], prob)
                #                    for p, prob in zip(passages, probs)]
                # merge_rerank_pids = merge_rerank(ori_scored_pids,scored_pids)
                # wids = to_distinct_doc_ids([
                #     pid for pid, _ ,_ in merge_rerank_pids
                # ])
                pred_record = {
                    'id':
                        inst_id,
                    'input':
                        query,
                    'scored_pids':
                        scored_pids,
                    'output': [{
                        'answer':
                            '',
                        'provenance': [{
                            'wikipedia_id': wid
                        } for wid in wids]
                    }]
                }
                if self.args['include_passages']:
                    pred_record['passages'] = passages
                self.guess.append(pred_record)


def main():
    model_dir = './model_storage/rerank_output_roberta'
    model_dir2 = './model_storage/rerank_output_infoxlm'
    model_configuration = {
        "framework": "pytorch",
        "task": "document-grounded-dialog-rerank",
        "model": {
            "type": "doc2bot"
        },
        "pipeline": {
            "type": "document-grounded-dialog-rerank"
        },
        "preprocessor": {
            "type": "document-grounded-dialog-rerank"
        }
    }
    file_out = open(f'{model_dir}/configuration.json', 'w')
    json.dump(model_configuration, file_out, indent=4)
    file_out.close()

    file_out = open(f'{model_dir2}/configuration.json', 'w')
    json.dump(model_configuration, file_out, indent=4)
    file_out.close()
    args = {
        'output': './',
        'max_batch_size': 32,
        'exclude_instances': '',
        'include_passages': False,
        'do_lower_case': True,
        'max_seq_length': 512,
        'query_length': 195,
        'tokenizer_resize': True,
        'model_resize': True,
        'kilt_data': True
    }
    model = RerankModel(model_dir)
    model2 = RerankModel(model_dir2)
    mypreprocessor = DocumentGroundedDialogRerankPreprocessor(
        model_dir, **args)
    pipeline_ins = myDocumentGroundedDialogRerankPipeline(
        model=model,model2=model2, preprocessor=mypreprocessor, **args)

    file_in = open('./results/input_test.jsonl', 'r')
    all_querys = []
    for every_query in file_in:
        all_querys.append(json.loads(every_query))
    passage_to_id = {}
    ptr = -1
    for file_name in ['fr', 'vi']:
        with open(f'./all_passages/{file_name}.json') as f:
            all_passages = json.load(f)
            for every_passage in all_passages:
                ptr += 1
                passage_to_id[every_passage] = str(ptr)

    file_in = open('./results/evaluate_result_retrieval_test_100.json', 'r')
    retrieval_result = json.load(file_in)['outputs']
    input_list = []
    passages_list = []
    ids_list = []
    output_list = []
    positive_pids_list = []
    ptr = -1
    for x in tqdm(all_querys):
        ptr += 1
        now_id = str(ptr)
        now_input = x
        now_wikipedia = []
        now_passages = []
        all_candidates = retrieval_result[ptr]
        for every_passage in all_candidates:
            get_pid = passage_to_id[every_passage]
            now_wikipedia.append({'wikipedia_id': str(get_pid)})
            now_passages.append({"pid": str(get_pid), "title": "", "text": every_passage})
        now_output = [{'answer': '', 'provenance': now_wikipedia}]
        input_list.append(now_input['query'])
        passages_list.append(str(now_passages))
        ids_list.append(now_id)
        output_list.append(str(now_output))
        positive_pids_list.append(str([]))
    evaluate_dataset = {'input': input_list, 'id': ids_list, 'passages': passages_list, 'output': output_list,
                        'positive_pids': positive_pids_list}
    pipeline_ins(evaluate_dataset)
    pipeline_ins.save(f'./results/rerank_output_ensemble.jsonl')

if __name__ == '__main__':
    main()