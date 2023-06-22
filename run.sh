#=====================================
#Install requirements
pip install -r requirements.txt

#=====================================
#训练并推理得到检索结果
python train_retrieval.py
python inference_retrieval.py

#=====================================
#训练重排序模型，预训练权重分别选用xlm_roberta-large和infoxlm-large，logits融合得出rerank结果
python train_rerank_xlm_roberta.py
python train_rerank_infoxlm.py
python inference_rerank.py

#=====================================
#训练生成模型并生成伪标签
python train_gen_pseudo.py
python inference_gen_pseudo.py

#将伪标签数据加进训练，得出的模型推理得到最终结果
python train_gen_infer.py
python inference_gen_result.py

#=====================================