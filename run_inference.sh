#=====================================
#Install requirements
pip install -r requirements.txt

#=====================================
#推理得到检索结果
python inference_retrieval.py

#=====================================
#推理得到重排序结果
python inference_rerank.py

#=====================================
#推理生成伪标签
#python inference_gen_pseudo.py

#推理生成最终结果
python inference_gen_result.py
#=====================================