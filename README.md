# 2023 ACL Doc2dial workshop, final rank 2
本团队在baseline的基础上进行改进，成绩为初赛第一，复赛第二，最终生成的结果文件为`results/test_result.json`。此方案对显存有一定要求（建议80G及以上）。

## 全流程复现
包含训练和推理全部流程（用时较长）
> run.sh

## 推理结果复现
无需训练模型，直接加载已经训练好的权重推理得出结果
> run_inference.sh


## 总体流程概述

### 检索任务

检索模型训练，得到检索模型
> python train_retrieval.py，使用的预训练权重是 xlm-roberta-base，[下载地址](https://huggingface.co/xlm-roberta-large/tree/main)

检索模型预测，得到检索结果
> python inference_retrieval.py

### 重排模型

重排模型1训练
> python train rerank_xlm_roberta.py，使用的预训练权重是 xlm-roberta-large，[下载地址](https://huggingface.co/xlm-roberta-large/tree/main)

重排模型2训练
> python train rerank_infoxlm.py，使用的预训练权重是 infoxlm-large，[下载地址](https://huggingface.co/microsoft/infoxlm-large/tree/main)

重排模型1和重排模型2融合预测，得到重排结果
> python inference_rerank.py

### 生成模型

生成模型训练，使用的预训练权重为本赛题baseline中nlp_convai_generation_pretrain中的权重,我们把它提了出来放在了`pretrain_storage/gen_pretrain_model`文件夹下，
实际加载的权重为`pretrain_storage/gen_pretrain_model/re2g/pytorch_model.bin`

> python train_gen_pseudo.py

生成模型预测测试集得到伪标签
> python inference_gen_pseudo.py

将伪标签加入到训练集，进行一个新的生成模型训练
> python train_gen_infer.py

生成模型预测，得到生成结果
> python inference_gen_result.py


(赛中尝试利用中英翻译为越法数据，不过没有提升，此处放置调用百度api翻译的代码及部分翻译结果，实际并没有使用到)