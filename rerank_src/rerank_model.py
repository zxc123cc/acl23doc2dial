import torch.nn as nn
from modeling_roberta import XLMRobertaForSequenceClassification

class RerankModel(nn.Module):
    def __init__(self, model_dir):
        super(RerankModel, self).__init__()
        self.model_dir= model_dir
        # self.base_model = XLMRobertaForSequenceClassification.from_pretrained(model_dir)
        self.model = ClassifyRerank(model_dir)

    def forward(self, input):
        outputs = self.model(input_ids=input['input_ids'],
                             attention_mask=input['attention_mask'])
        return outputs

    def resize_token_embeddings(self, size):
        self.model.base_model.resize_token_embeddings(size)

    def save_pretrained(self, addr):
        self.model.base_model.save_pretrained(addr)


class ClassifyRerank(nn.Module):

    def __init__(self, model_dir):
        super().__init__()
        self.base_model = XLMRobertaForSequenceClassification.from_pretrained(model_dir)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                *args,
                **kwargs):
        outputs = self.base_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        return outputs