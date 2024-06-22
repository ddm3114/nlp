from torch.nn import Identity
import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


class bert(nn.Module):
    def __init__(self,model_path):
        super(bert, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.fc = self.bert.cls.predictions.decoder
        self.bert.cls.predictions.decoder = Identity()
    def forward(self, input_ids):
        if  isinstance(input_ids,(list,tuple)):
            inputs = self.process(input_ids)
            print(type(inputs))
        else:
            inputs = input_ids

        outputs = self.bert(**inputs)
        class_token = outputs.logits[:,0]
        outputs = self.fc(class_token)

        return outputs
    def process(self,input_ids):
        if not isinstance(input_ids,list):
            input_ids = list(input_ids)

        inputs = self.tokenizer(input_ids, return_tensors="pt", padding=True, truncation=True,max_length=512)
        return inputs
    
model_path = 'Bert/chinese'
tokenizer = AutoTokenizer.from_pretrained(model_path) 

def process(input_ids):
        if not isinstance(input_ids,list):
            input_ids = list(input_ids)

        inputs = tokenizer(input_ids, return_tensors="pt", padding=True, truncation=True)
        return inputs

if __name__ == '__main__':
    model_path = "Bert/chinese"
    model = bert(model_path)
    # print(model)
    datas = ["你叫什么名字", "我叫张三"]
    outputs = model(datas)
    print(outputs.shape)
