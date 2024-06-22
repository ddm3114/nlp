from torch.nn import Identity
import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


class roberta(nn.Module):
    def __init__(self,model_path):
        super(roberta, self).__init__()
        self.reberta = AutoModelForMaskedLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.fc = self.reberta.cls.predictions.decoder
        self.reberta.cls.predictions.decoder = Identity()
    def forward(self, input_ids):
        if  isinstance(input_ids,(list,tuple)):
            inputs = self.process(input_ids)
            print(type(inputs))
        else:
            inputs = input_ids

        outputs = self.reberta(**inputs)
        class_token = outputs.logits[:,0]
        outputs = self.fc(class_token)

        return outputs
    def process(self,input_ids):
        if not isinstance(input_ids,list):
            input_ids = list(input_ids)

        inputs = self.tokenizer(input_ids, return_tensors="pt", padding=True, truncation=True)
        return inputs
    
model_path = 'RoBerta/chinese'
tokenizer = AutoTokenizer.from_pretrained(model_path) 

def process(input_ids):
        if not isinstance(input_ids,list):
            input_ids = list(input_ids)

        inputs = tokenizer(input_ids, return_tensors="pt", padding=True, truncation=True,max_length=512)
        return inputs

if __name__ == '__main__':
    model_path = "RoBerta/chinese"
    model = roberta(model_path)
    # print(model)
    datas = ["你叫什么名字", "我叫张三"]
    input = process(datas)
    t = torch.randn(128,847)
    input.input_ids = t
    print(input.input_ids.shape)
    outputs = model(input)
    print(outputs.shape)
