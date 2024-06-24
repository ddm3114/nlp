from torch.nn import Identity
import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM,BertTokenizer,RobertaForSequenceClassification
from transformers import RobertaTokenizer

class roberta(nn.Module):
    def __init__(self,model_path):
        super(roberta, self).__init__()
        self.roberta = AutoModelForMaskedLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.roberta.cls.predictions.decoder = Identity()
        self.fc = nn.Linear(768, 1024)

    def forward(self, input_ids):
        if  isinstance(input_ids,(list,tuple)):
            inputs = self.process(input_ids)
            print(type(inputs))
        else:
            inputs = input_ids
        
        outputs = self.roberta(**inputs)
        class_token = outputs.logits[:,0,:]
        outputs = self.fc(class_token)

        return outputs
    def process(self,input_ids):
        if not isinstance(input_ids,list):
            input_ids = list(input_ids)

        inputs = self.tokenizer(input_ids, return_tensors="pt", padding='max_length', truncation=True,max_length=512)
        return inputs
    
model_path = 'RoBerta/chinese'
tokenizer = AutoTokenizer.from_pretrained(model_path) 

def process(input_ids):
    if not isinstance(input_ids,list):
        input_ids = list(input_ids)

    inputs = tokenizer(input_ids, return_tensors="pt", padding='max_length', truncation=True,max_length=512)
    return inputs

if __name__ == '__main__':
    model_path = "RoBerta/chinese"
    model = roberta(model_path)
    model.to('cuda')
    print(model)
    datas = ["你叫什么名字123", "我叫张三"]
    datas = process(datas)
    datas.to('cuda')
    outputs = model(datas)
    print(outputs.shape)
