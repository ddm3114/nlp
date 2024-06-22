import torch
from dataset import My_Dataset
from torch.utils.data import DataLoader
from utlis import load_dict,process_text,load_weight
from get_model import get_model
import data_augment as da
import os
if torch.cuda.is_available():
    device = torch.device('cuda')
import torch.nn as nn
import json

def inference(config,batch=10):
    device = "cuda"
    data_root = config['data_root']
    load_path = os.path.join(config['save_dir'],'model.pth')

    dataset = My_Dataset(data_root)
    label_list = dataset.label_list
    test_dataset = dataset.test_dataset
    test_dataloader = DataLoader(test_dataset,batch_size=batch,shuffle=True)
    model_name = config['model']
    model = get_model(model_name,
                      pretrained=config['pretrained'],
                      num_classes=config['num_classes'],
                      hidden_dim=config['hidden_dim'])
    model = load_weight(model,load_path)

    model.to(device)
    
    model.eval()
    for i,(imgs,labels) in enumerate(test_dataloader):
        raw_text = imgs
        imgs = da.get_response(model = da.model,tokenizer= da.tokenizer,contents= list(imgs))
        print("get_response done")
        imgs,_ = process_text((imgs,labels),model_name)
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        _,pred = torch.max(outputs,1)
        pred_label = [label_list[p] for p in pred]
        labels = [label_list[l] for l in labels]
        for i in range(len(pred_label)):
            print(f"<{i}>对于案件:'{raw_text[i][:50]}',\n模型预测为'{pred_label[i]}',\n实际为'{labels[i]}'\n")
        break

if __name__ == '__main__':
    path =[ 'text_classification/Bert1/config.json','text_classification/RoBerta1/config.json']
    for p in path:
        with open(p) as f:
            config = json.load(f)
        inference(config)