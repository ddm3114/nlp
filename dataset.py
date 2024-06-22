
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
import json
if torch.cuda.is_available():
    device = torch.device('cuda')

torch.cuda.init()
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100,CIFAR10
from torch.utils.data import DataLoader,Dataset

class Augmentation_Dataset:
    def __init__(self,dataset,transform=None):
        self.transform = transform
        self.dataset = dataset
        self.length = int(dataset.__len__()//2)
        
        if dataset.classes:
            self.num_classes = len(dataset.classes)
        elif dataset.num_classes:
            self.num_classes = dataset.num_classes
        else:
            raise ValueError('Number of classes not found')
        self.cutmix = T.CutMix(num_classes = self.num_classes)
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        img,label = self.augmentation(idx)
        
        
        
        return img, label
    
    def augmentation(self,idx):
        imgs = []
        labels = []

        image, label = self.dataset.__getitem__(2*idx)
        imgs.append(image)
        labels.append(label)
        image, label = self.dataset.__getitem__(2*idx+1)
        imgs.append(image)
        labels.append(label)


        # to_tensor = ToTensor()
       
        labels = torch.tensor(labels)
        imgs = torch.stack(imgs)
        img, label = self.cutmix(imgs, labels)
        return img[0],label[0]
                  

# 加载CIFAR-100数据集
class CIFAR100_Dataset:
    def __init__(self,transform = None):
        train_dataset = CIFAR100(root='./dataset', train=True, download=True, transform =transform)
        print(type(train_dataset))
        test_dataset = CIFAR100(root='./dataset', train=False, download=True, transform=transform)
        num_classes = len(train_dataset.classes)
        print('CIFAR100 loaded')
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        print(f"类别数: {num_classes}")
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_classes = num_classes
    
   
class CIFAR10_Dataset:
    def __init__(self,transform = None):
        train_dataset = CIFAR10(root='./dataset', train=True, download=True, transform =transform)
        print(type(train_dataset))
        test_dataset = CIFAR10(root='./dataset', train=False, download=True, transform=transform)
        num_classes = len(train_dataset.classes)
        print('CIFAR10 loaded')
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        print(f"类别数: {num_classes}")
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_classes = num_classes

class Custom_Dataset(Dataset):
    def __init__(self, datas,label_list):
        self.label_list = label_list
        self.datas = datas
        self.length = len(datas)
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        text = self.datas[idx]['案情描述']
        label_name = self.datas[idx]['案件类别']
        label = self.label_list.index(label_name)

        return text, label

class My_Dataset:
    def __init__(self,data_root):
        self.data_root = data_root
        with open(data_root,'r',encoding='utf-8') as f:
            datas = json.load(f)
        self.length = len(datas)
        self.label_list = ['刷单返利类',
                           '虚假网络投资理财类',
                           '冒充电商物流客服类',
                           '贷款、代办信用卡类',
                           '网络游戏产品虚假交易类',
                           '虚假购物、服务类',
                           '冒充公检法及政府机关类',
                           '网黑案件',
                           '虚假征信类',
                           '冒充领导、熟人类',
                           '冒充军警购物类诈骗',
                           '网络婚恋、交友类（非虚假网络投资理财类）'
                           ]
        
        self.label_len = len(self.label_list)
        self.train_dataset = Custom_Dataset(datas[:int(0.9*self.length)],self.label_list)
        self.val_dataset = Custom_Dataset(datas[int(0.9*self.length):int(0.98*self.length)],self.label_list)
        self.test_dataset = Custom_Dataset(datas[int(0.98*self.length):],self.label_list)
        self.num_classes = self.label_len


if __name__ == '__main__':
    # dataset = CIFAR100_Dataset(transform=transform)
    # train_dataset,test_dataset = dataset.train_dataset,dataset.test_dataset
    # num_classes = dataset.num_classes
    # augmented_dataset = Augmentation_Dataset(train_dataset,transform=transform)
    # train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    # augmented_dataloader = DataLoader(augmented_dataset,batch_size=2,shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    # for i, (img, label) in enumerate(augmented_dataloader):
    #     print(img.shape,label.shape)
        
    #     img_pth = f'result/image_{i}.png'
        
    #     show_image(img[0],img_pth)
    #     print("label:",label[0])
    #     break
    data_root = 'dataset/train.json'
    dataset = My_Dataset(data_root)
    train_dataset = dataset.train_dataset
    print(len(train_dataset))
    val_dataset = dataset.val_dataset
    print(len(val_dataset))
    test_dataset = dataset.test_dataset
    print(len(test_dataset))