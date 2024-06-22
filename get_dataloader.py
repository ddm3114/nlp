import torch

from torch.utils.data import DataLoader
if torch.cuda.is_available():
    device = torch.device('cuda')
from utlis import read_sample,transform,save_dict
from dataset import CIFAR100_Dataset,Augmentation_Dataset,CIFAR10_Dataset,My_Dataset

def get_dataloader(dataset = 'CIFAR100',batch_size = 32,augment = False,transform = transform,data_type = 'image',data_root = None):
    if (dataset or data_root) and data_type == 'image':
        if dataset == 'CIFAR100':
            dataset = CIFAR100_Dataset(transform=transform)
            train_dataset,val_dataset = dataset.train_dataset,dataset.val_dataset
            num_classes = dataset.num_classes
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            if augment:
                augmented_dataset = Augmentation_Dataset(train_dataset,transform=transform)
                augmented_dataloader = DataLoader(augmented_dataset,batch_size=batch_size,shuffle=True)
                return train_dataloader,val_dataloader,augmented_dataloader
            
        elif dataset == 'CIFAR10':
            dataset = CIFAR10_Dataset(transform=transform)
            train_dataset,val_dataset = dataset.train_dataset,dataset.val_dataset
            num_classes = dataset.num_classes
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            if augment:
                augmented_dataset = Augmentation_Dataset(train_dataset,transform=transform)
                augmented_dataloader = DataLoader(augmented_dataset,batch_size=batch_size,shuffle=True)
                return train_dataloader,val_dataloader,augmented_dataloader
        else:
            raise NameError(f"Dataset:{dataset} for {data_type} is not supported now")


    elif (data_root or dataset)  and data_type == 'text':

        if data_root:
            dataset = My_Dataset(data_root=data_root)
            train_dataset,val_dataset = dataset.train_dataset,dataset.val_dataset
            num_classes = dataset.num_classes
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        else:
            raise NameError(f"Dataset:{dataset} for{data_type} is not supported now")
    
    else:
        raise NameError(f"Wrong data_root or data_type:{data_root} {data_type}")
    return train_dataloader,val_dataloader



if __name__ == '__main__':
    train,val = get_dataloader(data_type='text',data_root='dataset/train.json')