import torch
from torchvision.io import read_image
from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights,convnext_small,ConvNeXt_Small_Weights,convnext_base,ConvNeXt_Base_Weights,convnext_large,ConvNeXt_Large_Weights
from torchvision.models import densenet121,DenseNet121_Weights,densenet161,DenseNet161_Weights,densenet169,DenseNet169_Weights,densenet201,DenseNet201_Weights
from torchvision.models import resnet18,ResNet18_Weights,resnet34,ResNet34_Weights,resnet50,ResNet50_Weights,resnet101,ResNet101_Weights,resnet152,ResNet152_Weights

from torchvision.models import swin_t,Swin_T_Weights
import bert
import roberta
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
if torch.cuda.is_available():
    device = torch.device('cuda')

torch.cuda.init()
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

# class ClassifierHead(nn.Module):
#     def __init__(self, in_features,hidden_dim = 1024, num_classes=1024):
#         super(ClassifierHead, self).__init__()
#         self.fc1 = nn.Linear(in_features, in_features, bias=True)
#         self.tanh = nn.Tanh()
#         self.dropout = nn.Dropout(0.1, inplace=False)
#         self.fc2 = nn.Linear(in_features, num_classes, bias=True)
#         self._initialize_weights()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.tanh(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
        
#         return x
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # He 初始化 (Kaiming 初始化)
#                 init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 # Xavier 初始化 (Glorot 初始化)
#                 init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)

class ClassifierHead(nn.Module):
    def __init__(self, in_features,hidden_dim = 1024, num_classes=1024):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes, bias=True)
        
        self._initialize_weights()

    def forward(self, x):

        x = self.fc(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He 初始化 (Kaiming 初始化)
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier 初始化 (Glorot 初始化)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

## 通用于调取外部模型，只需要将mymodel的名字传入即可
class baseModel(torch.nn.Module):
    def __init__(self,model_name,pretrained = True,train_backbone = False,hidden_dim = 1024,num_classes = 200):
        super(baseModel, self).__init__()
        self.model = self.create_model(model_name,pretrained)
        if train_backbone:
            for param in self.model.parameters():
                param.requires_grad = True
        else:

            for param in self.model.parameters():
                param.requires_grad = False

        if hasattr(self.model, 'head'):
            num_ftrs = self.model.head.in_features
            self.model.head = ClassifierHead(num_ftrs,hidden_dim,num_classes)     
            for param in self.model.head.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'fc'):
            num_ftrs = self.model.fc.in_features
            self.model.fc = ClassifierHead(num_ftrs,hidden_dim,num_classes)     
            for param in self.model.fc.parameters():
                param.requires_grad = True

        elif hasattr(self.model, 'classifier'):
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = ClassifierHead(num_ftrs,hidden_dim,num_classes)     
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x
    
    def create_model(self,model_name,pretrained):
        if model_name == 'ResNet18':
            if pretrained  == True:
                model = resnet18(weights = ResNet18_Weights.DEFAULT)
            else:
                model = resnet18()
            print('ResNet18 model loaded')

        elif model_name == 'ResNet34':
            if pretrained == True:
                model = resnet34(weights = ResNet34_Weights.DEFAULT)
            else:
                model = resnet34()
            print('ResNet34 model loaded')

        elif model_name == 'ResNet50':
            if pretrained == True:
                model = resnet50(weights = 'DEFAULT')
            else:
                model = resnet50()
            print('ResNet50 model loaded')

        elif model_name == 'Swin_T':
            if pretrained == True:
                model = swin_t(weights = Swin_T_Weights.DEFAULT)
            else:
                model = swin_t()
            print('Swin_T model loaded')

        elif model_name == 'Bert':
            if pretrained  == True:
                model = bert.bert("Bert/chinese")
            else:
                raise ValueError(f'only pretrained model available for this model: {model_name}')
            print('bert model loaded')
        elif model_name == 'RoBerta':
            if pretrained  == True:
                model = roberta.roberta("RoBerta/chinese")
            else:
                raise ValueError(f'only pretrained model available for this model: {model_name}')
            print('RoBerta model loaded')

        elif model_name == 'MyModel':
            model = MyModel(pretrained=pretrained)
            print('Your Custom_Model loaded')
        else:
            raise ValueError('model not supported')
        
        return model




    
class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out
    
class MyModel(nn.Module):
    def __init__(self, pretrained = True,num_classes=1000):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = BasicBlock(64,64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer2 = BasicBlock(128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self._initialize_weights()

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer1(x)

        x = self.maxpool(x)


        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He 初始化 (Kaiming 初始化)
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier 初始化 (Glorot 初始化)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He 初始化 (Kaiming 初始化)
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier 初始化 (Glorot 初始化)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

## 通用于调取外部模型，只需要将mymodel的名字传入即可
    
if __name__ == "__main__":
    
    model = baseModel(model_name='RoBerta',pretrained=True,train_backbone=False,hidden_dim=1024,num_classes=12)   
    print(model)
    datas = ["你叫什么名字", "我叫张三"]
    datas = roberta.process(datas)
    outputs = model(datas)
    print(outputs.shape)
    
    # print("swin_t: ", sum(p.numel() for p in model.parameters()))
    # model = ResNet50(pretrained= False)
    # print("resnet50: ", sum(p.numel() for p in model.parameters()))
    # model = baseModel(num_classes=10)
    # print(model)
    # for name, parms in model.named_parameters():
    #         if parms.requires_grad:
    #             print(f"{name}: {parms.data}")
