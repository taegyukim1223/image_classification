from email import parser
from xmlrpc.client import Boolean, boolean
import timm
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import time
from tqdm import tqdm
import argparse
import wandb



parser = argparse.ArgumentParser(description='image classification.')
parser.add_argument('--epoch', type=int,   default=50)
parser.add_argument('--batch_size', type=int,   default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--resolution', type=int, default=256)
parser.add_argument('--data_path', type=str, default=os.path.dirname(os.path.realpath(__file__)))
parser.add_argument('--model_save_name', type=str, default='temp')
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--gpus', type=int, nargs='+', default = 0, help = 'type the gpus index when using the multi-gpus')
parser.add_argument('--class_num', type=int, help = 'The number of classes.', default = 2)
parser.add_argument('--model_load', type=str, help = 'model Name of Timm library or your own model path', default='resnet34')
parser.add_argument('--wandb_project_name', type=str, help = 'wandb project name', default="")

args = parser.parse_args()

#wandb init
if args.wandb != "" :
    wandb.login()
    experiment = wandb.init(project = args.wandb_project_name, entity = "")
    experiment.config.update(dict(epochs=args.epoch, batch_size=args.batch_size, learning_rate=args.lr))

resolution = (args.resolution, args.resolution) 
map_location = 'cuda:{}'.format(args.gpu_num)

device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU

print ('Current cuda device ', torch.cuda.current_device()) # check

# Additional Infos
if device.type == 'cuda':
    print(torch.cuda.get_device_name(args.gpu_num))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(args.gpu_num)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(args.gpu_num)/1024**3,1), 'GB')
    
# 데이터셋을 불러올 때 사용할 변형(transformation) 객체 정의
transforms_train = transforms.Compose([
    transforms.Resize(resolution),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 정규화(normalization)
])

transforms_val = transforms.Compose([
    transforms.Resize(resolution),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 정규화(normalization)
])

train_datasets = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transforms_train)
val_datasets = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transforms_val)
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=1, shuffle=True)

print('학습 데이터셋 크기:', len(train_datasets))
print('valid 데이터셋 크기:', len(val_datasets))
    
class_names = train_datasets.classes
print('클래스:', class_names)

#모델 로드
try :
    model = torch.load(args.model_load, map_location=map_location)
except :
    #전이 학습(transfer learning): 모델의 출력 뉴런 수를 클래스 갯수로 교체하여 마지막 레이어 다시 학습
    model = timm.create_model(args.model_load, pretrained=True, num_classes=args.class_num)

print(model)

if args.gpus != 0 :
    model = torch.nn.DataParallel(model, device_ids=args.gpus) # GPU병렬연결
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
model = model.to(device)

model.train()
start_time = time.time()

# 전체 반복(epoch) 수 만큼 반복하며
for epoch in range(args.epoch):
    running_loss = 0.
    running_corrects = 0

    # 배치 단위로 학습 데이터 불러오기
    for inputs, labels in tqdm(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 모델에 입력(forward)하고 결과 계산
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # 역전파를 통해 기울기(gradient) 계산 및 학습 진행
        loss.backward()
        optimizer.step()
        experiment.log({
                    'train loss per batch': loss.item()
                        })
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_datasets)
    epoch_acc = running_corrects / len(train_datasets) * 100.
    experiment.log({
                    'train loss per epoch': epoch_loss,
                    'train acc per epoch' : epoch_acc
                    })
    # 학습 과정 중에 결과 출력
    print('#{} Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))
    
    # 1epoch 진행 후 validat data!
    model.eval()
    start_time = time.time()
    acc_past = 0
    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0

        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_datasets)
        epoch_acc = running_corrects / len(val_datasets) * 100.
        print('[Test Phase] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch_loss, epoch_acc, time.time() - start_time))
        experiment.log({
                    'validation loss': epoch_loss,
                    'validation acc' : epoch_acc
                    })
        torch.save(model,'latest_{}.pth'.format(args.model_save_name))
        if acc_past < epoch_acc :
            torch.save(model,'best_{}.pth'.format(args.model_save_name))
            acc_past = epoch_acc