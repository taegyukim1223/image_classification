from email import parser
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

parser = argparse.ArgumentParser(description='image classification.')
parser.add_argument('--resolution', type=int, default=256)
parser.add_argument('--data_path', type=str, default=os.path.dirname(os.path.realpath(__file__)))
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--gpus', type=int, nargs='+', default = 0, help = 'type the gpus index when using the multi-gpus')
parser.add_argument('--model_load', type=str, help = 'model Name of Timm library or your own model path', default='resnet34')
args = parser.parse_args()

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
transforms_test = transforms.Compose([
    transforms.Resize(resolution),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 정규화(normalization)
])

test_datasets = datasets.ImageFolder(os.path.join(args.data_path, 'test'), transforms_test)
test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=1, shuffle=True)

print('test 데이터셋 크기:', len(test_datasets))

    
class_names = test_datasets.classes
print('클래스:', class_names)

#모델 로드
model = torch.load(args.model_load, map_location=map_location)
criterion = nn.CrossEntropyLoss()
print(model)

model.eval()
start_time = time.time()
array1 = []
running_TP = 0
running_FP = 0
running_FN = 0
running_TN = 0
a = torch.tensor([0]).to(device)
b = torch.tensor([1]).to(device)

with torch.no_grad():
    running_loss = 0.
    running_corrects = 0


    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        array1.append([preds,labels.data])

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        if preds == labels.data == b :
            running_TP += 1
        if preds == b and labels.data == a :
            running_FP += 1
        if preds == a and labels.data == b :
            running_FN += 1
        if preds == labels.data == a :
            running_TN += 1

    recall = running_TP / (running_TP + running_FN)
    precision = running_TP / (running_TP + running_FP)
    f1_score = 2*recall*precision / (precision + recall)
    print('recall:{} precision:{} f1_score{}'.format(recall, precision, f1_score))
    epoch_loss = running_loss / len(test_datasets)
    epoch_acc = running_corrects / len(test_datasets) * 100.
    print('[Val Phase] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch_loss, epoch_acc, time.time() - start_time))