import os
from tkinter import image_names
from tkinter.ttk import OptionMenu
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import vgg16
import torchvision.transforms as transforms
from datasets.loader import VOC
from torchvision import models

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)
MODEL_PATH = 'model.h5'
BATCH_SIZE = 16
EPOCH = 40


ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

# augmentation
train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),])

train_transformer_hard = transforms.Compose([transforms.RandomRotation(90),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),])

valid_transformer = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),])

voc = VOC(batch_size=BATCH_SIZE, year="2007")
train_loader = voc.get_loader(transformer_default=train_transformer, transformer_hard=train_transformer_hard, datatype='train')
valid_loader = voc.get_loader(transformer_default=valid_transformer, transformer_hard=train_transformer_hard,  datatype='val')

# load model
model = vgg16(pretrained=True).to(device)
model_dict = model.state_dict()

print("our model")
print(model_dict.keys())

for i, (name, param) in enumerate(model.features.named_parameters()):
    param.requires_grad = False

# Momentum / L2 panalty
total_optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9)
total_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=total_optimizer,
                                        milestones=[30, 80],
                                        gamma=0.1)

criterion = nn.BCEWithLogitsLoss()

best_loss = 100
train_iter = len(train_loader)
valid_iter = len(valid_loader)

model = model.to(device)
model.train()
for e in range(EPOCH):
    print("epoch : "+str(e))
    train_loss = 0
    valid_loss = 0
    train_loss_class = []
    valid_loss_class = []
    
    for idx in range(20):
        train_loss_class.append(0)
        valid_loss_class.append(0)

    for i, (images, targets) in tqdm(enumerate(train_loader), total=train_iter):
        images = images.to(device)
        targets = targets.to(device)

        # forward
        for idx in range(20):
            class_targets = []
            for j in range(targets.shape[0]):
                li = []
                li.append(targets[j][idx])
                class_targets.append(li)
            class_targets = torch.tensor(class_targets).to(device)
            
            pred = model(images, idx)
            # loss
            loss = criterion(pred.double(), class_targets)
            train_loss += loss.item()
            train_loss_class[idx]+=loss.item()
            if(idx==0):
                train_total_loss = loss
            else:
                train_total_loss += loss

        total_optimizer.zero_grad()
        train_total_loss.backward()
        total_optimizer.step()

    total_scheduler.step()
    for index in range(20):
        # scheduler_li[index].step()
        train_loss_class[index]/=train_iter
        print(VOC_CLASSES[index] + " : " + str(train_loss_class[index]))

    total_train_loss = (train_loss / 20) / train_iter

    with torch.no_grad():
        for images, targets in valid_loader:
            images = images.to(device)
            targets = targets.to(device)
            for idx in range(20):
                class_targets = []
                for j in range(targets.shape[0]):
                    li = []
                    li.append(targets[j][idx])
                    class_targets.append(li)
                class_targets = torch.tensor(class_targets).to(device)

                pred = model(images, idx)
                # loss
                loss = criterion(pred.double(), class_targets)
                valid_loss += loss.item()
                valid_loss_class[idx] += loss.item()

    total_valid_loss = (valid_loss /20) / valid_iter
    for index in range(20):
        valid_loss_class[index]/=train_iter
        print(VOC_CLASSES[index] + " : " + str(valid_loss_class[index]))

    print("[train loss / %f] [valid loss / %f]" % (total_train_loss, total_valid_loss))

    if best_loss > total_valid_loss:
        best_loss = total_valid_loss
        print("model saved\n")
        torch.save(model.state_dict(), 'model.h5')
