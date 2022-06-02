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

valid_transformer = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),])

voc = VOC(batch_size=BATCH_SIZE, year="2007")
train_loader = voc.get_loader(transformer=train_transformer, datatype='train')
valid_loader = voc.get_loader(transformer=valid_transformer, datatype='val')

# load model
model_path = "model_param_grad_per_class(ep=40, train).h5"
model = vgg16().to(device)
model.load_state_dict(torch.load(model_path))
# pretrained_model  = models.vgg16(pretrained=True).to(device)
# model_dict = model.state_dict()
# pretrained_dict = pretrained_model.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)

# Momentum / L2 panalty
optimizer_li = []
scheduler_li = []
for i in range(0, 20):
    optimizer_li.append(optim.SGD(model.classifiers[i].parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9))
    scheduler_li.append(optim.lr_scheduler.MultiStepLR(optimizer=optimizer_li[i],
                                            milestones=[3, 13, 23],
                                              gamma=0.1))
total_optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9)
total_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=total_optimizer,
                                        milestones=[3, 13, 23],
                                        gamma=0.1)

criterion = nn.BCEWithLogitsLoss()

# print(model.classifiers)
# print(model.classifiers[0].parameters)
# print(optimizer_li[0].state_dict)
# print(total_optimizer.state_dict)

best_loss = 100
train_iter = len(train_loader)
valid_iter = len(valid_loader)

model = model.to(device)
for i in range(20):
  model.classifiers[i] = model.classifiers[i].to(device)

model.train()
for e in range(EPOCH):
    print("epoch : "+str(e))
    train_loss = 0
    valid_loss = 0
    train_loss_class = []
    valid_loss_class = []
    # train_acc_class = []
    # valid_acc_class = []
    # train_correct_class = []
    # valid_correct_class = []
    
    for idx in range(20):
        train_loss_class.append(0)
        valid_loss_class.append(0)
        # train_acc_class.append(0)
        # valid_acc_class.append(0)
        # train_correct_class.append(0)
        # valid_correct_class.append(0)

    for i, (images, targets) in tqdm(enumerate(train_loader), total=train_iter):
        images = images.to(device)
        targets = targets.to(device)

        # forward
        for idx in range(20):
            for k in range(20):
                if(k==idx):
                    for param in model.classifiers[idx].parameters():
                        param.requires_grad = True
                else:
                    for param in model.classifiers[k].parameters():
                        param.requires_grad = False
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
            
            # train_correct_class[idx] += (torch.round(torch.sigmoid(pred))==class_targets).sum().item()
            # backward
            optimizer_li[idx].zero_grad()
            loss.backward()
            # weight update
            optimizer_li[idx].step()
        # total_loss.backward()
        # total_optimizer.step()
    for index in range(20):
        scheduler_li[index].step()
        train_loss_class[index]/=train_iter
        print(VOC_CLASSES[index] + " : " + str(train_loss_class[index]))
        # train_correct_class[index]/=train_iter
        # print(VOC_CLASSES[index] + " : " + str(train_loss_class[index]) + "  " + str(train_correct_class))
    # total_scheduler.step()

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
                # valid_correct_class[idx] += (torch.round(torch.sigmoid(pred))==class_targets).sum().item()

    total_valid_loss = (valid_loss /20) / valid_iter
    for index in range(20):
        valid_loss_class[index]/=train_iter
        print(VOC_CLASSES[index] + " : " + str(valid_loss_class[index]))
        # valid_correct_class[index]/=train_iter
        # print(VOC_CLASSES[index] + " : " + str(valid_loss_class[index]) + "  " + str(valid_correct_class))

    print("[train loss / %f] [valid loss / %f]" % (total_train_loss, total_valid_loss))
    print(" ")

    if best_loss > total_valid_loss:
        print("model saved")
        torch.save(model.state_dict(), 'model.h5')
        best_loss = total_valid_loss
