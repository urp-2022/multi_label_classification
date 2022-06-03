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
model = vgg16(pretrained=True).to(device)
# pretrained_model  = models.vgg16(pretrained=True).to(device)
model_dict = model.state_dict()
# pretrained_dict = pretrained_model.state_dict()

print("our model")
print(model_dict.keys())
# print("\n\npretrianed model")
# print(pretrained_dict.keys())

# print("\n\n\n\n")

# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)

# print("\nour model")
# print(model)
# print("\n\npretrianed model")
# print(pretrained_model)

# Momentum / L2 panalty
# optimizer_li = []
# scheduler_li = []
# for i in range(0, 20):
#     optimizer_li.append(optim.SGD(model.classifiers[i].parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9))
#     scheduler_li.append(optim.lr_scheduler.MultiStepLR(optimizer=optimizer_li[i],
#                                             milestones=[3, 13, 23],
#                                             gamma=0.1))
total_optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9)
total_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=total_optimizer,
                                        milestones=[30, 80],
                                        gamma=0.1)

criterion = nn.BCEWithLogitsLoss()

best_loss = 100
train_iter = len(train_loader)
valid_iter = len(valid_loader)

model = model.to(device)
# for i in range(20):
#   model.classifiers[i] = model.classifiers[i].to(device)

model.train()
for e in range(EPOCH):
    print("epoch : "+str(e))
    train_loss = 0
    valid_loss = 0
    train_loss_class = []
    valid_loss_class = []
    
    for idx in range(1):
        train_loss_class.append(0)
        valid_loss_class.append(0)

    for i, (images, targets) in tqdm(enumerate(train_loader), total=train_iter):
        images = images.to(device)
        targets = targets.to(device)

        # forward
        for idx in range(1):
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
    for index in range(1):
        # scheduler_li[index].step()
        train_loss_class[index]/=train_iter
        print(VOC_CLASSES[index] + " : " + str(train_loss_class[index]))

    total_train_loss = (train_loss / 1) / train_iter

    with torch.no_grad():
        for images, targets in valid_loader:
            images = images.to(device)
            targets = targets.to(device)
            for idx in range(1):
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

    total_valid_loss = (valid_loss /1) / valid_iter
    for index in range(1):
        valid_loss_class[index]/=train_iter
        print(VOC_CLASSES[index] + " : " + str(valid_loss_class[index]))

    print("[train loss / %f] [valid loss / %f]" % (total_train_loss, total_valid_loss))
    print(" ")

    if best_loss > total_valid_loss:
        print("model saved")
        torch.save(model.state_dict(), 'model.h5')
        best_loss = total_valid_loss
