import os
from tkinter import image_names
from tkinter.ttk import OptionMenu
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model_resnet import resnet18,resnet34,resnet50
import torchvision.transforms as transforms
from datasets.loader_custom_v3 import VOC

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)
MODEL_PATH = 'model.h5'
BATCH_SIZE = 16
EPOCH = 200


ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

# augmentation
voc = VOC(batch_size=BATCH_SIZE, year="2007")

train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),])

valid_transformer = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),])

train_transformer_hard = transforms.Compose([transforms.RandomRotation(90),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),])

train_loader = voc.get_loader(
    base_transformer=train_transformer, 
    hard_transformer=train_transformer_hard,
    datatype='train',
    augClassList=[]
)

valid_loader = voc.get_loader(
    base_transformer=valid_transformer, 
    hard_transformer=train_transformer_hard,
    datatype='val',
    augClassList=[]
)

aug_class_list = [4, 15, 17]
train_hard_loader = []
for i in aug_class_list:
    train_hard_loader.append(voc.get_loader(
        base_transformer=train_transformer,
        hard_transformer=train_transformer_hard,
        datatype='train',
        augClassList=[i]
    ))

for i in range(len(train_hard_loader)):
    print(len(train_hard_loader[i]))

# load model
model = resnet34(pretrained=True).to(device)
model_dict = model.state_dict()

print("our model")
print(model_dict.keys())


# Momentum / L2 panalty
# total_optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9)
# total_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=total_optimizer,
#                                         milestones=[30, 80],
#                                         gamma=0.1)
total_optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9)
total_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=total_optimizer,
                                        milestones=[2, 5, 15, 25, 40, 80],
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
    train_loss_li = []
    for z in range(len(aug_class_list)):
        train_loss_li.append(0)
    valid_loss = 0
    
    for i, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = images.to(device)
        targets = targets.to(device)

        # forward
        for idx in range(20):
            if(idx not in aug_class_list):

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

                total_optimizer.zero_grad()
                loss.backward()
                total_optimizer.step()

    for idx in range(len(aug_class_list)):        
        for i, (images, targets) in tqdm(enumerate(train_hard_loader[idx]), total=len(train_hard_loader[idx])):
            images = images.to(device)
            targets = targets.to(device)

            # forward
            class_targets = []
            for j in range(targets.shape[0]):
                li = []
                li.append(targets[j][aug_class_list[idx]])
                class_targets.append(li)
            class_targets = torch.tensor(class_targets).to(device)
            
            pred = model(images, aug_class_list[idx])
            # loss
            loss = criterion(pred.double(), class_targets)
            train_loss_li[idx] += loss.item()

            total_optimizer.zero_grad()
            loss.backward()
            total_optimizer.step()

    total_train_loss = train_loss/len(train_loader) 
    for z in range(len(aug_class_list)):
        total_train_loss += train_loss_li[z]/len(train_hard_loader[z])
    total_train_loss /= 20
    total_scheduler.step()

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

    total_valid_loss = (valid_loss /20) / len(valid_loader)

    print("[train loss / %f] [valid loss / %f]" % (total_train_loss, total_valid_loss))

    if best_loss > total_valid_loss:
        best_loss = total_valid_loss
        print("model saved\n")
        torch.save(model.state_dict(), 'model_aug.h5')
