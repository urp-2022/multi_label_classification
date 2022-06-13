import os
from tkinter import image_names
from tkinter.ttk import OptionMenu
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model_simple import vgg16
# from model import vgg11, vgg13, vgg16, vgg19
import torchvision.transforms as transforms
from datasets.loader_custom_v2 import VOC

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)
MODEL_PATH = 'model.h5'
BATCH_SIZE = 16
EPOCH = 100


ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

# augmentation
voc = VOC(batch_size=BATCH_SIZE, year="2007")

train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),])

valid_transformer = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),])

train_loader = voc.get_loader(
    transformer=train_transformer, 
    datatype='train',
    classtype=-1)
valid_loader = voc.get_loader(
    transformer=valid_transformer, 
    datatype='val',
    classtype=-1)


train_transformer_hard = transforms.Compose([transforms.RandomRotation(90),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),])
train_hard_loader = []
for i in range(len(VOC_CLASSES)):
    train_hard_loader.append(voc.get_loader(
        transformer=train_transformer_hard,
        datatype='train',
        classtype=i
    ))


# load model
# model = vgg11(pretrained=True).to(device)
model = vgg16(pretrained=True).to(device)
# model = vgg19(pretrained=True).to(device)
model_dict = model.state_dict()

print("our model")
print(model_dict.keys())

for i, (name, param) in enumerate(model.features.named_parameters()):
    param.requires_grad = False

# Momentum / L2 panalty
# total_optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9)
# total_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=total_optimizer,
#                                         milestones=[30, 80],
#                                         gamma=0.1)

total_optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9)
total_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=total_optimizer,
                                        milestones=[1, 5, 15, 40, 80],
                                        gamma=0.1)

criterion = nn.BCEWithLogitsLoss()

best_loss = 100
train_iter = len(train_loader)
valid_iter = len(valid_loader)

model = model.to(device)
model.train()

aug_class_list = [4, 9, 15, 16]

for e in range(EPOCH):
    print("epoch : "+str(e))
    train_loss = 0
    valid_loss = 0
    train_loss_class = []
    valid_loss_class = []
    

    for i, (images, targets) in tqdm(enumerate(train_loader), total=train_iter):
        total_optimizer.zero_grad()
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
            if(idx==0):
                train_total_loss = loss
            else:
                train_total_loss += loss

        train_total_loss.backward()
        total_optimizer.step()


    # for idx in aug_class_list:        
    #     for i, (images, targets) in tqdm(enumerate(train_hard_loader[idx]), total=len(train_hard_loader[idx])):
    #         total_optimizer.zero_grad()
    #         images = images.to(device)
    #         targets = targets.to(device)

    #         # forward
    #         class_targets = []
    #         for j in range(targets.shape[0]):
    #             li = []
    #             li.append(targets[j][idx])
    #             class_targets.append(li)
    #         class_targets = torch.tensor(class_targets).to(device)
            
    #         pred = model(images, idx)
    #         # loss
    #         loss = criterion(pred.double(), class_targets)
    #         train_loss += loss.item()

    #         loss.backward()
    #         total_optimizer.step()

    # total_train_loss = (train_loss / (20+4)) / train_iter

    total_train_loss = (train_loss / (20)) / train_iter
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

    total_valid_loss = (valid_loss /20) / valid_iter

    print("[train loss / %f] [valid loss / %f]" % (total_train_loss, total_valid_loss))

    if best_loss > total_valid_loss:
        best_loss = total_valid_loss
        print("model saved\n")
        torch.save(model.state_dict(), 'model.h5')
