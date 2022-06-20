import os
from tkinter import image_names
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model_origin_resnet import resnet34
import torchvision.transforms as transforms
from datasets.loader_custom_v3 import VOC

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)
MODEL_PATH = 'model_origin.h5'
BATCH_SIZE = 16
EPOCH = 100

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

# augmentation
voc = VOC(batch_size=BATCH_SIZE, year1="2007")

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
    
# load model
model = resnet34(pretrained=True).to(device)

# Momentum / L2 panalty
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                           milestones=[30, 80],
                                           gamma=0.1)
criterion = nn.BCEWithLogitsLoss()

best_loss = 100
train_iter = len(train_loader)
valid_iter = len(valid_loader)


aug_class_list = [4, 15, 17]

for e in range(EPOCH):
    print("epoch : "+str(e))
    train_loss = 0
    valid_loss = 0

    for i, (images, targets) in tqdm(enumerate(train_loader), total=train_iter):
        images = images.to(device)
        targets = targets.to(device)

        model = model.to(device)
        pred = model(images)
        loss = criterion(pred.double(), targets)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    total_train_loss = train_loss / train_iter
    scheduler.step()

    with torch.no_grad():
        for images, targets in valid_loader:
            images = images.to(device)
            targets = targets.to(device)

            pred = model(images)
            loss = criterion(pred.double(), targets)
            valid_loss += loss.item()

    total_valid_loss = valid_loss / valid_iter

    print("[train loss / %f] [valid loss / %f]" % (total_train_loss, total_valid_loss))

    if best_loss > total_valid_loss:
        print("model saved\n")
        torch.save(model.state_dict(), 'model_origin.h5')
        best_loss = total_valid_loss
