import os
from tkinter import image_names
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torchvision import models
import torchvision.transforms as transforms
from datasets.loader import VOC

class MultiLabelClassficationTrainer():
    def __init__(self):
        ctx = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(ctx)

        model = models.vgg16(pretrained=True).to(self.device)
        model.classifier[6] = nn.Linear(4096, 20)
        for i, (name, param) in enumerate(model.features.named_parameters()):
            param.requires_grad = False
        self.model = model

    def train(self, batch_size, epoch, model_name):
        VOC_CLASSES = (
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        )

        #train data class num
        class_num_arr = np.array([113, 122, 182, 87, 153, 100, 402, 166, 282, 71, 130, 210, 144, 123, 1070, 153, 49, 188, 128, 144], dtype=float)
        class_num_arr = np.reciprocal(class_num_arr)
        class_weight= torch.tensor(class_num_arr).to(self.device)

        train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.Resize((224, 224)),
                                                transforms.ToTensor(),])

        valid_transformer = transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.ToTensor(),])

        voc = VOC(batch_size=batch_size, year="2007")
        train_loader = voc.get_loader(transformer=train_transformer, datatype='train')
        valid_loader = voc.get_loader(transformer=valid_transformer, datatype='val')

        optimizer = optim.SGD(self.model.classifier.parameters(), lr=0.001, weight_decay=1e-5, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                milestones=[50, 100, 150],
                                                gamma=0.1)

        criterion_train = nn.BCEWithLogitsLoss(reduction='none')
        criterion_valid = nn.BCEWithLogitsLoss()

        best_loss = 100
        train_iter = len(train_loader)
        valid_iter = len(valid_loader)

        for e in range(epoch):
            train_loss = 0
            valid_loss = 0

            scheduler.step()

            for i, (images, targets) in tqdm(enumerate(train_loader), total=train_iter):
                images = images.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                # forward
                self.model = self.model.to(self.device)
                pred = self.model(images)
                # loss
                loss = criterion_train(pred.double(), targets)
                loss = loss*class_weight
                loss = loss.mean()
                train_loss += loss.item()
                # backward
                loss.backward(retain_graph=True)
                # weight update
                optimizer.step()

            total_train_loss = train_loss / train_iter

            with torch.no_grad():
                for images, targets in valid_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    pred = self.model(images)
                    # loss
                    loss = criterion_valid(pred.double(), targets)
                    valid_loss += loss.item()

            total_valid_loss = valid_loss / valid_iter

            print("epoch : " + str(e))
            print("[train loss / %f] [valid loss / %f]" % (total_train_loss, total_valid_loss))

            if best_loss > total_valid_loss:
                print("model saved")
                torch.save(self.model.state_dict(), model_name)
                best_loss = total_valid_loss

