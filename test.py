import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from datasets.loader import VOC

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

MODEL_PATH = '/content/drive/MyDrive/URP/models/model.h5'
BATCH_SIZE = 32

# test dataset
test_transformer = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor()])

voc = VOC(batch_size=BATCH_SIZE, year="2007")
test_loader = voc.get_loader(transformer=test_transformer, datatype='test')

# load model
model = models.vgg16().to(device)
model.classifier[6] = nn.Linear(4096, 20)

# load weight
model.load_state_dict(torch.load(MODEL_PATH))

# model eval
model = model.eval()

# tensor image generate
images = test_transformer(Image.open('cat.jpg')).view(1, 3, 224, 224)
images = images.to(device)

# prediction
model=model.to(device)
pred = model(images)
pred_sigmoid = torch.sigmoid(pred)
pred_rounded = torch.round(pred_sigmoid)
tmp=pred_rounded.cpu().detach().numpy()[0]

for i in range(20):
  if tmp[i]==1:
    print(VOC_CLASSES[i]) 

  
#Accuracy===================================
'''
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        outputs = torch.sigmoid(outputs).cpu()
        predicted = torch.round(outputs)
        total += labels.size(1)*32
        correct += (predicted==labels).sum().item()
        # print(correct)

accuracy = 100*correct/total
print("Accuracy: {}%".format(accuracy))'''

#mAP============================
from sklearn.metrics import average_precision_score, precision_recall_curve
np.seterr(invalid='ignore')

ap=np.empty((0,20), float)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        outputs = torch.sigmoid(outputs).cpu()
        a = average_precision_score(labels_np, outputs_np, average=None)
        ap = np.append(ap, [a], axis=0)
        #print(a)

print("\nMAP=\n")
mAP=ap.mean(axis=0)
print(mAP)