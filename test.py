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

# MODEL_PATH = 'model.h5'
MODEL_PATH = "/content/drive/MyDrive/URP/models/model.h5"
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
model = model.to(device)
model = model.eval()

# tensor image generate
images = test_transformer(Image.open('cat.jpg')).view(1, 3, 224, 224)
images = images.to(device)

# prediction
pred = model(images)

pred_sigmoid = torch.sigmoid(pred).cpu()
print(pred_sigmoid)
pred_rounded = np.round(pred_sigmoid)
print(pred_rounded)

print(VOC_CLASSES[pred.argmax()])
print(pred)
