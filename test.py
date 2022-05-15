import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
import time

from PIL import Image
from datasets.loader import VOC

from helper_functions import mAP, AverageMeter, CocoDetection

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

#ap=np.empty((0,20), float)

batch_time = AverageMeter()
prec = AverageMeter()
rec = AverageMeter()
mAP_meter = AverageMeter()

tp, fp, fn, tn, count = 0, 0, 0, 0, 0
preds = []
targets = []

end = time.time()

with torch.no_grad():
    for input, target in test_loader:
        input = input.to(device)
        output = model(input)
        output = torch.sigmoid(output).cpu()

        target = target
        #target = target.max(dim=1)[0]

        # for mAP calculation
        preds.append(output.cpu())
        targets.append(target.cpu())

        # measure accuracy and record loss
        pred = output.data.gt(0.5).long()
        #pred = torch.round(output)

        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        count += input.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()

        this_prec = this_tp.float() / (
            this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (
            this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        prec.update(float(this_prec), input.size(0))
        rec.update(float(this_rec), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for
               i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)

'''
print(
        '--------------------------------------------------------------------')
print(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
      .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))'''

mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
print("mAP score:", mAP_score)