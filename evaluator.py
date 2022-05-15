import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from datasets.loader import VOC

class MultiLabelClassificationEvalulator():
    def __init__(self, model_path):
        ctx = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(ctx)

        model = models.vgg16().to(self.device)
        model.classifier[6] = nn.Linear(4096, 20)
        model.load_state_dict(torch.load(model_path))
        model = model.to(self.device)
        self.model = model.eval()        

    def evaluate(self, batch_size):
        VOC_CLASSES = (
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        )
        test_transformer = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])

        voc = VOC(batch_size=batch_size, year="2007")
        test_loader = voc.get_loader(transformer=test_transformer, datatype='test')

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                outputs = torch.sigmoid(outputs).cpu()
                predicted = torch.round(outputs)
                
                total += labels.size(0)*labels.size(1)
                print(predicted.size())
                print(predicted)
                print("   ")
                print(labels.size())
                print(labels)
                correct += (predicted==labels).sum().item()
                print(correct)

        accuracy = 100*correct/total
        print("Accuracy: {}%".format(accuracy))

