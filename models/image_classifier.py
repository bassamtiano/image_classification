import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from torchmetrics.classification import MulticlassF1Score, PrecisionRecallCurve

class ImageClassifier(pl.LightningModule):
    
    def __init__(self, lr, num_classes) -> None:
        super(ImageClassifier, self).__init__()
        
        self.lr = lr
        
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.f1_metrics_micro = MulticlassF1Score(task = "multiclass", 
                                                  average = "micro",
                                                  num_classes = num_classes)
        
        self.f1_metrics_macro = MulticlassF1Score(task = "multiclass", 
                                                  average = "macro",
                                                  num_classes = num_classes)
        
        self.pr_curve = PrecisionRecallCurve(task = "multiclass", num_classes = num_classes)
        
        self.train_scores = {
            "epoch": [],
            "loss": [],
            "f1_micro": [],
            "f1_macro": [],
            "f1_weighted": [],
            "accuracy": [],
            "precission": [],
            "recall": []
        }
        
        self.val_scores = {
            "epoch": [],
            "loss": [],
            "f1_micro": [],
            "f1_macro": [],
            "f1_weighted": [],
            "accuracy": [],
            "precission": [],
            "recall": []
        }


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr = self.lr, momentum=0.9)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        out = self(x)
        # pred = out.argmax(1)

        loss = self.criterion(out, y)
        
        f1_mic = self.f1_metrics_micro(out.argmax(1), y)
        f1_mac = self.f1_metrics_macro(out.argmax(1), y)
        
        self.log_dict({"loss": loss,
                       "f1_micro": f1_mic,
                       "f1_macro": f1_mac
                      }, prog_bar = True)
        
        return {"loss": loss, "f1_micro": f1_mic, "f1_macro": f1_mac}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        out = self(x)
        loss = self.criterion(out, y)
        
        f1_mic = self.f1_metrics_micro(out.argmax(1), y)
        f1_mac = self.f1_metrics_macro(out.argmax(1), y)
        
        self.log_dict({"loss": loss,
                       "f1_micro": f1_mic,
                       "f1_macro": f1_mac
                      }, prog_bar = True)
        
        return {"loss": loss, "f1_micro": f1_mic, "f1_macro": f1_mac}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        out = self(x)
        
        loss = self.criterion(out, y)
        
        f1_mic = self.f1_metrics_micro(out.argmax(1), y)
        f1_mac = self.f1_metrics_macro(out.argmax(1), y)
        
        self.log_dict({"loss": loss,
                       "f1_micro": f1_mic,
                       "f1_macro": f1_mac
                      }, prog_bar = True)
        
        return {"pred": out.argmax(1), "true": y}
        
    