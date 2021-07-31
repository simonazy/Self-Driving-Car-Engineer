import numpy as np
from datetime import datetime

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms

import matplotlib.pyplot as plt 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class LeNet(nn.Module):
    def __init__(self, n_classes):
        super(LeNet,self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5,stride=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=16,out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tahn(),
            nn.Linear(in_features=84, out_features=n_classes)
        )
    
    def forward(self,x):
        x = self.feature_extractor(x)
        x = torch.flatten(x,1)
        logits = self.classifier(x)
        prob = F.softmax(logits, dim=1)
        return logits, prob

# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 15

IMG_SIZE = 32
N_CLASSES = 10


def train(train_loader, model, loss_fn, optimizer, device):
    model.train()
    running_loss = 0
    for X, y_true in train_loader:
        optimizer.zero_grad()
        
        X = X.to(device)
        y_true = y_true.to(device)
        
        y_hat,_ = model(X)
        loss = loss_fn(y_hat, y_true)
        running_loss += loss.item()*X.size(0)
        loss.backward()
        
        optimizer.step()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss

def validate(valid_loader, model, criterion, device):

    model.eval()
    running_loss = 0
    
    for X, y_true in valid_loader:
    
        X = X.to(device)
        y_true = y_true.to(device)

        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
    return model, epoch_loss