import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


class ConvNetWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(54*54*16, 120)
        self.dropout1 = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.5)  
        self.fc3 = nn.Linear(84, 4)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54*54*16)
        X = F.relu(self.fc1(X))
        X = self.dropout1(X)  
        X = F.relu(self.fc2(X))
        X = self.dropout2(X)  
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 10, 4, 1)
        self.conv3 = nn.Conv2d(10, 16, 3, 1)
        self.fc1 = nn.Linear(26*26*16, 150)
        self.fc2 = nn.Linear(150, 80) 
        self.fc3 = nn.Linear(80, 4)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 26*26*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def print_model_parameters(model):
    params=[item.numel() for item in model.parameters()]
    print(params)
    print("\nTotal parameters - ",sum(params))
    print("\n")

def reset_model_from_GPU():
    gc.collect()
    torch.cuda.empty_cache()

def save_model(model,file_name):
    torch.save(model,file_name)