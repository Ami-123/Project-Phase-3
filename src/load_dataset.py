import torch
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, transforms
import os

transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

def load_dataset_and_process(dataset_path='data/'):
    torch.manual_seed(5)

    dataset_root = os.path.abspath(dataset_path)
    dataset = datasets.ImageFolder(root=dataset_root, transform=transform)
    return dataset

def load_entire_dataset(device,dataset_path='data/',BATCH_SIZE=16):
    torch.manual_seed(6)
      
    dataset_root = os.path.abspath(dataset_path)
    
    # Create a custom dataset
    dataset = datasets.ImageFolder(root=dataset_root, transform=transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    data_loader = [(X.to(device), y.to(device)) for X, y in data_loader]

    return data_loader
    
def load_saved_model(device,model_path):
    model = torch.load(model_path) #'/home/umang/COMP6721_AK6/ConvNet2.pth'
    model.to(device)
    return model
    
def display_model(model):
    print(model)