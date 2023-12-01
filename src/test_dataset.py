import models
import torch
from PIL import Image
from torchvision import transforms
import os

labels=['angry', 'bored', 'focused', 'neutral']

def get_test_results(model,test_loader):
    y_pred = []
    y_true = []
    # iterate over test data
    for inputs, labels in test_loader:
            output = model(inputs) # Feed Network
    
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth
    return [y_true,y_pred]

def predict_emotion(device,image_relative_path,model):
       # Download the image
    img_root = os.path.abspath(image_relative_path)

    # response = requests.get(img_root, stream=True)
    img = Image.open(img_root)
    
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),
    ])
    
    img = transform(img)
    
    # Add an extra dimension to represent the batch (1 image in this case)
    img = img.unsqueeze(0)
    img=  img.to(device)
    with torch.no_grad():
         y = model(img)
    print(y)
    print(labels)
    predicted_class = torch.argmax(y).item()
    
    print("\n Predicted - ",labels[predicted_class],"\n")