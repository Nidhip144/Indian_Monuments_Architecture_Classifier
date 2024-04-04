import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import datasets, models, transforms
num_classes = 5
labels= {0: 'Ancient', 1: 'British', 2: 'Indoislamic', 3: 'Maratha', 4: 'Sikh'}
model = models.densenet201(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('densenet_ffe.pth')) # This line uses .load() to read a .pth file and load the network weights on to the architecture.
model.eval() 

# Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust image size based on your model architecture
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on your model's requirements
    ])
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image

# Load and preprocess the custom image
image_path = 'C:/Users/Prakash/Desktop/mon/aiml_proj/Data/Sikh/005.jpg'
input_image = preprocess_image(image_path)

# Make prediction
with torch.no_grad():
    output = model(input_image)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get predicted class
predicted_class = torch.argmax(probabilities).item()
print(f'Predicted class: {labels[int(predicted_class)]}')