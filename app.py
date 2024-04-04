from flask import Flask, render_template, request, jsonify

from time import time
import pickle
import os
import cv2
import tensorflow as tf
import numpy as np
import keras.utils as image
import json
from PIL import Image
import random
from werkzeug.utils import secure_filename
from flask.helpers import get_root_path
# Densenet:
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import datasets, models, transforms
num_classes = 5

app = Flask(__name__,template_folder='template')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

with open('model_RandomForest.pickle', 'rb') as f:
    random_forest_model = pickle.load(f)

with open('model_SVM.pickle', 'rb') as f:
    svm_model = pickle.load(f)

model1 = models.densenet201(pretrained=True)
num_ftrs = model1.classifier.in_features
model1.classifier = nn.Linear(num_ftrs, num_classes)
model1.load_state_dict(torch.load('densenet_ffe.pth')) # This line uses .load() to read a .pth file and load the network weights on to the architecture.
model1.eval() 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])


def predict():
     labels= {0: 'Ancient', 1: 'British', 2: 'Indoislamic', 3: 'Maratha', 4: 'Sikh'}

     if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

     file = request.files['image']

     if file.filename == '':
        return jsonify({'error': 'No selected file'})
        # Load and preprocess the uploaded image
        image1 = Image.open(file)
        preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust image size based on your model architecture
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on your model's requirements
            ])
        image1 = preprocess(image1).unsqueeze(0)  # Add batch dimension

    # Save the uploaded image
     filename = secure_filename(file.filename)
     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
     file.save(file_path)

     if file:
        # Load and preprocess the uploaded image
        image1 = Image.open(format(file_path))
        preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust image size based on your model architecture
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on your model's requirements
            ])
        image1 = preprocess(image1).unsqueeze(0)  # Add batch dimension
     custom_image = cv2.imread(file_path, 0)
     orb = cv2.ORB_create()
     kp, features = orb.detectAndCompute(custom_image, None)
     features = features.reshape(1, 500*32)

     preprocessed_image = np.reshape(features, (1, -1))
     random_forest_prediction = random_forest_model.predict(preprocessed_image)
     # Predict using the SVM model
     svm_prediction = svm_model.predict(preprocessed_image)

     with torch.no_grad():
        output1 = model1(image1)
        probabilities = torch.nn.functional.softmax(output1[0], dim=0)

    #  #img_path = 'C:\\Users\\Prakash\\Desktop\\internship\\Waste_Segregation\\plastic1.jpg'
    #  img = image.load_img(file_path, target_size=(32,32))
    #  img = image.img_to_array(img, dtype=np.uint8)
    #  img = np.array(img)/255.0
    #  model = tf.keras.models.load_model("model1.h5")
     
     prob1 = int(random_forest_prediction[0])
     prob1 = prob1*10 
     prob1 = round(prob1,2)
     prob1 = str(prob1) + '%'
     print("p.shape:",random_forest_prediction.shape)
     print("prob",prob1)
     

     predicted_class1 = labels[int(random_forest_prediction[0])]
     print("classified label:",predicted_class1)
     print("class:",int(random_forest_prediction[0]))

     prob2 = int(svm_prediction[0])
     prob2 = prob2*10
     prob2 = round(prob2,2)

     prob2 = str(prob2) + '%'
     print("p.shape:",svm_prediction.shape)
     print("prob",prob2)

    

     predicted_class2 = labels[int(svm_prediction[0])]
     print("classified label:",predicted_class2)
     print("class:",svm_prediction[0])



     probabilities = torch.nn.functional.softmax(output1[0], dim=0)
    

        # Get the predicted class index
     predicted_class3 = torch.argmax(probabilities).item()
     predicted_probability3 = probabilities[predicted_class3].item()
     print("classified label:",labels[int(predicted_class3)])
     print("prob",predicted_probability3*100)
     

     return jsonify({'prediction1': predicted_class1, 'probability1': f'{prob1}','prediction2': predicted_class2, 'probability2': f'{prob2}','prediction3': predicted_class3, 'probability3': predicted_class3})

if __name__ == '__main__':
    app.run(debug=True, port="2000")

    #on clicking upload, display image and call predict function, and display the output