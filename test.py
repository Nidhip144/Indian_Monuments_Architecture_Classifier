import pickle
import cv2
import numpy as np
labels= {0: 'Ancient', 1: 'British', 2: 'Indoislamic', 3: 'Maratha', 4: 'Sikh'}

# Load the saved models
with open('model_RandomForest.pickle', 'rb') as f:
    random_forest_model = pickle.load(f)

with open('model_SVM.pickle', 'rb') as f:
    svm_model = pickle.load(f)

# Define a function to preprocess the input image
def preprocess_image(image_path):
    # Read the image
    custom_image = cv2.imread(image_path, 0)
    orb = cv2.ORB_create()
    kp, features = orb.detectAndCompute(custom_image, None)
    features = features.reshape(1, 500*32)
    return features

# Define a function to predict the monument type of an input image
def predict_monument_type(image_path):
    # Preprocess the input image
    preprocessed_image = preprocess_image(image_path)
    # Reshape the preprocessed image to match the input shape expected by the models
    preprocessed_image = np.reshape(preprocessed_image, (1, -1))
    # Predict using the Random Forest model
    random_forest_prediction = random_forest_model.predict(preprocessed_image)
    # Predict using the SVM model
    svm_prediction = svm_model.predict(preprocessed_image)
    # Return the predictions
    return random_forest_prediction, svm_prediction

# Path to the input image
image_path = 'C:/Users/Prakash/Desktop/mon/aiml_proj/Data/Sikh/005.jpg'
# Predict the monument type of the input image
random_forest_prediction, svm_prediction = predict_monument_type(image_path)
# Print the predictions
print("Random Forest prediction:", labels[int(random_forest_prediction)])
prob1 = int(random_forest_prediction[0])
prob1 = prob1*10 
prob1 = round(prob1,2)
prob1 = str(prob1) + '%'
print("prob",prob1)

print("SVM prediction:", labels[int(svm_prediction)])
prob2 = int(svm_prediction[0])
prob2 = prob2*10
prob2 = round(prob2,2)

prob2 = str(prob2) + '%'
print("prob",prob2)