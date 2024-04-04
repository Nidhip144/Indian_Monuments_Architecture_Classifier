import pickle
import os
import numpy
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

class model:
    def __init__(self):
        print("initiated")

    def preprocess_data(self, path):
        # Your preprocessing code here
        files = os.listdir(path)
        orb = cv2.ORB_create()
        #sift = cv2.xfeatures2d.SIFT_create()
        X=numpy.zeros(shape=(1,500*32))
        Y=numpy.zeros(shape=(1,))
        count=-1
        for f in files:
             if not (f.startswith('.')):
                folder = os.path.join(path, f)
                filenames = os.listdir(folder)
                count+=1
                for fi in filenames:
                    if not (fi.startswith('.')):
                        filepath = os.path.join(folder, fi)
                        #print filepath
                        image = cv2.imread(filepath, 0)
                        kp, features = orb.detectAndCompute(image,None)
                        #print features.shape
                        features=features.reshape(1,500*32)
                        #print features.shape
                        X = numpy.vstack((X, features))
                        Y=numpy.vstack((Y,numpy.asarray([count])))
        #print X.shape
        Y=Y.reshape(Y.shape[0],)
        return X,Y

    def model_randomforest(self, X, Y):
        print("printing data shape {}".format(X.shape))

        # Splitting the data into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

        # Training the Random Forest Classifier
        K = 1000
        clf = RandomForestClassifier(n_estimators=K)
        clf.fit(X_train, Y_train)

        # Evaluating the model
        #y_pred = clf.predict(X_test)
        #print("Accuracy on test set: {}".format(accuracy_score(Y_test, y_pred)))
        # Predicting for custom input file
        custom_input_file = "C:/Users/Prakash/Desktop/mon/Architecture_Classification/Data/Ancient/Ajanta--Ellora-6095_1.jpg"  # Change this to your custom input file path
        custom_image = cv2.imread(custom_input_file, 0)
        orb = cv2.ORB_create()
        kp, features = orb.detectAndCompute(custom_image, None)
        features = features.reshape(1, 500*32)
        # Make prediction 
        predicted_label = clf.predict(features)
        print("Predicted label for custom input file: {}".format(predicted_label))


    def model_svm(self,X,y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        C_score = []
        grid = numpy.arange(0.01, 3, 0.1)
        for K in grid:
            clf = SVC(C=K)
            # Calculate the mean scores for each value of hyperparameter C
            scores = cross_val_score(clf, X_train, Y_train, cv=5)
            print(scores.mean())
            C_score.append(scores.mean())

        # Display the maximum score achieved at which hyperparameter value
        print (" max score is ", max(C_score), " at C = ", grid[C_score.index(max(C_score))])
        clf = SVC(C=grid[C_score.index(max(C_score))])
        clf.fit(X_train, Y_train)

        # Predicting for custom input file
        custom_input_file = "C:/Users/Prakash/Desktop/mon/Architecture_Classification/Data/Ancient/Ajanta--Ellora-6095_1.jpg"  # Change this to your custom input file path
        custom_image = cv2.imread(custom_input_file, 0)
        orb = cv2.ORB_create()
        kp, features = orb.detectAndCompute(custom_image, None)
        features = features.reshape(1, 500*32)
        # Make prediction 
        predicted_label = clf.predict(features)
        print("Predicted label for custom input file: {}".format(predicted_label))

        # y_pred = clf.predict(X_test)
        # print("accuracy is {}".format(accuracy_score(Y_test, y_pred)))

# Create an instance of your model
m = model()
# Preprocess your data
X, Y = m.preprocess_data("./../Data")
# Train and evaluate your model
m.model_randomforest(X, Y)
m.model_svm(X,Y)

