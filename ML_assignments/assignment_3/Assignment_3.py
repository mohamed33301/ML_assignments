from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import cv2
import numpy as np
import os


X_train = []
y_train = []


def extract_features(directory):
    X = []
    y = []
    for label in os.listdir(directory):
        for file in os.listdir(os.path.join(directory, label)):
            img = cv2.imread(os.path.join(directory, label, file), 0)
            img = cv2.resize(img, (128,64)) # Resize image to fixed size
            X.append(hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=False)) # Extract HOG features
            y.append(label)
    return np.array(X), np.array(y)



X_train, y_train = extract_features("D:\\ass_ml_3_2024\\Assignment dataset\\train")
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = []
y_test = []

X_test, y_test = extract_features("D:\\ass_ml_3_2024\\Assignment dataset\\test")

X_test = np.array(X_test)
y_test = np.array(y_test)

# Train 
svm_model = svm.SVC(kernel='linear', C=0.1)
svm_model.fit(X_train, y_train)

#  model
y_pred_train = svm_model.predict(X_train)
y_pred = svm_model.predict(X_test)

accuracy_train = metrics.accuracy_score(y_train, y_pred_train)
accuracy = metrics.accuracy_score(y_test, y_pred)

precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

print(f"\nAccuracy_train = {accuracy_train}")
print(f"\nAccuracy_test = {accuracy}")