import cv2
from sklearn.externals import joblib
from sklearn.svm import SVC
import numpy as np

count = 0
clf = joblib.load("alphabet_classifier.pkl")
#clf = cv2.ml.SVM_load("alphabet_classifier2.xml")
for i in range(122):
    img = cv2.imread("Testing/"+str(i+1)+".jpg")
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    retval, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = cv2.dilate(img, (3, 3))
    img = img.reshape(1, -1)
    prediction = clf.predict(img)
    print("For "+str(i+1)+".jpg:")
    if i+1 <= 88:
        if prediction == 0:
            count = count + 1
    if i+1 >= 89:
        if prediction == 2:
            count = count + 1
    print(prediction)


print("Accuracy: ")
print(count/122)