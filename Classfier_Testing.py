import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
from sklearn.metrics import accuracy_score

clf = joblib.load("characters_clf.pkl")
labels_predicted = []
hog_testing_images = []
testing_images = []
testing_labels = [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0]


def load_testing_images():
    for y in range(70):
        img = cv2.imread("Training_Images/"+str(y+1)+".jpg", 0)
        img = cv2.resize(img, (100, 100))
        testing_images.append(img)


def create_hog_list():
    for image in testing_images:
        feature = hog(image, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        hog_testing_images.append(feature)


load_testing_images()
create_hog_list()

hog_testing_images = np.array(hog_testing_images, 'float64')
testing_labels = np.array(testing_labels, 'int')


labels_predicted = clf.predict(hog_testing_images)

print(labels_predicted)
accuracy = accuracy_score(testing_labels, labels_predicted)
print(accuracy)