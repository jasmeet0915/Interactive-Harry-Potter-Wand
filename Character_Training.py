import cv2
import numpy as np
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

training_samples = []
# 1 for keyhole and 0 for C
training_labels = [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0]
hog_training_samples = []


def load_training_images():
    for i in range(70):
        img = cv2.imread("Training_Images/"+str(i+1)+".jpg", 0)
        img = cv2.resize(img, (30, 30))
        training_samples.append(np.array(img))


def create_hog_features():
    for sample in training_samples:
        feature = hog(sample, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        hog_training_samples.append(feature)


load_training_images()
create_hog_features()
hog_training_samples = np.array(hog_training_samples, 'float64')

# training
clf = LinearSVC()
clf.fit(hog_training_samples, training_labels)
joblib.dump(clf, "characters_clf.pkl", compress=3)


print(hog_training_samples)
print(training_samples)
print(training_labels)