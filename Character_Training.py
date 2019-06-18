import cv2
import numpy as np
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.svm import SVC

training_samples = []
# 1 for keyhole and 0 for C
training_labels = []
hog_training_samples = []


def append_keyholes(i):
    img = cv2.imread("Character_Keyhole_Samples/"+str(i+1)+".jpg", 0)
    img = cv2.resize(img, (28, 28))
    training_samples.append(img)
    training_labels.append(1)


def append_c(i):
    img2 = cv2.imread("Character_C_Samples/"+str(i+1)+".jpg", 0)
    img2 = cv2.resize(img2, (28, 28))
    training_samples.append(img2)
    training_labels.append(0)


for x in range(31):
    append_keyholes(x)
    if x < 30:
        append_c(x)

training_samples = np.array(training_samples, 'int16')
training_labels = np.array(training_labels, 'int')

for sample in training_samples:
    feature = hog(sample, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    hog_training_samples.append(feature)

hog_training_samples = np.array(hog_training_samples, 'float64')

clf = SVC()
clf.fit(hog_training_samples, training_labels)
joblib.dump(clf, "characters_clf.pkl", compress=3)


print(hog_training_samples)
print(training_samples)
print(training_labels)