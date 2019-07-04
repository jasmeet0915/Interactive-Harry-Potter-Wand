from PIL import Image
from sklearn.externals import joblib
import numpy as np

# Loading the processed last frame form Desktop
img = Image.open("/home/pi/Desktop/lastframe.jpg")

# Loading the SVM classifier
clf = joblib.load("/home/pi/Desktop/alphabet_classifier.pkl")

# Converting image to numpy array
img = np.array(img)
# Converting the numpy array to 1-Dimensional array
img = img.reshape(1, -1)


prediction = clf.predict(img)
print(prediction)
