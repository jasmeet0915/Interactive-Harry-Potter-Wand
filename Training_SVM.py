import pandas as pd
from pandas import DataFrame
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("....Reading DataSet and Creating Pandas DataFrame....")
alphabet_data = pd.read_csv("/home/jasmeet/PycharmProjects/Character_Recognition/SVM_WithFoundDataset/A_ZHandwrittenData.csv")
print("...DataFrame Created...")


print("...Slicing and creating initial training and testing set...")
# Dataset of letter A containg features
X_Train_A = alphabet_data.iloc[:13869, 1:]
# Dataset of letter A containg labels
Y_Train_A = alphabet_data.iloc[:13869, 0]
# Dataset of letter C containg features
X_Train_C = alphabet_data.iloc[22537:45946, 1:]
# Dataset of letter C containg labels
Y_Train_C = alphabet_data.iloc[22537:45946, 0]
# Joining the Datasets of both letters
X_Train = pd.concat([X_Train_A, X_Train_C], ignore_index=True)
Y_Train = pd.concat([Y_Train_A, Y_Train_C], ignore_index=True)
print("...X_Train and Y_Train created...")



train_features, test_features, train_labels, test_labels = train_test_split(X_Train, Y_Train, test_size=0.25, random_state=0)
print(type(test_features))

# SVM classifier created
clf = SVC(kernel='linear')
print("")
print("...Training the Model...")
clf.fit(train_features, train_labels)
print("...Model Trained...")


labels_predicted = clf.predict(test_features)
print(test_labels)
print(labels_predicted)
accuracy = accuracy_score(test_labels, labels_predicted)

print("")
print("Accuracy of the model is:  ")
print(accuracy)

print("...Saving the trained model...")
joblib.dump(clf, "alphabet_classifier.xml", compress=3)
print("...Model Saved...")