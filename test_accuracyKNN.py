# K-Nearest Neighbours ML Model

import numpy as np
import pickle
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def test_accuracy(medicine):
    print(medicine)

    # Acquire Data
    pickle_dump = open('./pickle/' + medicine, "rb")
    data = pickle.load(pickle_dump)
    print("Acquired Training Data / Collection.")

    # Data Pre-processing
    le = LabelEncoder()
    data['result'] = le.fit_transform(data['result'])
    data['disease'] = le.fit_transform(data['disease'])
    data['medicine'] = le.fit_transform(data['medicine'])
    print("Data Pre-processing.")

    # Setting Features and Labels
    print("Setting up Features & Labels.")
    X = np.array(data.drop(['result'], 1))
    Y = np.array(data['result'])
    print("Setting Labels & Features.")

    # Train Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    print("Train-Test Split.")

    # Fitting Classifier
    classifier = neighbors.KNeighborsClassifier()
    classifier.fit(X_train, Y_train)
    print("Fitting Classifier.")

    # Accuracy Testing
    accuracy = classifier.score(X_test, Y_test)
    print(accuracy)
    print(data.shape)
    print("Accuracy Test.")

    return accuracy
