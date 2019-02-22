from sklearn.neighbors import NearestNeighbors
import pickle
import numpy as np
from collections import Counter
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf

number_of_autoencoder_modules = 2
dataset_name = "mnist"
classifiers = []
k = 1

def noramalize_dataset(dataset):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(dataset)


def flatten_labels(labels):
    final_labels = []
    if dataset_name == "convex" or dataset_name == "rectangles":
        return labels.flatten()
    for i in labels:
        final_labels.append(np.argmax(i))
    return final_labels

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

print("Fitting points started")
for i in range(number_of_autoencoder_modules):
    test_x, test_y = pickle.load(open(dataset_name+str(i)+"train.pcl", "rb"))
    one_dimensional_features = session.run(tf.reshape(test_x, shape=[1, 196]))
    nbrs = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, p=2)
    nbrs.fit(noramalize_dataset(one_dimensional_features), flatten_labels(test_y))
    classifiers.append(nbrs)


def final_votes(votes):
    final_votes = []
    for column in np.array(votes).T.tolist():
        most_common, num_most_common = Counter(column).most_common(1)[0]
        final_votes.append(most_common)
    return final_votes


def majority_voting():
    votes = np.empty((0, 50000), int)
    for i in range(number_of_autoencoder_modules):
        test_x, test_y = pickle.load(open(dataset_name + str(i) + "test.pcl", "rb"))
        one_dimensional_features = session.run(tf.reshape(test_x, shape=[1, 196]))
        prediction = classifiers[i].predict(noramalize_dataset(one_dimensional_features))
        if dataset_name == "convex" or dataset_name == "rectangles":
            prediction = prediction.flatten()
            prediction = np.round(prediction)
            votes = np.append(votes, [prediction], axis=0)
        else:
            votes = np.append(votes, [prediction], axis=0)
    votes = final_votes(votes)
    return votes


def calculate_accuracy(votes, labels):
    voting_accuracy = 0
    for i in range(50000):
        if votes[i] == labels[i]:
            voting_accuracy += 1
    return voting_accuracy

_ , labels = pickle.load(open(dataset_name + str(0) + "test.pcl", "rb"))

votes = majority_voting()
accuracy = calculate_accuracy(votes, flatten_labels(labels))
print("final accuracy is ", accuracy)