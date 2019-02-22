from sklearn import svm
import pickle
import numpy as np
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

number_of_autoencoder_modules = 1
dataset_name = "convex"
classtype = "twoclass"
classifiers = []

def noramalize_dataset(dataset):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(dataset)

def flatten_labels(labels):
    final_labels = []
    if classtype == "twoclass":
        return labels.flatten()
    elif classtype == "multyclass":
        for i in labels:
            final_labels.append(np.argmax(i))
    return final_labels


#gamma_range = np.logspace(-9,10,13)
#c_range = np.logspace(-2,10,13)

for i in range(number_of_autoencoder_modules):
    train_x, train_y = pickle.load(open(dataset_name+str(i)+"train.pcl", "rb"))
    flattened_train_labels = flatten_labels(train_y)
 #   param_grid = dict(gamma=gamma_range, C=c_range)
 #   cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
 #   grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
    clf = svm.SVC(gamma=0.021, C=100)
    if classtype == "twoclass":
        clf.fit(noramalize_dataset(train_x), train_y)
    else:
        clf.fit(noramalize_dataset(train_x), flattened_train_labels)
    classifiers.append(clf)


def final_votes(votes):
    final_votes = []
    for column in votes.T:
        most_common, num_most_common = Counter(column).most_common(1)[0]
        final_votes.append(most_common)
    return final_votes


def majority_voting():
    votes = np.empty((0, 50000), int)
    for i in range(number_of_autoencoder_modules):
        test_x, test_y = pickle.load(open(dataset_name + str(i) + "test.pcl", "rb"))
        prediction = classifiers[i].predict(noramalize_dataset(test_x))
        if classtype == "twoclass":
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
