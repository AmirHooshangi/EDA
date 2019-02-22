import tensorflow as tf
from collections import Counter

class Ensemble:
    def __init__(self, ensemble_size, num_class):
        self.num_class = num_class
        self.ensemble_size = ensemble_size

    ensemble_members = []
    feature_mappers = []

    def multy_class_nn_model(self, in_data):
        bn = tf.layers.batch_normalization(in_data)
        fc1 = tf.layers.dense(bn, 512, activation=tf.nn.relu)
        fc21 = tf.layers.dropout(fc1)
        fc11 = tf.layers.dense(fc21, 512, activation=tf.nn.relu)
        fc2 = tf.layers.dropout(fc11)
        fc3 = tf.layers.dense(fc2, self.num_class)
        return fc3

    def two_class_nn_model(self, in_data):
        bn = tf.layers.batch_normalization(in_data)
        fc1 = tf.layers.dense(bn, 128, activation=tf.nn.relu)
        fc11 = tf.layers.dense(fc1, 128, activation=tf.nn.relu)
        fc2 = tf.layers.dropout(fc11)
        fc3 = tf.layers.dense(fc2, 1, activation=None)
        return fc3

    def addMutlyClassModel(self, X, Y):
        ensemble_member = []
        #next_element = X.get_next()
        logits = self.multy_class_nn_model(X)
        # add the optimizer and loss
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits))
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        # get accuracy
        prediction = tf.argmax(logits, 1)
        equality = tf.equal(prediction, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        ensemble_member.append([logits, loss, optimizer, prediction, equality, accuracy])
        self.ensemble_members.append(ensemble_member)
        return self.ensemble_members

    def addTwoClassModel(self, X, Y):
        ensemble_member = []
        # next_element = X.get_next()
        logits = self.two_class_nn_model(X)
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits))
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        # get accuracy
        prediction = tf.nn.sigmoid(logits)
        equality = tf.equal(tf.round(prediction), Y)
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        ensemble_member.append([logits, loss, optimizer, prediction, equality, accuracy])
        self.ensemble_members.append(ensemble_member)
        return self.ensemble_members

    def addModel(self, X, Y, problemType):
        if problemType == "multyclass":
            return self.addMutlyClassModel(X, Y)
        elif problemType == "twoclass":
            return self.addTwoClassModel(X, Y)
        else:
            raise ValueError("No Valid Ensemble problem class implementation")


    def get_ensemble_members(self):
            return self.ensemble_members

    def set_feature_mappers(self, feature_mappers):
        self.feature_mappers = feature_mappers

    def get_feature_mappers(self):
        return self.feature_mappers
