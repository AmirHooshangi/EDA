from __future__ import division, print_function, absolute_import

from tensorflow.contrib import autograph
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import eda_cae_autoEncoder
import time

import ensemble

###############################################################################
dataset_name = "mnist"

datasets_metadate = {"mnist": {"problem_type": "multyclass", "input_size": 784, "num_classes": 10, "train_size": 12000, "test_size": 50000}
                    ,"mnist-random-background": {"problem_type": "multyclass", "input_size": 784, "num_classes": 10, "train_size": 12000, "test_size": 50000}
                    ,"mnist-rotation": {"problem_type": "multyclass", "input_size": 784, "num_classes": 10, "train_size": 12000, "test_size": 50000}
                    ,"rectangles": {"problem_type": "twoclass", "input_size": 784, "num_classes":1, "train_size": 12000, "test_size": 50000}
                    ,"mnist-background": {"problem_type": "multyclass", "input_size": 784, "num_classes": 10, "train_size": 12000, "test_size": 50000}
                    ,"convex": {"problem_type": "twoclass", "input_size": 784, "num_classes": 1, "train_size": 8000, "test_size": 50000}
                    ,"NUS-WIDE": {"problem_type": "twoclass", "input_size": 784, "num_classes": 1, "train_size": 1,"test_size": 1}}

save_datasets_flag = True
# Training Parameters
learning_rate = 0.01
number_of_autoencoder_modules = 2
autoencoder_num_steps = 1
batch_size = 256
display_step = 1000
examples_to_show = 10
# Network Parameters
hidden_layer_size = 40 # 1st layer num features
input_size = datasets_metadate[dataset_name]["input_size"] # MNIST data input (img shape: 28*28)
# tf Graph input (only pictures)
num_classes = datasets_metadate[dataset_name]["num_classes"]
classtype = datasets_metadate[dataset_name]["problem_type"]
total_train_size = datasets_metadate[dataset_name]["train_size"]
total_test_size = datasets_metadate[dataset_name]["test_size"]
input_x = tf.placeholder("float", [None, input_size])
keepprob = tf.placeholder(tf.float32)
dropout_value = -1
ae_first_layer_hidden_layer_size = 16
#############################################################################



modular_AutoEncoder = eda_cae_autoEncoder.EdaCaeAutoEncoder(number_of_autoencoder_modules, input_size
                                                            , ae_first_layer_hidden_layer_size
                                                            , keepprob
                                                            , input_x)

y_true = input_x

# Define loss and optimizer, minimize the squared error

autoencoder_loss = modular_AutoEncoder.loss_function()
ae_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(autoencoder_loss)
# Initialize the variables (i.e. assign their default value)

##########################################################################################
ensemble = ensemble.Ensemble(number_of_autoencoder_modules, num_classes)

from functools import partial


def feature_extraction_mapping(x, y, weight1, bias1):
    _input_r = tf.reshape(x, shape=[-1, 28, 28, 1])
    first_transform = tf.add(tf.nn.conv2d(_input_r, weight1
                        , strides=[1, 2, 2, 1], padding='SAME'), bias1)
    return (first_transform, y)


import dataset_helper as ds_helper
laoded_dataset = ds_helper.load_dataset_by_name(dataset_name)
train_x = laoded_dataset['train_x']
train_y = laoded_dataset['train_y']
test_x = laoded_dataset['test_x']
test_y = laoded_dataset['test_y']

def prepare_dataset(cae):
    feature_extraction_function = partial(feature_extraction_mapping
                                          , weight1=cae['weights']['ce1'], bias1=cae['biases']['be1'])

    # create the training datasets
    dx_train = tf.data.Dataset.from_tensor_slices(train_x)
    dy_train = tf.data.Dataset.from_tensor_slices(train_y)

    # TODO: 10000 and 50000 are sizes of train and test set
    train_dataset = tf.data.Dataset.zip((dx_train, dy_train)).repeat().batch(1).map(feature_extraction_function)
    # do the same operations for the validation set
    dx_test = tf.data.Dataset.from_tensor_slices(test_x)
    dy_test = tf.data.Dataset.from_tensor_slices(test_y)
    valid_dataset = tf.data.Dataset.zip((dx_test, dy_test)).repeat().batch(1).map(feature_extraction_function)

    # create general iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    training_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(valid_dataset)

    return [iterator, training_init_op, validation_init_op, feature_extraction_function]


nn_models = []
iterators = []

#//TODO: remove 16 from hard code
X = tf.placeholder("float", [None, 196])
Y = tf.placeholder("float", [None, 10])


for i in range(number_of_autoencoder_modules):
    ''' iterators =  [iterator, training_init_op, validation_init_op] 
        nn_models =[logits, loss, optimizer, prediction, equality, accuracy]
    '''
    cae_module = modular_AutoEncoder.get_list_of_modules()[i]
    iterators.append(prepare_dataset(cae_module))
    nn_models.append(ensemble.addModel(X, Y, classtype))



###########################################################################################
# Training AutoEncoder

# create the training datasets
ae_dx_train = tf.data.Dataset.from_tensor_slices(train_x)
ae_dy_train = tf.data.Dataset.from_tensor_slices(train_y)

ae_train_dataset = tf.data.Dataset.zip((ae_dx_train, ae_dy_train)).repeat().batch(batch_size)
# do the same operations for the validation set
ae_dx_test = tf.data.Dataset.from_tensor_slices(test_x)
ae_dy_test = tf.data.Dataset.from_tensor_slices(test_y)
ae_valid_dataset = tf.data.Dataset.zip((ae_dx_test, ae_dy_test)).repeat().batch(batch_size)
# create general iterator
ae_iterator = tf.data.Iterator.from_structure(ae_train_dataset.output_types,
                                           ae_train_dataset.output_shapes)
ae_training_init_op = ae_iterator.make_initializer(ae_train_dataset)
#ae_validation_init_op = ae_iterator.make_initializer(ae_valid_dataset)

###########################################################################################
init = tf.global_variables_initializer()


session = tf.Session()

# Run the initializer
session.run(init)

############################################################################################


session.run(ae_training_init_op)
ae_next_element = ae_iterator.get_next()

start = time.time()
for i in range(1, autoencoder_num_steps+1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    batch_x, _ = session.run(ae_next_element)

    # Run optimization op (backprop) and cost op (to get loss value)
    _, l = session.run([ae_optimizer, autoencoder_loss], feed_dict={input_x: batch_x, keepprob: dropout_value})
    # Display logs per step
    if i % display_step == 0 or i == 1:
        print('Step %i: Minibatch Loss: %f' % (i, l))

end = time.time()
print(end - start)

#ensemble.set_feature_mappers(modular_AutoEncoder.get_list_of_mappings())

print("AutoEncoder finished ###########################################################################################################")


import pickle

stride = 2

if (save_datasets_flag == True):
    for i in range(number_of_autoencoder_modules):
        cae = modular_AutoEncoder.get_list_of_modules()[i]
        weights = cae['weights']['ce1']
        bias1 = cae['biases']['be1']
        _input_r = tf.reshape(train_x, shape=[-1, 28, 28, 1])

        # sigmoid added these stuff are expremental
        first_transform = tf.add(tf.nn.conv2d(_input_r, weights
                                              , strides=[1, stride, stride, 1], padding='SAME'), bias1)
        pooling = tf.layers.max_pooling2d(first_transform, pool_size=[2, 2], strides=2)

        #     print(first_transform.shape)
        #     output = tf.add(tf.nn.conv2d_transpose(first_transform, weights, tf.stack([tf.shape(first_transform)[0], 20, 20, 1]),
        #                                          strides=[1, stride, stride, 1]
        #                                          , padding='SAME'), bias1)
        #     print("hala mah shodam")
        #     print(output.shape)

        pickle.dump((session.run(pooling), train_y), open(dataset_name + str(i) + 'train.pcl', "wb"))

        cae = modular_AutoEncoder.get_list_of_modules()[i]
        weights = cae['weights']['ce1']
        bias1 = cae['biases']['be1']
        _input_r_test = tf.reshape(test_x, shape=[-1, 28, 28, 1])
        first_transform_test = tf.add(tf.nn.conv2d(_input_r_test, weights
                                                   , strides=[1, stride, stride, 1], padding='SAME'), bias1)

        pooling_test = tf.layers.max_pooling2d(first_transform_test, pool_size=[2, 2], strides=2)
        pickle.dump((session.run(pooling_test), test_y), open(dataset_name + str(i) + 'test.pcl', "wb"))

    print("exiting program: save_datasets_flag is True")
    session.close()
    exit(0)




#it_op = tf.global_variables_initializer()

for i in range(number_of_autoencoder_modules):
    training_init_op = iterators[i][1]
    session.run(training_init_op)

for i in range(number_of_autoencoder_modules):
    # run the training
    epochs = 1
    ensemble_loss = nn_models[0][i][0][1]
    ensemble_optimizer = nn_models[0][i][0][2]
    ensemble_accuracy = nn_models[0][i][0][5]
    next_element = iterators[i][0].get_next()
    for k in range(epochs):
        (features, labels) = session.run(next_element)
        #TODO: batch size should be fixed for dataset api 12000 size
        one_dimensional_features = session.run(tf.reshape(features, shape=[1, 196]))
        l, _, acc = session.run([ensemble_loss, ensemble_optimizer, ensemble_accuracy], feed_dict={X: one_dimensional_features, Y: labels})
        if k % 200 == 0:
            print("Epoch: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(k, l, acc * 100))
    # now setup the validation run

for i in range(number_of_autoencoder_modules):
    # run the training
    ensemble_accuracy = nn_models[0][i][0][5]
    valid_iters = 1
    # re-initialize the iterator, but this time with validation data
    validating_op = iterators[i][2]
    session.run(validating_op)
    avg_acc = 0
    next_element = iterators[i][0].get_next()
    for m in range(valid_iters):
        (features, labels) = session.run(next_element)
        # TODO: batch size should be fixed for dataset api 12000 size
        one_dimensional_features = session.run(tf.reshape(features, shape=[1, 196]))
        acc = session.run([ensemble_accuracy], feed_dict={X: one_dimensional_features, Y: labels})
        avg_acc += acc[0]
    print("Average validation set accuracy over {} iterations is {:.2f}%".format(valid_iters,
                                                                                     (avg_acc / valid_iters) * 100))




print("Ensemble finished ###########################################################################################################")
from collections import Counter

# //TODO: epochs should be changed based on dataset's size
voting_batch_number = total_test_size

#test_x = mnist.test.images
#test_y = mnist.test.labels
# create the training datasets
##################################
dx_voting_test = tf.data.Dataset.from_tensor_slices(test_x)
dy_voting_test = tf.data.Dataset.from_tensor_slices(test_y)
valid_voting_dataset = tf.data.Dataset.zip((dx_voting_test, dy_voting_test)).repeat().batch(voting_batch_number)  # .map(feature_extraction_function)
#############################################
# create general iterator
ensmble_valid_test_iterator = tf.data.Iterator.from_structure(valid_voting_dataset.output_types,
                                           valid_voting_dataset.output_shapes)
ensmble_valid_test_voting_op = ensmble_valid_test_iterator.make_initializer(valid_voting_dataset)

#''' enabling voting iterator (in Dataset api)'''
#for j in range(number_of_autoencoder_modules):

session.run(ensmble_valid_test_voting_op)

def final_votes(votes):
    final_votes = []
    for column in votes.T:
        most_common, num_most_common = Counter(column).most_common(1)[0]
        final_votes.append(most_common)
    return final_votes

def flatten_labels(labels):
    final_labels = []
    if classtype == "twoclass":
        return labels.flatten()
    elif classtype == "multyclass":
        for i in labels:
            final_labels.append(np.argmax(i))
    return final_labels


def calculate_accuracy(votes, labels):
    voting_accuracy = 0
    for i in range(voting_batch_number):
        if votes[i] == labels[i]:
            voting_accuracy += 1
    return voting_accuracy


#@autograph.convert()
def majority_voting(session, x, y):
    votes = np.empty((0, voting_batch_number), int)
    for i in range(number_of_autoencoder_modules):
        # run the training
        feature_extractor = iterators[i][3]
        input, label = feature_extractor(x, y)
        transformed_x = session.run(input)
        ensemble_prediction = nn_models[0][i][0][3]
        prediction = session.run(ensemble_prediction, feed_dict={X[i]: transformed_x, Y: y})
        if classtype == "twoclass":
            prediction = prediction.flatten()
            prediction = np.round(prediction)
            votes = np.append(votes, [prediction], axis=0)
        else:
            votes = np.append(votes, [prediction], axis=0)
    votes = final_votes(votes)
    labels = flatten_labels(y)
    voting_accuracy = calculate_accuracy(votes, labels)
    return voting_accuracy

next_element = ensmble_valid_test_iterator.get_next()
features, labels = session.run(next_element)
accuracy = majority_voting(session, features, labels)
#    print("Average validation set accuracy over {} iterations is {:.2f}%".format(valid_iters,
#                                                                                     (avg_acc / valid_iters) * 100))
print("Voting Finished ##################################################################")
print((accuracy / voting_batch_number) * 100)

session.close()
