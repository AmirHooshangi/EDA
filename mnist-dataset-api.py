from __future__ import division, print_function, absolute_import

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

batch_size = 128

mapping_weights = tf.Variable(tf.random_normal([784, 128]), tf.float64)
mapping_bias = tf.Variable(tf.random_normal([128]), tf.float64)

def test_function(x,y):
    return (tf.add(tf.matmul(x, mapping_weights), mapping_bias),y)

def MNIST_dataset_example():
    # split into train and validation sets
    train_images = mnist.train.images
    train_labels = mnist.train.labels
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    # create the training datasets
    dx_train = tf.data.Dataset.from_tensor_slices(train_images)
    # apply a one-hot transformation to each label for use in the neural network
    dy_train = tf.data.Dataset.from_tensor_slices(train_labels)
    # zip the x and y training data together and shuffle, batch etc.
    train_dataset = tf.data.Dataset.zip((dx_train, dy_train)).repeat().batch(batch_size).map(test_function)
    # do the same operations for the validation set
    dx_test = tf.data.Dataset.from_tensor_slices(test_images)
    dy_test = tf.data.Dataset.from_tensor_slices(test_labels)
    valid_dataset = tf.data.Dataset.zip((dx_test, dy_test)).shuffle(500).repeat().batch(batch_size).map(test_function)
    # create general iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    next_element = iterator.get_next()

    # make datasets that we can initialize separately, but using the same structure via the common iterator
    training_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(valid_dataset)
    # create the neural network model
    logits = nn_model(next_element[0])
    # add the optimizer and loss
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=next_element[1], logits=logits))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    # get accuracy
    prediction = tf.argmax(logits, 1)
    equality = tf.equal(prediction, tf.argmax(next_element[1], 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    init_op = tf.global_variables_initializer()
    # run the training
    epochs = 700
    with tf.Session() as sess:

        sess.run(init_op)
        sess.run(training_init_op)
        print(sess.run(next_element[0][0]))
        for i in range(epochs):
            l, _, acc = sess.run([loss, optimizer, accuracy])
            if i % 50 == 0:
                print("Epoch: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i, l, acc * 100))
        # now setup the validation run
        valid_iters = 1000
        # re-initialize the iterator, but this time with validation data
        sess.run(validation_init_op)
        avg_acc = 0
        for i in range(valid_iters):
            acc = sess.run([accuracy])
            avg_acc += acc[0]
        print("Average validation set accuracy over {} iterations is {:.2f}%".format(valid_iters,
                                                                                     (avg_acc / valid_iters) * 100))

def nn_model(in_data):
    bn = tf.layers.batch_normalization(in_data)
    fc1 = tf.layers.dense(bn, 64, activation=tf.tanh)
    fc2 = tf.layers.dropout(fc1)
    fc3 = tf.layers.dense(fc2, 10)
    return fc3


MNIST_dataset_example()