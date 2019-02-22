from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import ensemble
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.01
num_steps = 4000
batch_size = 128
display_step = 1000
examples_to_show = 10
# Network Parameters
hidden_layer_size = 128 # 1st layer num features
input_size = 784 # MNIST data input (img shape: 28*28)
# tf Graph input (only pictures)
num_classes = 10


# tf Graph input
X = tf.placeholder("float", [None, input_size])
Y = tf.placeholder("float", [None, num_classes])

data = [1]
ensemble = ensemble.Ensemble(data)
ensemble.create_ensemble(X, Y)
ensemble_members = ensemble.get_ensemble_members()


init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
#tf.InteractiveSession()
session = tf.Session()
session.run(init)
for i in range(len(ensemble_members)):
    inner_mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    prediction, train_op = ensemble_members[i]
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    for step in range(1, num_steps+1):
        # Prepare Data
       # Get the next batch of MNIST data (only images are needed, not labels)
       batch_x, batch_y = inner_mnist.train.next_batch(batch_size)
       session.run(train_op, feed_dict={X: batch_x, Y: batch_y})
       if step % display_step == 0 or step == 1:
           # Calculate batch loss and accuracy
           loss, acc = session.run([train_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
           print("Step " + str(step) + ", Training Accuracy= " + \
           "{:.3f}".format(acc))
           print("Optimization Finished!")
           # Calculate accuracy for MNIST test images
           print("Testing Accuracy:", \
           session.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))



finl_accuracy_count = 0
for i in range(0, 300):
    test_X, test_y = mnist.test.next_batch(1)
    #    print("label is: ", test_y[0])
    # sess = tf.InteractiveSession()
    #       print("test_y[0]", test_y[0])
    majority_vote = ensemble.majority_vote(session, X, Y, test_X, test_y)
    #print("tf.argmax(test_y[0]) : ", tf.argmax(test_y[0]))
    correct_pred = tf.equal(majority_vote, tf.argmax(test_y[0]))
    #print(tf.argmax(test_y[0]).eval())
    #        print(correct_pred.eval())
    # python_bool = tf.cast(correct_pred, tf.bool)
    # print("@#@#$@#$@#$@#$@#$@#$ " , sess.run(python_bool))
    #print(i)
    if (session.run(correct_pred)):
        finl_accuracy_count += 1  # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # final_result = sess.run(accuracy, feed_dict={X: test_X, Y: test_y})
        # print("Dadach Ensemble accuracy is: ", final_result)
        # print(correct_pred.eval())
print("final correct predictions ", finl_accuracy_count)
print("final fantasy ", (finl_accuracy_count / 300))

