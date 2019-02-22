from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import eda_cae_autoEncoder
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.01
num_steps = 1
batch_size = 128
display_step = 1000
examples_to_show = 10
# Network Parameters
hidden_layer_size = 128 # 1st layer num features
input_size = 784 # MNIST data input (img shape: 28*28)
# tf Graph input (only pictures)
num_classes = 10

number_of_autoencoder_modules = 9
ae_first_layer_hidden_layer_size = -1 # 1st layer num features

keepprob = tf.placeholder(tf.float32)
input_x = tf.placeholder("float", [None, input_size])
modular_AutoEncoder = eda_cae_autoEncoder.EdaCaeAutoEncoder(number_of_autoencoder_modules, input_size
                                                            , ae_first_layer_hidden_layer_size
                                                            , keepprob
                                                            , input_x)

#y_pred = modular_AutoEncoder.get_list_of_modules()[0]
# Targets (Labels) are the input data.
y_true = input_x

# Define loss and optimizer, minimize the squared error

loss = modular_AutoEncoder.loss_function()

#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

#Found AdamOptimizer useful in experimental results
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
#tf.InteractiveSession()
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={input_x: batch_x, keepprob: 0.3})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    canvas_recon2 = np.empty((28 * n, 28 * n))
    canvas_recon3 = np.empty((28 * n, 28 * n))
    canvas_recon4 = np.empty((28 * n, 28 * n))
    canvas_recon5 = np.empty((28 * n, 28 * n))
    canvas_recon6 = np.empty((28 * n, 28 * n))
    canvas_recon7 = np.empty((28 * n, 28 * n))
    canvas_recon8 = np.empty((28 * n, 28 * n))
    canvas_recon9 = np.empty((28 * n, 28 * n))

    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(modular_AutoEncoder.get_list_of_modules()[0]['cae'], feed_dict={input_x: batch_x, keepprob: 0.3})
        f = sess.run(modular_AutoEncoder.get_list_of_modules()[1]['cae'], feed_dict={input_x: batch_x, keepprob: 0.3})
        z = sess.run(modular_AutoEncoder.get_list_of_modules()[2]['cae'], feed_dict={input_x: batch_x, keepprob: 0.3})
        z1 = sess.run(modular_AutoEncoder.get_list_of_modules()[3]['cae'], feed_dict={input_x: batch_x, keepprob: 0.3})
        z2 = sess.run(modular_AutoEncoder.get_list_of_modules()[4]['cae'], feed_dict={input_x: batch_x, keepprob: 0.3})
        z3 = sess.run(modular_AutoEncoder.get_list_of_modules()[5]['cae'], feed_dict={input_x: batch_x, keepprob: 0.3})
        z4 = sess.run(modular_AutoEncoder.get_list_of_modules()[6]['cae'], feed_dict={input_x: batch_x, keepprob: 0.3})
        z5 = sess.run(modular_AutoEncoder.get_list_of_modules()[7]['cae'], feed_dict={input_x: batch_x, keepprob: 0.3})
        z6 = sess.run(modular_AutoEncoder.get_list_of_modules()[8]['cae'], feed_dict={input_x: batch_x, keepprob: 0.3})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon2[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                f[j].reshape([28, 28])

        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon3[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                z[j].reshape([28, 28])

        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon4[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                z1[j].reshape([28, 28])

        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon5[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                z2[j].reshape([28, 28])

        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon6[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                z3[j].reshape([28, 28])

        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon7[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                z4[j].reshape([28, 28])

        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon8[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                z5[j].reshape([28, 28])

        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon9[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                z6[j].reshape([28, 28])


    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images2")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon2, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images3")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon3, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images4")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon4, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images5")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon5, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images6")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon6, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images7")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon7, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images8")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon8, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images9")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon9, origin="upper", cmap="gray")
    plt.show()
