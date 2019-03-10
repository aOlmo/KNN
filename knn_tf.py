import numpy as np
import tensorflow as tf

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

K = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

Xtr, Ytr = mnist.train.next_batch(6000)
Xte, Yte = mnist.test.next_batch(1000)

xtr = tf.placeholder("float", [None, 28*28])
xte = tf.placeholder("float", [28*28])
k = tf.placeholder(tf.int32, None)

# L1 calculation
l1_distance1 = tf.reduce_sum(tf.abs(xtr - xte), reduction_indices=1)

# L2 calculation
l2_distance = tf.sqrt(tf.reduce_sum(tf.pow(xtr - xte, 2), reduction_indices=1))
pred = tf.nn.top_k(tf.negative(l1_distance1), k)


init = tf.global_variables_initializer()
Xte = Xtr
Yte = Ytr
with tf.Session() as sess:
    sess.run(init)

    for k_val in K:
        accuracy = 0
        print("----------------")
        print("Using K={}".format(k_val))
        print("----------------")

        for i in range(len(Xte)):
            indices = []

            # Get nearest neighbor
            nn_indices = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :], k: k_val})
            nn_indices = nn_indices.indices

            for j in nn_indices:
                indices.append(np.argmax(Ytr[j]))

            y_pred = np.bincount(indices).argmax()

            # Get nearest neighbor class label and compare it to its true label
            if i % 200 == 0:
                print("Test", i, "Prediction:", y_pred, "True Class:", np.argmax(Yte[i]))

            # Calculate accuracy
            if y_pred == np.argmax(Yte[i]):
                accuracy += 1. / len(Xte)

        print("Done!")
        print("Accuracy:", accuracy)
