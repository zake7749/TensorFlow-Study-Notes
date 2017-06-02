''''
Use a single layer fully-connected neural network to solve MNIST.
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load the MNIST Data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Setting the data placeholder
features = tf.placeholder('float', shape=[None, 784], name="Image_Features")
labels = tf.placeholder('float', shape=[None, 10], name="One_Hot_Labels")

# Define a fully connected layer
def fullyConnected(input, out_dim, in_dim, activation_func=None):
    W = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=0.1))
    B = tf.Variable(tf.truncated_normal([out_dim], stddev=0.1))
    Y = tf.matmul(input, W) + B

    if activation_func is not None:
        Y = activation_func(Y)
    return Y


Y = fullyConnected(features, 10, 784, activation_func=tf.nn.softmax)
loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(Y), reduction_indices=[1]))  # cross-entropy as loss function
# 採用 1 作為 reduction_indices 是因為我們是針對整個 batch 去求 total loss

train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)  # iterate the gradient descent processes to minimize the loss value

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for batch in range(1001):

        batch_xs, batch_ys = mnist.train.next_batch(128)
        sess.run(train, feed_dict={features: batch_xs, labels: batch_ys})

        prediction = tf.argmax(Y, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        if batch % 50 == 0:
            print("Loss for batch {} : {}".format(batch, sess.run(loss, feed_dict={features: batch_xs, labels: batch_ys})))
            print("Accuracy for batch {} : {}".format(batch, sess.run(accuracy, feed_dict={features: batch_xs, labels: batch_ys})))