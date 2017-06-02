''''
Use a CNN to solve MNIST.
'''
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Hpyer parameter
TRAINING_EPOCH = 5
BATCH_SIZE = 128
LEARNING_RATE = 1e-2
CLASS = 10
KEEPRATIO = .75


class CNN(object):

    def __init__(self, session):

        self.features = tf.placeholder(dtype='float32', shape=[None,784])
        self.labels = tf.placeholder(dtype='float32', shape=[None,10])
        self.sess = session
        self.build()

    def build(self):

        cnn_x = tf.reshape(self.features, shape=(-1,28,28,1)) # (batch, height, width, channel)

        # Convolution layer 1

        # conv2d takes a filter with shpae [filter_height, filter_width, in_channels, out_channels]
        W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
        CL1 = tf.nn.conv2d(cnn_x, W1, strides=[1, 1, 1, 1], padding='SAME')

        # ksize is for the window size for each dimension, which is (batch, height, width, channel)
        # we usally set ksize for [1, pool_window_height, pool_window_width, 1], because doing maxpooling on the
        # batch or channel does not make sense.

        CL1 = tf.nn.max_pool(CL1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        CL1 = tf.nn.dropout(CL1, keep_prob=KEEPRATIO)

        # Convolution layer 2
        W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        CL2 = tf.nn.conv2d(CL1, W2, [1,1,1,1], padding='SAME')
        CL2 = tf.nn.max_pool(CL2, ksize=[1, 2, 2, 1], strides=[1, 2, 2 ,1], padding='SAME')
        CL2 = tf.nn.dropout(CL2, keep_prob=KEEPRATIO)

        # Fully connected NN
        flattened_x = tf.reshape(CL2, shape=[-1, 7 * 7 * 64]) # get a image with size= (28/2/2) = 7, and we have 64 filters.
        W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 10]))
        B3 = tf.Variable(tf.random_normal([10]))
        logits = tf.matmul(flattened_x , W3) + B3

        self.logits = tf.nn.softmax(logits)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=LEARNING_RATE).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.features: x_test})

    def predict_class(self, x_test):
        hypothesis = self.predict(x_test)
        return np.argmax(hypothesis)

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.features: x_test, self.labels: y_test})

    def train(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run(
                    [self.cost, self.optimizer],
                    feed_dict={
                        self.features: x_data,
                        self.labels: y_data
                    }
                )

def main():

    sess = tf.Session()
    cnn = CNN(sess)

    sess.run(tf.global_variables_initializer())

    # Load the MNIST Data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Training
    for epoch in range(TRAINING_EPOCH):

        cost = 0.
        total_batch = int(mnist.train.num_examples / BATCH_SIZE)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            c, _ = cnn.train(batch_xs, batch_ys)
            cost += c

        avg_cost = c / total_batch

        print('Epoch #%2d' % (epoch+1))
        print('- Average cost: %4f' % (avg_cost))

    # Testing
    print('Accuracy:', cnn.get_accuracy(mnist.test.images, mnist.test.labels))

if __name__ == "__main__":
    main()