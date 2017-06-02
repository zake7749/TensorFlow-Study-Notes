import numpy as np
import tensorflow as tf

def main():

    # Training Data :  y = 2x
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    W = tf.Variable(tf.random_normal([1]), name='W1')

    learning_rate = 0.1

    # A low level gradient descent
    gradient = tf.reduce_mean((W * X - Y) * X)
    descent = W - learning_rate * gradient
    update = W.assign(descent)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Stochastic Gradient Desent
    for _ in range(50):
        for tx,ty in zip(x,y):
            sess.run(update, feed_dict={X:tx, Y:ty})

    print("W is : {}".format(sess.run(W)))
    # W is : [ 2.]

if __name__ == '__main__':
    main()