import numpy as np
import tensorflow as tf

def main():

    # Training Data :  y = 2x
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    W = tf.Variable(3., name='W1')
    cost = tf.square(X * W - Y)

    # Calcluate and update the gradient using tensorflow built-in optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    gradient = optimizer.compute_gradients(cost,[W])
    apply_gradient = optimizer.apply_gradients(gradient)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for _ in range(100):
        sess.run(apply_gradient, feed_dict={X:x, Y:y})

    print("W is : {}".format(sess.run(W)))
    # W is : 2.0

if __name__ == '__main__':
    main()