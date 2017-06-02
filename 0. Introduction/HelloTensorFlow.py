import tensorflow as tf

helloworld = tf.constant("Hello Tensorflow.")
sess = tf.Session()
print(sess.run(helloworld))