import tensorflow as tf

# placeholder是一个等待输入的变量，通常与feed_fict搭配
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [3.], input2: [4.5]}))