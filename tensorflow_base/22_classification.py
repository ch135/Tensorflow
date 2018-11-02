import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
"""
    @author: chen hao
    @time: 2018/11/1
"""
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 添加神经网络层，返回网络层输出
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weight = tf.Variable(tf.random_normal([in_size, out_size]), name="Weight")
    Basies = tf.Variable(tf.random_normal([1, out_size]), name="Basies")
    Wx_plus_b = tf.matmul(inputs, Weight)+Basies
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 测试训练结果
def comute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs}) #将测试集放入训练好的神经网络中训练，输出测试结果
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1)) #tf.argmax()：输出某一维上最大数据的索引；判断测试值与实际值是否相等，输出布尔值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #tf.cast()将布尔值转化浮点数；并计算出平均值
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

xs = tf.placeholder(tf.float32, [None, 784]) #28*28
ys = tf.placeholder(tf.float32, [None, 10]) #one-hot:每个数字表示一个位置
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)
corss_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1])) #使用交叉熵定义模型的好坏；我们最小化这个值
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(corss_entropy)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(comute_accuracy(mnist.test.images, mnist.test.labels))

