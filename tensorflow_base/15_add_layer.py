import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定义神经层
def add_layer(inputs, in_sicze, out_sice, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_sicze, out_sice]))
    biases = tf.Variable(tf.zeros([1, out_sice])+0.1)
    Wx_plus_b = tf.matmul(inputs, Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 定义数据
# 将1*300 转化为 300*1
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data)-0.5 + noise
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# 添加神经元并计算出净输入、最后输出
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)
# 运算结果
# reduction_indices：函数处理的维度。详见https://blog.csdn.net/qq_33096883/article/details/77479766
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
result = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化所有变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()
    for i in range(1000):
        sess.run(result, feed_dict={xs: x_data, ys: y_data})
        if i % 20 == 0:
            # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
                prediction_value = sess.run(prediction, feed_dict={xs: x_data})
                lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
                # 暂停运行
                plt.pause(1)