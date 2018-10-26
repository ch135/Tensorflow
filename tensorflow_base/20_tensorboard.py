import tensorflow as tf
import numpy as np

# 定义神经层
def add_layer(layer_num,inputs, in_sicze, out_sice, activation_function=None):
    layer = layer_num
    with tf.name_scope("weights"):
        Weights = tf.Variable(tf.random_normal([in_sicze, out_sice]), name="W")
        tf.summary.histogram(layer, Weights)
    with tf.name_scope("biases"):
        biases = tf.Variable(tf.zeros([1, out_sice])+0.1, name="B")
        tf.summary.histogram(layer, biases)
    with tf.name_scope("Wx_plus_b"):
        Wx_plus_b=tf.matmul(inputs, Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 定义数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data)-0.5 + noise
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_data")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_data")

# 添加神经元并计算出净输入、最后输出
with tf.name_scope("layer1"):
    l1 = add_layer("layer1", xs, 1, 10, activation_function=tf.nn.relu)
with tf.name_scope("layer2"):
    prediction = add_layer("layer2", l1, 10, 1, activation_function=None)

# 运算结果
# reduction_indices：函数处理的维度。详见https://blog.csdn.net/qq_33096883/article/details/77479766
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
    tf.summary.scalar("loss", loss)
with tf.name_scope("result"):
    result = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化所有变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    file_write = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init)

    for i in range(100):
        node = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        file_write.add_summary(node, i)