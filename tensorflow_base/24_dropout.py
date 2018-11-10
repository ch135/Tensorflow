import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 每次取一样的随机值
tf.set_random_seed(1)
np.random.seed(1)

N_SAMPLES = 20
N_HIDEN = 300
LR = 0.01

# training data
x = np.linspace(-1, 1, N_SAMPLES)[:, np.newaxis]
y = x+0.3 * np.random.randn(N_SAMPLES)[:, np.newaxis]

# test data
text_x = np.copy(x)
text_y = text_x+0.3 * np.random.randn(N_SAMPLES)[:, np.newaxis]

# tf placeholders
tf_x = tf.placeholder(tf.float32, [None, 1])
tf_y = tf.placeholder(tf.float32, [None, 1])
tf_is_training = tf.placeholder(tf.bool, None)

# overfitting net
"""
    tf.layers.dense(
        inputs, #该层的输入
        units,  #输出的大小（维数），整数或log
        activation=None,
        use_bias=True, # 使用bias为True（默认使用），不用bias改成False即可
        ）
"""
o1 = tf.layers.dense(tf_x, N_HIDEN, tf.nn.relu)
o2 = tf.layers.dense(o1, N_HIDEN, tf.nn.relu)
o_out = tf.layers.dense(o2, 1)
o_loss = tf.losses.mean_squared_error(tf_y, o_out)
o_train = tf.train.AdamOptimizer(LR).minimize(o_loss)

# dropout net
"""
    dropout之 inverted dropout
    1.设定保留神经元比例。keep_prob = 0.8(20%的元素设置为0)
    2.将80%神经元与输入元素计算，输出结果 a
    3.对a进行 scale up(保证a的期望值与之前变化不大)：a/ =keep_prob
    ------测试时不需要dropout------
"""
d1 = tf.layers.dense(tf_x, N_HIDEN, tf.nn.relu)
d1 = tf.layers.dropout(d1, rate=0.5, training=tf_is_training)
d2 = tf.layers.dense(d1, N_HIDEN, tf.nn.relu)
d2 = tf.layers.dropout(d2, rate=0.5, training=tf_is_training)
d_out = tf.layers.dense(d2, 1)
d_loss = tf.losses.mean_squared_error(tf_y, d_out)
d_train = tf.train.AdamOptimizer(LR).minimize(d_loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

plt.ion()

# show result
"""
    plt.ion()
    .....
    plt.ioff()
    plt.show()
===========================    
    cla()    Clear axis
    clf()    Clear figure
    close()  Close a figure window
"""
for t in range(500):
    sess.run([o_train, d_train], {tf_x: x, tf_y: y, tf_is_training: True})
    if t % 10 == 0:
        plt.cla()
        o_loss_, d_loss_, o_out_, d_out_ = sess.run([o_loss, d_loss, o_out, d_out],
                                                {tf_x: text_x, tf_y: text_y, tf_is_training: False})
        plt.scatter(x, y, c="magenta", s=50, alpha=0.3, label="train")
        plt.scatter(text_x, text_y, c="cyan", s=50, alpha=0.3, label="dropout(50%)")
        plt.plot(text_x, o_out_, 'r-', lw=3, label="overfitting")
        plt.plot(text_x, d_out_, 'b--', lw=3, label="dropout(50%)")
        plt.text(0, -1.2, 'overfitting loss=%.4f' % o_loss_, fontdict={'size': 20, 'color': "red"})
        plt.text(0, -1.5, 'dropout loss=%.4f' % d_loss_, fontdict={'size': 20, 'color': "blue"})
        plt.legend(loc="upper left")
        plt.ylim(-2.5, 2.5)
        # pause 暂停，暂时的停顿
        # plt.savefig() #保存图片
        plt.pause(0.1)
plt.ioff()
plt.show()