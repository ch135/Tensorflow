import tensorflow as tf
import matplotlib.pyplot as plt
import  numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tensorflow.examples.tutorials.mnist import input_data
"""
    @author chenhao
    @time 2018/11/13
    @title: AutoEncoder
    @Content
        AutoEncoder 可以看成一种有损压缩算法
        特点：
            1）数据相关(只能压缩训练类似的数据)
            2）从数据样本自动学习（训练一种特定的编码器）
            3）自监督学习
        应用：
            1)数据降噪
            2)可视化降维（PCA、TSNE）
"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.set_random_seed(1)

# Data
BATCH_SIZE = 64
N_TEST_IMG = 5
LR = 0.002
mnist = input_data.read_data_sets("./MNIST_data", one_hot=False)
test_x = mnist.test.images[:200]
test_y = mnist.test.labels[:200]
tf_x = tf.placeholder(tf.float32, [None, 28*28])
view_data = mnist.test.images[:N_TEST_IMG]

# encode
en0 = tf.layers.dense(tf_x, 128, tf.nn.tanh)
en1 = tf.layers.dense(en0, 64, tf.nn.tanh)
en2 = tf.layers.dense(en1, 12, tf.nn.tanh)
encode = tf.layers.dense(en2, 3)
# decode
de0 = tf.layers.dense(encode, 12, tf.nn.tanh)
de1 = tf.layers.dense(de0, 64, tf.nn.tanh)
de2 = tf.layers.dense(de1, 128, tf.nn.tanh)
decode = tf.layers.dense(de2, 28*28, tf.nn.sigmoid)
loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decode)
train = tf.train.AdamOptimizer(LR).minimize(loss)

# show source image
"""
    f: figure
    a: figure上划分小块的数组
    ========================================================
    plt.subplot() 与 plt.subplots() 区别
    两者都可以实现画子图功能，只不过subplots帮我们把画板规划好了，
    返回一个坐标数组对象，而subplot每次只能返回一个坐标对象，
    subplots可以直接指定画板的大小。
"""
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data[i], (28, 28)), cmap="gray")
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

# Train
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)
for step in range(8000):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    train_, encode_, decode_, loss_ = sess.run([train, encode, decode, loss], {tf_x: b_x})

    if step % 1000 == 0:
        print("step %d" % step, "loss is %.4f" % loss_)
        decode_data = sess.run(decode, {tf_x: view_data})
        for i in range(N_TEST_IMG):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(decode_data[i], (28, 28)), cmap="gray")
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
        """
            plt.show() 与 plt.draw() 区别
            plt.show() 展示当前画板；plt.draw() 展示已有的画板上，并可改变上面数据
        """
        plt.draw()
        plt.pause(0.01)
plt.ioff()

# Test
encode_data = sess.run(encode, {tf_x: test_x[:200]})
fig = plt.figure(2)
ax = Axes3D(fig)
X, Y, Z = encode_data[:, 0], encode_data[:, 1], encode_data[:, 2]
labels = test_y[:200]
for x, y, z, s in zip(X, Y, Z, labels):
    c = cm.rainbow(int(255*s/9))
    ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()
