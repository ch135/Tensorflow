import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.manifold import TSNE
"""
    # 设置最小GPU使用量
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess=tf.Session(config=config)
    ==================================================
    # 自定义GPU使用量
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4 # 占用GPU40%的显存
    session = tf.Session(config=config)
"""
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5

tf.get_seed(1)
np.random.seed(1)

def plot_with_labels(lowDWeight,labels):
    plt.cla()
    X, Y = lowDWeight[:, 0], lowDWeight[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255*s/9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title("The lasrt layer")
    plt.show()
    plt.pause(0.01)

# Data
BATCH_SIZE = 50
LR = 0.001
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]
tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255
image = tf.reshape(tf_x, [-1, 28, 28, 1])   # (batch, height, width, channel): -1表示任意数量的样本数,大小为28x28深度为一的张量
tf_y = tf.placeholder(tf.int32, [None, 10])

# CNN
conv1 = tf.layers.conv2d(
    inputs=image,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)
pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
flat = tf.reshape(pool2, [-1, 7*7*32])
out_put = tf.layers.dense(flat, 10)

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=out_put)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)
accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(out_put, axis=1))[1]

# Train
sess = tf.Session(config=config)
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)

plt.ion()
for step in range(600):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:
        accuracy_, flat_repressentation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
        print("Step:", step, "| train loss:%.4f" % loss_,"| test accuracy:%.2f"%accuracy_)
        # 将{2000,7*7*18} 转化为{2，7*7*18}
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 500
        low_dim_embs = tsne.fit_transform(flat_repressentation[:plot_only, :])
        labels = np.argmax(test_y, axis=1)[:plot_only]
        plot_with_labels(low_dim_embs, labels)
plt.ioff()

# Test
test_op = sess.run(out_put, {tf_x: test_x[:10]})
test_l = np.argmax(test_op, 1)
print(test_l, "text number")
real_l = np.argmax(test_y[:10], 1)
print(real_l, "real number")
