import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

# Enviroment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.set_random_seed(1)
np.random.seed(1)

# data
BATCH_SIZE = 64
NEURONS_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
test_x = mnist.test.images[:200]
test_y = mnist.test.labels[:200]

tf_x = tf.placeholder(tf.float32, [None, TIME_STEP*INPUT_SIZE])
tf_y = tf.placeholder(tf.float32, [None, 10])
image = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])

# RNN
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=NEURONS_SIZE)   # num_units:设置隐藏层神经元y的个数
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    rnn_cell,
    image,
    initial_state=None,
    dtype=tf.float32,
    time_major=False
)
output = tf.layers.dense(outputs[:, -1, :], 10)
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
train_o = tf.train.AdamOptimizer(LR).minimize(loss)
accuracy = tf.metrics.accuracy(
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))[1]

# Train
sess = tf.Session(config=config)
init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
sess.run(init)

for step in range(1200):
    train_x, train_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_o, loss],{tf_x: train_x, tf_y: train_y})
    if step% 50 == 0:
        accuracy_ = sess.run(accuracy, {tf_x: test_x,tf_y: test_y})
        print("train loss: %.4f" % loss_, "| test accuracy is %.2f" % accuracy_)

# Result
test_output = sess.run(output, {tf_x: test_x[:10]})
test_result = np.argmax(test_output, 1)
print("test result:", test_result)
real_result = np.argmax(test_y[:10], 1)
print("real result:", real_result)