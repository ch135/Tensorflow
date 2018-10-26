import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3
# 创建1*1的矩阵，范围在-0.1到0.1之间
Weight = tf.Variable(tf.random_uniform([1], -0.1, 0.1))
# 创建1*1数组
biases = tf.Variable(tf.zeros([1]))
y = Weight*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weight), sess.run(biases))
