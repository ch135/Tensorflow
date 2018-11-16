import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
    @author chen hao 
    @time 2018/11/14
    @Content
        线性回归和分类的区别： 看 22_classification.py
"""
# Data
x = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, x.shape)
y = np.power(x, 2) + noise
tf_x = tf.placeholder(tf.float32, x.shape)
tf_y = tf.placeholder(tf.float32, y.shape)

# Layer
layer_1 = tf.layers.dense(tf_x, 10, tf.nn.relu)
output = tf.layers.dense(layer_1, 1)
loss = tf.losses.mean_squared_error(tf_y, output)
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

result = {}
for step in range(200):
    _, loss_ = sess.run([train_op, loss], {tf_x: x, tf_y: y})
    result[step] = loss_

print(result.values())

plt.figure()
plt.plot(result.keys(), result.values(), 'r-', lw=1.0, label="loss")
plt.legend(loc="best")
plt.show()