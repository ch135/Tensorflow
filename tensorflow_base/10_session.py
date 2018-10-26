import tensorflow as tf

# tf.constant()：创建常数张量，可传入数列或数组
# 分别创建1*2和2*1张量
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])
produce = tf.matmul(matrix1, matrix2)
# method1
# sess = tf.Session()
# result = sess.run(produce)
# print(result)
# sess.close()

# method2
with tf.Session() as sess:
    result = sess.run(produce)
    print(result)
