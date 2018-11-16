import tensorflow as tf
"""
    @Author chen hao
    @Date 2018/11/15
    @Content
        tensorflow 命名空间
        tf.name_scope() 不会标记 tf.get_variable(), 只标记 tf.variable();
        tf.variable_scope() 会标记 tf.get_variable() 和 tf.Variable();
        tf.variable_scope().reuse_variables() 重新标记变量
"""
with tf.name_scope("name_scope"):
    initializer = tf.constant_initializer(1)
    var1 = tf.get_variable(name="var1", dtype=tf.float32, initializer=initializer, shape=[1])
    var2 = tf.Variable(name="var2", initial_value=[2], dtype=tf.float32)
    var21 = tf.Variable(name="var2", initial_value=[2.1], dtype=tf.float32)
    var22 = tf.Variable(name="var2", initial_value=[2.2], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name)
    print(sess.run(var1))
    print(var2.name)
    print(sess.run(var2))
    print(var21.name)
    print(sess.run(var21))
    print(var22.name)
    print(sess.run(var22))

with tf.variable_scope("variable_scope") as scope:
    initializer = tf.constant_initializer(value=3)
    var3 = tf.get_variable(name='var3', shape=[1], dtype=tf.float32, initializer=initializer)
    var4 = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
    var4_reuse = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
    scope.reuse_variables()
    var3_reuse = tf.get_variable(name='var3',)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var3.name)            # a_variable_scope/var3:0
    print(sess.run(var3))       # [ 3.]
    print(var4.name)            # a_variable_scope/var4:0
    print(sess.run(var4))       # [ 4.]
    print(var4_reuse.name)      # a_variable_scope/var4_1:0
    print(sess.run(var4_reuse)) # [ 4.]
    print(var3_reuse.name)      # a_variable_scope/var3:0
    print(sess.run(var3_reuse)) # [ 3.]
