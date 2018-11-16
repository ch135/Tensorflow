import tensorflow as tf
import  matplotlib.pyplot as plt
import numpy as  np

"""
    @author chenhao
    @time 2018/11/13
    @title: AutoEncoder
    @Content
        在 tensorflow 中保存和导入数据
        1. 保存数据
            1.1 简单保存
                saver = tf.train.Saver()
                saver.save(sess, "my-test-model")
            1.2 迭代1000次后保存
                saver.save(sess, "my_test_model", global_step=1000)
            1.3 训练中只保存一次
                saver.save(sess, "my-model", global_step=step, write_meta_graph=False)
            1.4 每 2 小时保存最新的 4 个模型数据
                saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
            1.5 保存模型中特定变量
                saver = tf.train.Saver([w1,w2])
                saver.save(sess, 'my_test_model',global_step=1000)
        2. 导入数据
            2.1 创建数据
                saver = tf.train.import_meta_graph("my_test_model-1000.meta")
            2.2 导入数据
                with tf.Session() as sess:    
                    saver = tf.train.import_meta_graph('my-model-1000.meta')
                    saver.restore(sess,tf.train.latest_checkpoint('./'))
                    print(sess.run('w1:0'))
            *** 
                import_meta_graph 函数只将保存在文件中的数据添加到当前网络。虽然
                创建了额外的图/网络，但还要导入这个图训练好的模型参数。
                                                                                ***
            2.3 使用新数据训练原来网络
                saver = tf.train.import_meta_graph('my_test_model-1000.meta')
                saver.restore(sess,tf.train.latest_checkpoint('./'))
                graph = tf.get_default_graph()
                w1 = graph.get_tensor_by_name("w1:0")
                w2 = graph.get_tensor_by_name("w2:0")
                feed_dict ={w1:13.0,w2:17.0}
"""
# Enviroment
tf.set_random_seed(1)
np.random.seed(1)

# Data
x = np.linspace(-1, 1, 100)[:, np.newaxis]  # shape(1000, 1)
noise = np.random.normal(0, 0.1, size=x.shape)  # shape(1000, 1)
y = np.power(x, 2) + noise

def save():
    tf_x = tf.placeholder(tf.float32, x.shape)
    tf_y = tf.placeholder(tf.float32, y.shape)
    layer1 = tf.layers.dense(tf_x, 10, tf.nn.relu)
    output = tf.layers.dense(layer1, 1)
    loss = tf.losses.mean_squared_error(output, tf_y)
    train_out = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    saver = tf.train.Saver()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for step in range(100):
        sess.run(train_out, {tf_x: x,tf_y: y})
    saver.save(sess, "./DATA_save/params", write_meta_graph=False)
    # plotting
    out_, loss_ = sess.run([output, loss], {tf_x: x, tf_y: y})
    plt.figure(1, figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(x, y)
    plt.plot(x, out_, "r-", lw=5)
    plt.text(-1, 1.2, "Save loss %.4f" % loss_, fontdict={"size": 15, "color": "red"})

def reload():
    tf_x = tf.placeholder(tf.float32, x.shape)
    tf_y = tf.placeholder(tf.float32, y.shape)
    layer1 = tf.layers.dense(tf_x, 10, tf.nn.relu)
    output = tf.layers.dense(layer1, 1)
    loss = tf.losses.mean_squared_error(output, tf_y)
    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, "./DATA_save/params")
    # plotting
    out_, loss_ = sess.run([output, loss], {tf_x: x, tf_y: y})
    plt.subplot(122)
    plt.scatter(x, y)
    plt.plot(x, out_, "r-", lw=5)
    plt.text(-1, 1.2, "Reload loss %.4f" % loss_, fontdict={"size": 15, "color": "red"})
    plt.show()
save()
tf.reset_default_graph()
reload()