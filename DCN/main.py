import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
import config


# 模型输入
def init():
    tf.reset_default_graph()

    feat_dense = tf.placeholder(dtype=tf.float64, shape=[None, config.dense_zie], name='feat_dense')
    feat_spare = tf.placeholder(dtype=tf.int64, shape=[None, config.spare_size], name='feat_spare')


    # 将离散特征Embedding
    fixed_embedding = embedding(feat_spare)
    # Embedding后拼接连续特征
    x0 = tf.concat([fixed_embedding, feat_dense], 1)
    # 计算cross层的输出
    x_cross = cross_net(x0)
    x_deep = deep_net(x0)
    # 拼接Cross和Deep输出
    x_out = tf.concat([x_cross, x_deep], 1)
    return predict(x_out)

# 特征Embedding
def embedding(feat_spare):

    # 构造Embedding矩阵 spare_size * dim_size[i] * embed_size
    embed_array = []
    for i in range(config.spare_size):
        embed_array.append(
            tf.Variable(np.random.normal(
                size=(config.dimension_array[i], config.embed_size)
            ).astype(np.float64)
                        )
        )

    # 离散特征Embedding
    after_embedding_array = []
    for i in range(config.spare_size):
        before_embedding = tf.expand_dims(feat_spare[:, i], 1)
        after_embedding = tf.nn.embedding_lookup(embed_array[i], before_embedding)
        after_embedding = tf.squeeze(after_embedding, 1)
        after_embedding_array.append(after_embedding)
    fixed_embedding = tf.concat(after_embedding_array, 1)
    return fixed_embedding

# Cross网络
def cross_net(x0):
    cross_w = tf.get_variable(name='cross_w',
                              shape=[config.cross_layers, config.dense_zie + config.spare_size * config.embed_size],
                              dtype=tf.float64, initializer=tf.glorot_normal_initializer())
    cross_b = tf.get_variable(name='cross_b',
                              shape=[config.cross_layers, config.dense_zie + config.spare_size * config.embed_size],
                              dtype=tf.float64, initializer=tf.glorot_normal_initializer())
    x_cross = x0
    for l in range(config.cross_layers):
        wl = tf.reshape(cross_w[l], shape=[-1, 1])
        xlw = tf.matmul(x_cross, wl)
        x_cross = x0 * xlw + x_cross + cross_b[l]

    return x_cross


# Deep网络
def deep_net(x0):
    x_deep = x0
    for i in range(len(config.deep_layers)):
        x_deep = tf.contrib.layers.fully_connected(inputs=x_deep, num_outputs=config.deep_layers[i],
                                                   activation_fn=tf.nn.relu,
                                                   weights_regularizer=tf.contrib.layers.l2_regularizer(config.l2_reg),
                                                   scope='mlp%d' % i)
        x_deep = tf.nn.dropout(x_deep, keep_prob=config.dropout[i])

    return x_deep


def predict(x_out):

    labels = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='labels')

    # 加权求和
    y = tf.contrib.layers.fully_connected(inputs=x_out, num_outputs=1, activation_fn=tf.identity,
                                          weights_regularizer=tf.contrib.layers.l2_regularizer(config.l2_reg),
                                          scope='out_layer')

    # 构造auc
    pred = tf.nn.sigmoid(y)
    auc_value, auc_op = tf.metrics.auc(labels, pred)

    y = tf.reshape(y, shape=[-1])

    # 构造Loss
    labels = tf.reshape(labels, [-1, ])
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels))

    # 优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    # optimizer = tf.train.AdagradOptimizer(learning_rate=config.learning_rate, initial_accumulator_value=1e-8)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=config.learning_rate, momentum=0.95)
    # optimizer = tf.train.FtrlOptimizer(config.learning_rate)

    # 构造训练过程
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return auc_value, auc_op, train_op, loss, pred


def get_all_data(path):
    data = pd.read_csv(path)
    feat_dense = data.values[:, 1:14]
    feat_sparse = data.values[:, 14:]
    feat_labels = data.values[:, 1:2]
    return feat_dense, feat_sparse, feat_labels


def get_batch_data(data, start, batch_size):
    batch_data = train.loc[start * batch_size:start * batch_size + batch_size - 1]
    feat_dense = batch_data.values[:, 1:14]
    feat_sparse = batch_data.values[:, 14:]
    feat_labels = batch_data.values[:, 1:2]
    return feat_dense, feat_sparse, feat_labels


if __name__ == '__main__':

    print('开始.........')

    print('加载计算图')
    auc_value, auc_op, train_op, loss, pred = init()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print('加载数据')
    test_dense, test_sparse, test_labels = get_all_data(config.test_path)
    val_dense, val_sparse, val_labels = get_all_data(config.val_path)
    train = pd.read_csv(config.train_path)

    # 训练
    print('开始训练')
    for i in range(config.epoch):
        print('epoch of {}/{}'.format((i + 1), config.epoch))

        for i in range(int(train.shape[0] / config.batch_size)):

            train_feat_dense, train_feat_sparse, train_feat_labels = get_batch_data(train, i, config.batch_size)

            sess.run([train_op],feed_dict={'feat_dense:0': train_feat_dense,
                                           'feat_spare:0': train_feat_sparse,'labels:0': train_feat_labels})
            # 每200 batch_zie计算训练集AUC
            if (i % 200 == 0):
                auc,_ = sess.run([auc_value,auc_op],feed_dict={'feat_dense:0': train_feat_dense,
                                           'feat_spare:0': train_feat_sparse,'labels:0': train_feat_labels})

                # auc = auc_value.eval(session=sess, feed_dict={'feat_dense:0': train_feat_dense, 'feat_spare:0':train_feat_sparse, 'labels:0': train_feat_labels})
                print(auc)

        # 每个epoch计算验证集AUC
        auc,_ = sess.run([pred,auc_value], feed_dict={'feat_dense:0': val_dense,
                                            'feat_spare:0': val_sparse,'labels:0': val_labels})
        print('AUC in evl set {}'.format(auc))

    # 测试
    auc, _ = sess.run([auc_value, auc_op], feed_dict={'feat_dense:0': test_dense,
                                                      'feat_spare:0': test_sparse, 'labels:0': test_labels})
    print('AUC in test set {}'.format(auc))
