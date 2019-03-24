import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from dcn import *
import pandas as pd
import config


# 模型输入

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
