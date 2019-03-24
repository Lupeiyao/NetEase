import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import config



def add_head(sample_path):
    head = '\t'.join(['label'] + ['I' + str(i) for i in range(1, 14)] + ['C' + str(i) for i in range(1, 27)]) + '\n'
    file = open(sample_path, 'r+')
    old = file.read()
    file.seek(0)
    file.write(head)
    file.write(old)
    file.close()


def sample_data_from_org_file(input_file,sample_path,size):
    input_file = open(input_file,'r')
    if(os.path.exists(sample_path)):
        pass
    else:
        sample_file = open(sample_path, 'w')
        tag = '1'
        i = 0
        for line in input_file:
            # 一条label：1，一条label：0
            label = line.split('\t')[0]
            if(label != tag):
                sample_file.write(line)
                tag = label
                i += 1
            if(i >= size):
                break
        sample_file.close()
    input_file.close()

def feature_enginee(sample_path):
    data = pd.read_csv(sample_path, '\t')

    #连续、离散数据表头
    dense_features = config.dense_features
    spare_features = config.spare_features

    #空值填充
    data[dense_features] = data[dense_features].fillna(0.0)
    data[spare_features] = data[spare_features].fillna('-1')

    #截取连续值，并min-max归一化
    continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
    for i in range(len(dense_features)):
        data[dense_features[i]] = pd.to_numeric(data[dense_features[i]])
        data[dense_features[i]] = data[dense_features[i]].map(lambda x: x if x < continous_clip[i] else continous_clip[i])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    #删除出现次数少于10的特征取值
    for feature in spare_features:
        dic = dict()
        for value in data[feature]:
            dic[value] = dic.get(value,0) + 1
        data[feature] = data[feature].map(lambda value: '-1' if dic[value] < 10 else value)

    #One-Hot离散值
    for feature in spare_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])


    return data


if __name__ == '__main__':

    # 从原始数据文件抽取实验数据
    print("采样")
    sample_data_from_org_file(config.org_path,config.sample_path,config.sample_size)

    # 给采样文件添加表头
    print('增加表头')
    add_head(config.sample_path)

    # 特征工程
    print('特征工程')
    data = feature_enginee(config.sample_path)

    # 记录离散值取值数
    print('记录离散取值数')
    spare_feature_num_list = []
    for feature in config.spare_features:
        spare_feature_num_list.append(data[feature].nunique())
    print(spare_feature_num_list)

    #构造训练、验证、测试集合
    print('构造训练、验证、测试')
    train,other = train_test_split(data,test_size=0.4)
    val,test = train_test_split(other,test_size=0.5)

    # 保存
    print('保存')
    train.to_csv(config.train_path,index=0)
    val.to_csv(config.val_path,index=0)
    test.to_csv(config.test_path,index=0)

    print('done')