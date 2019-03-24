

sample_size = 1000 * 10000

org_path = "/home/data/train.txt"
sample_path = "/home/data/sample_small.txt"
train_path = "/home/data/train.csv"
val_path = "/home/data/val.csv"
test_path = "/home/data/test.csv"

dense_features = ['I' + str(i) for i in range(1, 14)]
spare_features = ['C' + str(i) for i in range(1, 27)]

feature_size = 39
spare_size = 26
dense_zie = 13
embed_size = 10
learning_rate = 0.0001
deep_layers = [128,128,128]
cross_layers = 3
dropout = [0.8,0.8,0.8]
l2_reg = 0.2
batch_size = 512
epoch = 100

dimension_array = [1369, 541, 48430, 44710, 285, 15, 10772, 605, 3, 23384, 4681, 48567, 3101, 27, 8446, 48765, 10, 3736, 1786, 4, 48737, 14, 15, 24809, 67, 20119]