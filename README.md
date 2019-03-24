# NetEase入职作业

#### GitHub地址：https://github.com/Lupeiyao/NetEase

#### 1.大数据生态(Spark/Hadoop/Hive/Hbase)

- 要求：了解学习大数据生态常用组件的基本原理、使用场景和使用方法，以Spark的使用为主，其余组件了解即可；
  通读官方文档，尝试编写demo程序，可参考《Spark快速大数据分析》或《Spark高级数据分析》

- 计划：2019.01.1-2019.01.15，基于Spark快速大数据分析：
  - Spark环境搭建（StandAlone）
  - 基于Idea完成Spark WordCount代码编写，打包，上传，提交，集群运行全过程
  - 学习spark基本算子，RDD(DataFram)编程
  - 学习Spark进阶（累加器，广播变量，分区操作，进程间通信）
  - 学习spark调优，熟悉Spark SQL，Streaming，ML
- 成果：于2019.01.17完成，总结Spark相关技术文档，内容参见附件或GitHub。

#### 2.Tensorflow/Scikit-Learn实战

- 要求：学习Tensorflow的基本原理和使用方法，建议阅读《机器学习实战：基于Scikit-Learn和TensorFlow》并配
  合书中的例子进行实践，熟练使用python

- 计划：2019.01.17-2019.02.01，基于Hands-On Machine Learning with Scikit-Learn & TensorFlow：
  - 熟练使用Python
  - 基于Python + TF 完成书中部分例子
- 成果：于2019.02.20完成阅读，学习Python+TensorFlow代码编写，内容参见附件或GitHub。

#### 3.Linux Shell/Git使用

- 要求：自行安装一台Linux环境进行日常开发使用，建议阅读《Linux命令行大全》前10章配合实践；学习使用git管
  理代码，参考https://git-scm.com/book/zh/v1/熟悉常用命令

- 计划：2019.01.17-2019.02.01，学习Shell脚本编写，熟练Linux命令，熟练使用Git
  - 学习Shell脚本
  - 学习Linux基本操作
  - 学习Git基本操作
  - 基于Git管理文档与代码
- 成果：于2019.01.29完成学习，总结Git与Linux技术文档，内容参见附件或GitHub。

#### 4.推荐/搜索/广告/NLP前沿论文阅读 

- 要求：调研阅读近5年在KDD/sigir/www/WSDM/Recsys等偏应用的会议上工业界发表的推荐/搜索/广告/NLP方面
  的论文，例如Wide&Deep/DIN/DeepFM/XGBoost等，推荐使用google scholar，备选百度学术

- 计划：2019.02.01-2019.03.10
  - 搜集推荐系统/FM/DCN/DIN/DeepFM等相关论文15-20篇。详细阅读，并完成内容总结
  - 每天[1|2]篇
  - 每篇总结中心思想
- 成果：与2019.03.24完成，包括FM、FFM、DCN，DIN，Wide&Deep，DeepFM，xDeepFM等推荐类论文17篇，论文总结文档内容参见附件或GitHub。

#### 5.实战训练 Done 2019.03.24

- 要求：选择主线任务四中学习到部分模型以Tensorlfow实现，并尝试在Avazu/Criteo/Movielens-1M等任一公开数
  据集上重现出论文中的结果；哪怕无法重现也请尽量逼近论文中的结果。
  根据自己以前学的机器学习知识和此次任务，整理一个算法知识体系图谱，格式不限，可以是脑图或者图标分类模式

- 计划：2019.02.01-2019.03.15
  - 挑选Deep & Cross模型进行实现
  - 总结知识图谱
- 成果：
  - 实现DCN模型，代码参见附件或GitHub。DCN模型说明参见DCN论文内容总结。
  - 在Criteo数据集（随机，按照标签1:1）抽取100W条、1000W条数各一次。共4个数据集，平衡数据集{100W，1000W}，非平衡数据集{100W，1000W}。
  - 按照6:2:2构造train，val，test数据集，利用验证集调整模型超参数，在测试集估计模型效果。
  - AUC分别为：非平衡{100W：0.79450034, 1000W：0.68456846}，平衡{100W：0.8544794, 1000W：0.6944913}
  - 知识图谱，详情见附件后GitHub。
