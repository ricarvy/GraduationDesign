# 基于卷积矩阵分解（FM）的推荐系统

1.Convolutional Matrix Factorization for Document Context-Aware Recommendation

论文理解：
* 1.CSDN翻译：https://blog.csdn.net/somtian/article/details/72666901
* 2.知乎专栏：https://zhuanlan.zhihu.com/p/27070343


epoches_error_rate: X轴表示训练轮数， y轴表示推荐项目的错误率

k_fold: X轴表示切分数据集（训练集/测试集的比例， 如20表示20%被切分成测试集， 80%被切分成训练集）， y轴表示推荐正确率

method_Dataset: 对比COMFMF和CONFMF+LSTM两个方法在MOVIELEN和KKBOX上的效果， X轴表示采用数据集的比例， Y轴表示推荐正确率