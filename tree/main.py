import numpy as np
import random
from tree.buildList import BuildList
from tree.bayesAnalysis import bayes

data_set = None #数据集
train_set = None    #训练数据集
test_set = None #测试数据集
h = None    #列表节点指针
model_size = 200    #模型大小阈值

if __name__ == '__main__':
    #构建贝叶斯网络，计算属性对结果的影响度
    b = bayes()
    b.prepareData('./car.data')
    data, input_index = b.pridict()

    #为数据赋值
    data_set = b.data_set
    train_set = b.train_data
    test_set = b.test_data

    #构建帕累托最优列表
    # print(data)
    List = BuildList()
    h = List.build(data)



