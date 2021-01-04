import numpy as np
import random
from tree.buildList import BuildList
from tree.bayesAnalysis import bayes
from tree.modelPruning import modelPruning
from tree.DNN import prepare_DNN
import os
from tree.DNN import DNN

data_set = None #数据集
train_set = None    #训练数据集
test_set = None #测试数据集
h = None    #列表节点指针
model_size = 200    #模型大小阈值
model_accuracy = 80
target_size = 0

if __name__ == '__main__':
    #构建贝叶斯网络，计算属性对结果的影响度
    b = bayes()
    b.prepareData('./q.xls')
    data, input_index = b.pridict()
    target_size = b.target_size
    # iris = datasets.load_breast_cancer()
    # data = iris.data

    #为数据赋值
    data_set = b.data_set
    train_set = b.train_data
    test_set = b.test_data

    #构建帕累托最优列表
    # print(data)
    List = BuildList()
    h = List.build(data)

    #训练初始模型
    preDnn = prepare_DNN()
    model, model_accuracy = preDnn.train_DNN(train_set, test_set, b.target, target_size)
    preDnn.save_model(model, 'E:\workspace\Graduatio-Project\model\初始模型.pkl')
    model_size = os.path.getsize('E:\workspace\Graduatio-Project\model\初始模型.pkl')

    #模型删减策略
    mp = modelPruning(model_size, model_accuracy, data_set, train_set, test_set, h, input_index, b.target, target_size)
    mp.pruning(model_accuracy)



