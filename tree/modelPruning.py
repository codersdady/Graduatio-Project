from tree.buildList import BuildList


class modelPruning(object):
    model_size = None    #模型大小
    model_accuracy = None #模型准确率
    data_set = None #数据集
    train_set = None    #训练数据集
    test_set = None #测试数据集
    head = None #最优前沿节点指针
    input_index = None  #输入属性对应名
    def __init__(self, model_size, model_accuracy, data_set, train_set, test_set, head, input_index):
        self.model_size = model_size
        self.model_accuracy = model_accuracy
        self.data_set = data_set
        self.train_set = train_set
        self.test_set = test_set
        self.head = head
        self.input_index = input_index

    def pruning(self):
        accuracy = 100
        while len(self.head.data) != 0 and accuracy > self.model_accuracy:
            trainData = self.train_set
            testData = self.test_set





