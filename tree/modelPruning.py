from tree.buildList import BuildList
from tree.DNN import prepare_DNN
import os

class modelPruning(object):
    model_size = 0    #模型大小阈值
    model_accuracy = None #模型准确率阈值
    m_size = model_size #当前模型大小
    data_set = None #数据集
    train_set = None    #训练数据集
    test_set = None #测试数据集
    head = None #最优前沿节点指针
    input_index = None  #输入属性对应名
    target = None   #模型目标属性
    url = 'E:\workspace\Graduatio-Project\model\\'  #模型保存路径
    i = 0
    target_size = 0

    def __init__(self, model_size, model_accuracy, data_set, train_set, test_set, head, input_index, target, target_size):
        self.m_size = model_size
        self.model_accuracy = model_accuracy-0.2
        self.data_set = data_set
        self.train_set = train_set
        self.test_set = test_set
        self.head = head
        self.input_index = input_index
        self.target = target
        self.target_size = target_size
        self.model_size = self.m_size / 2

    def getSize(self, filePath):
        fsize = os.path.getsize(filePath)
        return fsize

    def pruning(self, accuracy):
        accuracy = 100
        trainData = self.train_set
        testData = self.test_set
        b_model = None
        b_accuracy = 0
        model = None
        preDnn = prepare_DNN()


        while len(self.head.data) != 0 and accuracy > self.model_accuracy:
            self.train_set = trainData
            self.test_set = testData
            b_model = model
            b_accuracy = accuracy

            for i in self.head.index:
                del trainData[self.input_index[i]]
                del testData[self.input_index[i]]
            model, accuracy = preDnn.train_DNN(trainData, testData, self.target, self.target_size)
            self.head = self.head.next
            self.i += 1

        if self.i == 0:
            print("无法压缩模型！")
            return
        if accuracy < self.model_accuracy :
            print("模型的准确率为：" + str(b_accuracy))
            url = self.url + str(self.i - 1) + '.pkl'
            preDnn.save_model(b_model, url)
            self.m_size = self.getSize(url)
            print("模型的大小为：" + str(self.getSize(url)))
        else:
            url = self.url + str(self.i) + '.pkl'
            preDnn.save_model(model, url)
            self.m_size = self.getSize(url)
            print("模型准确率为：" + accuracy)
            print("模型的大小为：" + self.getSize(url))







