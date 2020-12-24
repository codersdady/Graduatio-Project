from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from graphviz import Digraph
from pgmpy.inference import VariableElimination
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BDeuScore, K2Score, BicScore

class bayes(object):
    target_size = 0 #目标集大小
    model = None    #模型
    data_set = None #全部数据集
    test_data = None    #测试数据集
    train_data = None   #训练数据集
    input_index = None  #输入属性列表
    target = None   #目标属性值
    state_names_map = None  #属性对应数据的类别
    result_list = []    #影响度向量集合
    def buildBayes(self, type):
        if type is not 'self':
            self.model = BayesianModel([('feel', 'buying'),
                               ('feel', 'persons'),
                               ('feel', 'maint'),
                               ('feel', 'safety'),
                               ('feel', 'doors'),
                               ('feel', 'lug_boot'),
                               ('buying', 'maint'),
                               ('buying', 'safety'),
                               ('safety', 'lug_boot'),
                               ('lug_boot', 'doors')])
        else:
            hc = HillClimbSearch(self.test_data)
            model = hc.estimate(scoring_method=BicScore(self.test_data))
            self.model = BayesianModel(model.edges())

    """数据的准备，属性的初始化"""
    def prepareData(self, url):
        data_set = pd.read_excel(url)
        self.train_data, self.test_data = train_test_split(data_set, test_size=0.2, random_state=4)
        self.data_set = data_set
        self.buildBayes('self')
        self.fitBayes(data_set)
        all_index = list(data_set.columns)
        self.target = all_index[-1]
        self.input_index = all_index[0:-1]

    """构建贝叶斯网络"""
    def fitBayes(self, data, estimator=BayesianEstimator, prior_type="K2"):
        self.model.fit(data, estimator=BayesianEstimator, prior_type="BDeu")

    """绘画贝叶斯网络图"""
    def showBN(self, model, save=True):
        node_attr = dict(
            style='filled',
            shape='box',
            align='left',
            fontsize='12',
            ranksep='0.1',
            height='0.2'
        )
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
        seen = set()
        edges = model.edges()
        for a, b in edges:
            dot.edge(a, b)
        if save:
            dot.view(cleanup=True)
        return dot

    def pridict(self):
        input_index = self.input_index
        target = self.target

        infer = VariableElimination(self.model)
        self.state_names_map = infer.state_names_map
        self.target_size = len(self.state_names_map[target])
        # valuess = infer.query(['feel'], evidence={'buying': 'med'}, joint=False)
        # value = valuess.get("feel").values
        # print(value)
        for i in input_index:
            if i not in self.state_names_map.keys():
                self.result_list.append([0] * self.target_size)
            else:
                self.result_list.append(self.getIndexResult(i, target, infer))
        return np.array(self.result_list), input_index

    """获得每个属性的影响度
       [] :return
    """
    def getIndexResult(self, index, target, infer):
        data = []
        for v in self.state_names_map[index].values():
            d = infer.query([target], evidence={index: v}, joint=False)\
                .get(target).values
            data.append(d)
        data = np.array(data)
        result = np.std(data, axis=0)
        return result

