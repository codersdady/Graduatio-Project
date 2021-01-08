import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



# HIDDEN_UNITS = 15
LEARNING_RATE = 0.005  #0.001
EPOCH = 800      #400 ;
BATCH_SIZE = 75    #15
# Using 2 hidden layers  dnn网络
# input_size = 14
# num_classes = 2

class DNN(nn.Module):    #dnn网络
    def __init__(self, input_size, num_classes, HIDDEN_UNITS):
        super().__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_UNITS)
        # self.fc2 = nn.Linear(25, 10)
        self.fc2 = nn.Linear(HIDDEN_UNITS, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        y_hat = self.fc2(x)
        # out = F.dropout(y_hat, p=0.2)
        return y_hat

class prepare_DNN():
    def genarate_data(self,device, train, test, target):    #准备数据
        # train = pd.read_csv('./car.data')
        # test = pd.read_csv('./test.txt')


        # del  data['Unnamed: 13']
        # del data['OBJECTID']

        # train, test = train_test_split(data, test_size=0.2, random_state=4)
        #print (test)

        # print(all)
        train_label=train[target].values
        train_data=train.drop([target],axis=1).values
        test_label = test[target].values
        test_data = test.drop([target],axis=1).values
        test_data = torch.Tensor(test_data).to(device)
        test_label = torch.LongTensor(test_label).to(device)
        train_data = torch.Tensor(train_data).to(device)
        train_label = torch.LongTensor(train_label).to(device)
        return test_data, test_label, train_data, train_label,test

    def save_model(self, model, url):
        torch.save(model, url)

    def train_DNN(self):
        # we want to use GPU if we have one
        data = pd.read_excel("./q.xls")
        trainData, testData = train_test_split(data, test_size=0.2, random_state=4)
        target = 'target'
        #收入（data.xlsx）
        #13, 2, 8, 1, 0, 6, 9, 12, 5, 7, 4, 3, 11, 10
        # del trainData['nc']
        # del testData['nc']
        # del trainData['fn']
        # del testData['fn']
        # #
        # del trainData['race']
        # del testData['race']
        # #
        # del trainData['wc']
        # del testData['wc']
        # #
        # del trainData['age']
        # del testData['age']
        # #
        # del trainData['occ']
        # del testData['occ']
        # #
        # del trainData['sex']
        # del testData['sex']
        # #
        # del trainData['hpw']
        # del testData['hpw']
        # #
        # del trainData['mstatus']
        # del testData['mstatus']
        # #
        # del trainData['rela']
        # del testData['rela']
        # #
        # del trainData['edcnum']
        # del testData['edcnum']
        # #
        # del trainData['edc']
        # del testData['edc']
        # #
        # del trainData['cl']
        # del testData['cl']
        #
        # del trainData['cg']
        # del testData['cg']


        #选举（ElectionData.csv）
        #2, 4, 5, 7, 9, 10, 11, 12, 14, 16, 17, 18, 21, 22, 23
        #0, 3
        #1
        #8, 13, 15
        #6
        #19, 20, 24

        # del trainData['availableMandates']
        # del testData['availableMandates']
        # #
        # del trainData['numParishesApproved']
        # del testData['numParishesApproved']
        # #
        # del trainData['blankVotes']
        # del testData['blankVotes']
        # #
        # del trainData['nullVotes']
        # del testData['nullVotes']
        # #
        # del trainData['votersPercentage']
        # del testData['votersPercentage']
        # #
        # del trainData['subscribedVoters']
        # del testData['subscribedVoters']
        # #
        # del trainData['totalVoters']
        # del testData['totalVoters']
        # #
        # del trainData['pre.blankVotes']
        # del testData['pre.blankVotes']
        # #
        # del trainData['pre.nullVotes']
        # del testData['pre.nullVotes']
        # #
        # del trainData['pre.votersPercentage']
        # del testData['pre.votersPercentage']
        # #
        # del trainData['pre.subscribedVoters']
        # del testData['pre.subscribedVoters']
        # #
        # del trainData['pre.totalVoters']
        # del testData['pre.totalVoters']
        # #
        # del trainData['Percentage']
        # del testData['Percentage']
        # #
        # del trainData['validVotesPercentage']
        # del testData['validVotesPercentage']
        # #
        # del trainData['Votes']
        # del testData['Votes']
        # #
        # del trainData['territoryName']
        # del testData['territoryName']
        # #
        # del trainData['numParishes']
        # del testData['numParishes']
        # #
        # del trainData['totalMandates']
        # del testData['totalMandates']
        # #
        # del trainData['nullVotesPercentage']
        # del testData['nullVotesPercentage']
        # #
        # del trainData['pre.blankVotesPercentage']
        # del testData['pre.blankVotesPercentage']
        # #
        # del trainData['pre.nullVotesPercentage']
        # del testData['pre.nullVotesPercentage']
        # #
        # del trainData['blankVotesPercentage']
        # del testData['blankVotesPercentage']
        # #
        # del trainData['Party']
        # del testData['Party']
        # #
        # del trainData['Mandates']
        # del testData['Mandates']
        #
        # del trainData['Hondt']
        # del testData['Hondt']

        #蘑菇（q.xls）
        #14, 1, 16, 0, 9, 19, 15, 5, 2, 6, 3, 20, 13, 7, 10, 17, 12, 8, 11, 18, 14
        del trainData['veil-type']
        del testData['veil-type']
        #
        del trainData['cap-sur']
        del testData['cap-sur']

        del trainData['ring-num']
        del testData['ring-num']

        del trainData['cap-sha']
        del testData['cap-sha']

        del trainData['stalk-sha']
        del testData['stalk-sha']

        del trainData['popu']
        del testData['popu']

        del trainData['veil-color']
        del testData['veil-color']

        del trainData['gill-atta']
        del testData['gill-atta']

        del trainData['cap-col']
        del testData['cap-col']

        del trainData['gill-spa']
        del testData['gill-spa']

        del trainData['bruises']
        del testData['bruises']

        del trainData['habitat']
        del testData['habitat']

        del trainData['scbr']
        del testData['scbr']

        del trainData['gill-size']
        del testData['gill-size']

        del trainData['ssar']
        del testData['ssar']

        del trainData['ring-typ']
        del testData['ring-typ']

        del trainData['scar']
        del testData['scar']

        del trainData['gill-color']
        del testData['gill-color']

        del trainData['ssbr']
        del testData['ssbr']

        del trainData['spc']
        del testData['spc']

        del trainData['odor']
        del testData['odor']

        target_size = 4

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_data, test_label, train_data, train_label,test = self.genarate_data(device, trainData, testData, target)
        input_size = len(trainData.columns) - 1
        HIDDEN_UNITS = input_size * 2 + 1
        # HIDDEN_UNITS = 29
        num_classes = target_size

        # prepare the data loader
        training_set = Data.TensorDataset(train_data,
                                          train_label)
        training_loader = Data.DataLoader(dataset=training_set,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)
        testing_set = Data.TensorDataset(test_data,
                                         test_label)
        testing_loader = Data.DataLoader(dataset=testing_set,
                                         batch_size=BATCH_SIZE,
                                         shuffle=False)
        model = DNN(input_size, num_classes, HIDDEN_UNITS).to(device)
        # using crossentropy loss on classification problem
        criterion = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        accuracy = 0
        for epoch in range(EPOCH):
            correct_train = 0
            total_train = 0
            for (data, label) in training_loader:
                pred_label = model(data)
                loss = criterion(pred_label, label)
                optim.zero_grad()
                loss.backward()
                optim.step()
                _, answer = torch.max(pred_label.data, 1)

                total_train += label.size(0)
                correct_train += (answer == label).sum()
            print('Epoch {:3d} Accuracy on training data: {}% ({}/{})'
                  .format(epoch, (100 * correct_train / total_train), correct_train, total_train))
            # pytorch 0.4 feature, not calculate grad on test set
            # 预测阶段，不跟新权值
            with torch.no_grad():
                correct_test = 0
                total_test = 0
                for (data, label) in testing_loader:
                    pred_label = model(data)
                    _, answer = torch.max(pred_label.data, 1)
                    total_test += label.size(0)
                    correct_test += (answer == label).sum()
                print('          Accuracy on testing data: {}% ({}/{})'
                      .format((100 * correct_test / total_test), correct_test, total_test))
                accuracy = (100 * correct_test / total_test)

        return model, accuracy





