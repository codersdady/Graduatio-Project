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
LEARNING_RATE = 0.001  #0.001
EPOCH = 300      #400 ;
BATCH_SIZE = 100    #15
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

    def train_DNN(self, trainData, testData, target, target_size):
        # we want to use GPU if we have one
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_data, test_label, train_data, train_label,test = self.genarate_data(device, trainData, testData, target)
        input_size = len(trainData.columns) - 1
        HIDDEN_UNITS = input_size * 2 + 1
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





