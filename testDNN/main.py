from testDNN.DNN import prepare_DNN

if __name__ == '__main__':
    preDNN = prepare_DNN()
    _, accuracy = preDNN.train_DNN()
    print(accuracy)
