from torchpruner.attributions import RandomAttributionMetric  # or any of the methods above
from torchpruner.pruner import Pruner

import torch
if __name__ == '__main__':

    model = torch.load("./初始模型.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attr = RandomAttributionMetric(model, data_generator, criterion, device)
    for module in model.children():
        if len(list(module.children())) == 0:  # leaf module
            scores = attr.run(module)
            print (scores)