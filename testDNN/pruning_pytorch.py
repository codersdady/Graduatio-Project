import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

if __name__ == '__main__':
    model = torch.load("./初始模型.pkl")

    # print(list(module.named_parameters()))
    parameters_to_prune = (
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )

    print(
        "Sparsity in conv1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc1.weight == 0))
            / float(model.fc1.weight.nelement())
        )
    )
    print(
        "Sparsity in conv1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc2.weight == 0))
            / float(model.fc2.weight.nelement())
        )
    )
    module = model.fc1
    # print(list(module.weight))
    # print(list(module.named_buffers()))
    print(list(module.named_parameters()))
    prune.remove(module, 'weight')
    print(list(module.named_parameters()))

