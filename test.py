import numpy as np
from tree.buildList import BuildList
import torch

a=[1,2,1,2]
b=[2,1,2,2]
c=[2,1,2,1]
if __name__ == "__main__":
    # print(BuildList().compare(data1=c,data2=b))
    c = torch.randn(1,3,2,2)
    print(len(c[0][2]))
