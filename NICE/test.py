import torch
import torch.nn as nn


# split the vector into two parts #
def SplitVector(vector):               # input: [batch_size, vector_dim]
    part1 = vector[:, 1::2]            # odd list
    part2 = vector[:, 0::2]            # even list
    return [part1, part2]


# basic model #
class BasicModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicModel, self).__init__()
        half_dim = int(input_dim/2)
        layer = [nn.Sequential(nn.Linear(half_dim, output_dim),
                               nn.ReLU(), nn.BatchNorm1d(output_dim),
                               nn.Dropout())]
        for i in range(3):
            layer.append(nn.Sequential(nn.Linear(output_dim, output_dim),
                                       nn.ReLU(), nn.BatchNorm1d(output_dim),
                                       nn.Dropout()))
        layer.append(nn.Sequential(nn.Linear(output_dim, half_dim),
                                   nn.Tanh()))
        self.conv = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv(x)


# concat two parts into the vector #
def ConcatVector(part1, part2):
    batch, dim = part1.shape[0], 2 * part1.shape[1]
    vector = torch.zeros(size=(batch, dim))
    vector[:, 1::2] = part1
    vector[:, 0::2] = part2
    return vector


class Block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Block, self).__init__()
        self.m = BasicModel(input_dim, output_dim)

    def forward(self, vector, isinverse=False):
        part1, part2 = SplitVector(vector)
        if net.train():
            print(1+1)
        else:
            print(1-1)
        out = ConcatVector(part1, part2 + self.m(part1)) if net.train() \
            else ConcatVector(part1, part2 - self.m(part1))
        return out


# vector = torch.arange(20, dtype=torch.float).reshape((2, -1))
# net = Block(input_dim=10, output_dim=10)
# out = net(vector, isinverse=False)
# num_params = sum(param.numel() for param in net.parameters())
# # out = net(vector, isinverse=False)
# print(out)
# a = torch.tensor([[1,2],[3,4]])
# print(torch.sum(a, dim=1))
a = [[1, 2],
     [3, 4]]
print(list(reversed(a)))
