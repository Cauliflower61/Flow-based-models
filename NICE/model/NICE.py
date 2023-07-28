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
                               nn.ReLU())]
        for i in range(3):
            layer.append(nn.Sequential(nn.Linear(output_dim, output_dim),
                                       nn.ReLU()))
        layer.append(nn.Sequential(nn.Linear(output_dim, half_dim),
                                   nn.ReLU()))
        self.conv = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv(x)


# concat two parts into the vector #
def ConcatVector(part1, part2):
    batch, dim = part1.shape[0], 2 * part1.shape[1]
    vector = torch.zeros(size=(batch, dim)).to('cuda:0')
    vector[:, 1::2] = part1
    vector[:, 0::2] = part2
    return vector


# shuffle the vector #
class Shuffle(nn.Module):
    def __init__(self):
        super(Shuffle, self).__init__()

    def forward(self, x):
        vector, isinverse = x[0], x[1]
        part1, part2 = SplitVector(vector)
        return ConcatVector(part2, part1), isinverse


# Build the block #
class Block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Block, self).__init__()
        self.m = BasicModel(input_dim, output_dim)

    def forward(self, x):
        vector, isinverse = x[0], x[1]
        part1, part2 = SplitVector(vector)
        out = ConcatVector(part1, part2 + self.m(part1)) if isinverse \
            else ConcatVector(part1, part2 - self.m(part1))
        return out, isinverse


# build the network #
class Nice(nn.Module):
    def __init__(self, input_dim, hidden_dim=1000):
        super(Nice, self).__init__()
        self.layers = [Block(input_dim, hidden_dim)]
        for count in range(3):
            self.layers.append(Shuffle())
            self.layers.append(Block(input_dim, hidden_dim))
        self.scale = nn.Parameter(torch.ones(size=(1, input_dim)))
        self.encoder = nn.Sequential(*self.layers)

    def forward(self, x):
        vector, _ = self.encoder(x)
        return vector * torch.exp(self.scale)

    def decode(self, x):
        vector, isinverse = x[0], x[1]
        vector = vector * torch.exp(-self.scale)
        decoder = nn.Sequential(*list(reversed(self.layers)))
        out, _ = decoder((vector, isinverse))
        return out


if __name__ == "__main__":
    vector = torch.arange(20, dtype=torch.float).reshape((2, -1))
    net = Nice(input_dim=10, hidden_dim=100)
    out = net.decode((vector, False))
    print(out)
