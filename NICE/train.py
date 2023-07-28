import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MyData
from model.NICE import Nice
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# loss function of per batch #
class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

    def forward(self, y_pre, log_s):
        return 1/2 * torch.mean(torch.sum(y_pre ** 2, dim=1)) - torch.sum(log_s)


# parameters definition #
t_npy = './datasets/MNIST_train.npy'
v_npy = './datasets/MNIST_valid.npy'
Checkpoint_dir = './Checkpoint/'
log_dir = './TensorBoardSave/NICE/'
device = 'cuda:0'
batch_size = 60
learning_rate = 0.001
epoch_num = 30

# load data #
train_dataset = MyData(t_npy, v_npy)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
# TensorBoard #
writer = SummaryWriter(log_dir)
batch_num = len(train_dataloader)

# generate model #
net = Nice(input_dim=28*28, hidden_dim=1000).to(device)
MyLoss = LossFunction()
optim = torch.optim.Adam(net.parameters(), lr=learning_rate)


# model train #
net.train()
for epoch in range(epoch_num):
    total_loss = 0
    iter1 = 0
    loop = tqdm(train_dataloader, desc='Train')
    for train_data in loop:
        batch = train_data.shape[0]
        train_data = train_data.view(batch, -1).to(device)
        optim.zero_grad()
        z = net((train_data, False))
        loss = MyLoss(z, net.scale)
        total_loss += loss.data * batch
        loss.backward()
        optim.step()
        writer.add_scalar(tag="training loss", scalar_value=loss,
                          global_step=epoch * batch_num + iter1)
        loop.set_description(f'Epoch [{epoch + 1}/{epoch_num}]')
        loop.set_postfix(loss=loss.data.item())
        iter1 += 1

    torch.save(net.state_dict(), Checkpoint_dir + 'epoch' + str(epoch+1) + '.pkl')
writer.close()