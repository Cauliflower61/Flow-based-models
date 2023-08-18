import torch
from dataset import MyData
from torch.utils.data import DataLoader
from model.GLOW import Glow
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import utils

# Parameters Setting #
device = 'cuda:0'
img_dir = 'D:/neural network/Dataset/celebA/Img/img_align_celeba/'
Checkpoint_dir = './Checkpoint/'
log_dir = './TensorboardSave/'
img_size = 32
sample_size = 64000
batch_size = 32
learning_rate = 1e-4
num_flow = 21
num_block = 3
epoch_num = 20
temp = 0.7


# Data Loading #
train_dataset = MyData(dir=img_dir, img_size=img_size, sample_size=sample_size)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

# Model Building #
net = Glow(in_channels=3, num_flow=num_flow, num_block=num_block).to(device)
optimizer = Adam(net.parameters(), lr=learning_rate)
MyLoss = net.loss_fn

# Tensorboard Setting #
writer = SummaryWriter(log_dir)
batch_num = len(train_dataloader)

# model train #
net.train()
for epoch in range(epoch_num):
    total_loss = 0
    loop = tqdm(train_dataloader, desc='train')
    for iter1, train_data in enumerate(loop):
        img = (train_data - 0.5 + torch.rand_like(train_data) / 256).to(device)
        batch = img.shape[0]

        optimizer.zero_grad()
        z_list, logdet, log_p = net(img)
        loss, logdet, log_p = MyLoss(logdet, log_p)
        total_loss += loss.data * batch
        loss.backward()
        optimizer.step()

        global_step = epoch * batch_num + iter1
        writer.add_scalar(tag="training loss", scalar_value=loss,
                          global_step=global_step)
        loop.set_description(f'Epoch [{epoch + 1}/{epoch_num}]')
        loop.set_postfix(loss=loss.data.item(), log_p=log_p.data.item(), logdet=logdet.data.item())

        if global_step % 400 == 0:
            with torch.no_grad():
                utils.save_image(
                    net.sample(3, img_size, 16, temp).cpu().data,
                    f"sample/{str(global_step + 1).zfill(6)}.png",
                    normalize=True,
                    nrow=4,
                    range=(-0.5, 0.5),
                )
    torch.save(net.state_dict(), Checkpoint_dir + 'epoch' + str(epoch + 1) + '.pkl')
writer.close()
