import torch
from model.GLOW import Glow
from torchvision import utils

# Parameters Setting #
load_file = './Checkpoint/epoch20.pkl'
num_flow = 21
num_block = 3
sample_size = 4
inchannels = 3
image_size = 32
sigma = 0.7

net = Glow(in_channels=3, num_flow=num_flow, num_block=num_block).to("cuda:0")
net.load_state_dict(torch.load(load_file))

cnt = 0

with torch.no_grad():
    a = net.sample(inc=inchannels, image_size=image_size, sample_size=16, sigma=sigma)
    utils.save_image(a.cpu().data, f"./generate.png", normalize=True, nrow=sample_size, range=(-0.5, 0.5))
