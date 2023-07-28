import torch
from model.NICE import Nice
import numpy as np
import matplotlib.pyplot as plt

load_file = './Checkpoint/epoch8.pkl'
device = 'cuda:0'
pixel = 28
# load model #
net = Nice(input_dim=pixel**2, hidden_dim=1000).to(device)
net.load_state_dict(torch.load(load_file))
net.eval()
# sample #
num = 15
save_image = np.zeros(shape=(pixel*num, pixel*num))
for i in range(num):
    for j in range(num):
        z = torch.randn(size=(1, pixel*pixel)).to(device)
        out = net.decode((z, True)).reshape((pixel, pixel)).detach().cpu().numpy()
        save_image[i * pixel: (i + 1) * pixel, j * pixel: (j + 1) * pixel] = out

save_image = np.clip(save_image * 255, 0, 255)
plt.imshow(save_image, cmap='gray')
plt.show()
