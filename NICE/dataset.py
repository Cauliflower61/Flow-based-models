import numpy as np
import torch
from torch.utils.data import Dataset


class MyData(Dataset):
    def __init__(self,
                 train_npy,
                 valid_npy,
                 flag="train"):
        super(MyData, self).__init__()
        # load data
        dataset = np.load(train_npy) if flag == "train" else np.load(valid_npy)
        self.dataset = torch.from_numpy(dataset)

    def __getitem__(self, item):
        return self.dataset[item, :, :].float() / 255

    def __len__(self):
        return self.dataset.shape[0]


if __name__ == '__main__':
    t_npy = './datasets/MNIST_train.npy'
    v_npy = './datasets/MNIST_valid.npy'
    train_data = MyData(t_npy, v_npy)
    print(train_data)
