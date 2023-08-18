import random
from torch.utils.data import Dataset
from os import listdir
from os.path import join
import cv2
from torchvision import transforms


class MyData(Dataset):
    def __init__(self, dir, img_size, sample_size):
        super().__init__()
        self.imgdir = dir
        self.imgList = listdir(dir)
        random.shuffle(self.imgList)
        self.imgList = self.imgList[:sample_size]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
        ])

    def __getitem__(self, item):
        path = join(self.imgdir, self.imgList[item])
        b, g, r = cv2.split(cv2.imread(path, cv2.IMREAD_UNCHANGED))
        image = self.transform(cv2.merge([r, g, b]))
        return image

    def __len__(self):
        return len(self.imgList)


if __name__ == "__main__":
    imageDir = 'D:/neural network/Dataset/celebA/Img/img_align_celeba/'
    dataset = MyData(imageDir, img_size=32, sample_size=72000)
    img = dataset[0].numpy().transpose(1, 2, 0)
    cv2.imshow('img', img)
    cv2.waitKey()