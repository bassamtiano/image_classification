import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader

import numpy as np

import pytorch_lightning as pl

import matplotlib.pyplot as plt

class MultiClassPreprocessor(pl.LightningDataModule):
    
    def __init__(self,
                 batch_size):
        super(MultiClassPreprocessor, self).__init__()
        
        self.classes = (
            'plane',
            'car',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck'
        )
        
        self.batch_size = batch_size
        
    def show_image(self, img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    
    def load_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        train_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                                download=True, transform=transform)
        
        train_len = int(len(train_set) * 0.8)
        val_len = int(len(train_set) - train_len)
        
        
        train_set, val_set = torch.utils.data.random_split(train_set, [train_len, val_len])
        
        test_set = torchvision.datasets.CIFAR10(root='./data/cifar10',
                                               train=False,
                                               download=True,
                                               transform=transform)
        
        return train_set, val_set, test_set
        # return train_set, test_set
        
    def setup(self, stage = None):
        train_set, val_set, test_set = self.load_data()
        # train_set, test_set = self.load_data()
        
        if stage == "fit":
            self.train_data = train_set
            self.val_data = val_set
        elif stage == "test":
            self.test_data = test_set
    
    def train_dataloader(self):
        return  DataLoader(self.train_data, 
                           batch_size=self.batch_size,
                           shuffle=True, 
                           num_workers=2)
    
    def val_dataloader(self):
        return  DataLoader(self.val_data, 
                           batch_size=self.batch_size,
                           shuffle=False,
                           num_workers=2)
        
    def test_dataloader(self):
        return  DataLoader(self.test_data, 
                           batch_size=self.batch_size,
                           shuffle=False,
                           num_workers=2)

    
# if __name__ == '__main__':
#     mcp = MultiClassPreprocessor(batch_size = 10)
#     mcp.load_data()