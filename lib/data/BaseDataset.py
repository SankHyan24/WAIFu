import torch
import torchvision
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self,opt,root=None):
        self.opt = opt
        self.root = root

        

