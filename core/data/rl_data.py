from torch.utils.data import Dataset, DataLoader
from yarg import get
from core.data.base import DataBase
from core.data.parameters import PData, Parameters
import torch
import os

class RLData(DataBase):
    def __init__(self, cfg, **kwargs):
        super(RLData, self).__init__(cfg, **kwargs)
        self.root = getattr(self.cfg, 'data_root', './data')
        self.val_root= getattr(self.cfg, 'val_data_root', './data')
        self.k = getattr(self.cfg, 'k', 200)
        self.batch_size = getattr(self.cfg, 'batch_size', 64)


        # check the root path is  exist or not
        assert os.path.exists(self.root), f'{self.root} not exists'

        # check the root is a directory or file
        if os.path.isfile(self.root):
            state = torch.load(self.root, map_location='cpu')

            val_state= torch.load(self.val_root, map_location='cpu')

            self.param_data = state['pdata']
            self.val_data= val_state['pdata']
            self.train_layer = state['train_layer']

        elif os.path.isdir(self.root):
            pass
    
    def __len__(self):
        return len(self.param_data)

    
    def get_train_layer(self):
        return self.train_layer

    @property
    def train_dataset(self):
        data= Parameters(self.param_data, self.k, split='train')
        return data

    @property
    def val_dataset(self):
        return Parameters(self.val_data, self.k, split='val')

    @property
    def test_dataset(self):
        return Parameters(self.param_data, self.k, split='test')
