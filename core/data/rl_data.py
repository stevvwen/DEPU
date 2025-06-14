import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from torch.utils.data._utils.collate import default_collate

def custom_collate(batch):
    # If every sample is a one-element tuple, return a single tensor
    if batch and isinstance(batch[0], (list, tuple)) and len(batch[0]) == 1:
        return default_collate([item[0] for item in batch])
    return default_collate(batch)

class RLData(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.root = getattr(cfg, 'data_root', './data')
        self.val_root= getattr(cfg, 'val_data_root', './data')
        self.k = getattr(cfg, 'k', 200)
        self.batch_size = getattr(cfg, 'batch_size', 64)
        self.num_workers = getattr(cfg, 'num_workers', 0)

        # check the root path exists
        assert os.path.exists(self.root), f'{self.root} not exists'

        # check whether the root is a file or directory
        if os.path.isfile(self.root):
            state = torch.load(self.root, map_location='cpu')
            val_state = torch.load(self.val_root, map_location='cpu')

            self.param_data = state['pdata']
            self.val_data = val_state['pdata']
            self.train_layer = state['train_layer']

            self.train_dataset = TensorDataset(self.param_data)
            self.val_dataset = TensorDataset(self.val_data)
            self.test_dataset = TensorDataset(self.val_data)
        elif os.path.isdir(self.root):
            Warning(f'{self.root} is a directory, please provide a file path')
    
    def _make_loader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=shuffle,
            collate_fn=custom_collate
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset)