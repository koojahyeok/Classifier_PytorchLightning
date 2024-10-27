from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import os
import torch
import json
import numpy as np

import utils


class CustomDataset(Dataset):
    def __init__(self, cfg, pth, len):
        self.json_pth = pth

        self.img_transform = utils.get_image_transform(cfg)
        self.box_transform = utils.get_box_transform(cfg)

        self.img_size = cfg.img_size
        self.len = len

    def _load_image(self, img_pth):
        return Image.open(img_pth).convert("RGB")

    def __getitem__(self, idx):
        with open(self.json_pth, "r") as f:
            data = json.load(f)
        
        lbl = data['annotations'][idx]['category_id']
        lbl = torch.tensor(lbl, dtype=torch.int64)

        img = self._load_image(os.path.join(data['images'][idx]['folder'], data['images'][idx]['file_name']))
        img_processed = self.img_transform(img)

        return img_processed, lbl
    
    def __len__(self):
        return self.len
    
class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_pth = cfg.train_pth
        self.val_pth = cfg.val_pth
        self.test_pth = cfg.test_pth
        self.train_len = cfg.train_len
        self.val_len = cfg.val_len
        self.test_len = cfg.test_len
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomDataset(self.cfg, self.train_pth, self.train_len)
            self.val_dataset = CustomDataset(self.cfg, self.val_pth, self.val_len)
        
        if stage == 'test' or stage is None:
            self.test_dataset = CustomDataset(self.cfg, self.test_pth, self.test_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)