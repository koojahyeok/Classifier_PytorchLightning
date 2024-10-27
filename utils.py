import os
import json
import torch
import torch.nn.functional as F

from torchvision.transforms import transforms

def get_image_transform(cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg.img_size, cfg.img_size)),
        # transforms.RandomHorizontalFlip(0.5),
        # transforms.RandomVerticalFlip(0.5),
        # processor
    ])
    
    return transform

def get_box_transform(cfg):
    # return lambda x : torch.FloatTensor(x) / cfg.img_size
    return lambda x : torch.FloatTensor([x[1]+x[2]/2, x[0]+x[3]/2, x[2], x[3]]) / cfg.img_size
    # return lambda x: torch.FloatTensor(x)