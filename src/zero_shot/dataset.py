import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.zero_shot.config import image_dir, mask_dir, batch_size, image_size

class WHUDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(512, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        
        self.img_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        
        mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask

dataset = WHUDataset(image_dir=image_dir, mask_dir=mask_dir, image_size=image_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)