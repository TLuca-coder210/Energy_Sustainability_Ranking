import os
import torch
import kagglehub
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np

dataset_path = kagglehub.dataset_download("sengulgs/whu-building-dataset")
image_dir = os.path.join(dataset_path, "WHU", "test", "Image")
mask_dir = os.path.join(dataset_path, "WHU", "test", "Mask")

batch_size = 4
image_size = (512, 512)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")