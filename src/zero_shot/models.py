import os
import torch
from transformers import SegformerForSemanticSegmentation
from src.zero_shot.config import device

os.system("mkdir -p mit-b1-fix")
os.system("curl -L https://huggingface.co/nvidia/mit-b1/resolve/main/config.json -o mit-b1-fix/config.json")
os.system("curl -L https://huggingface.co/nvidia/mit-b1/resolve/main/pytorch_model.bin -o mit-b1-fix/pytorch_model.bin")

model_b1 = SegformerForSemanticSegmentation.from_pretrained(
    "./mit-b1-fix",
    num_labels=2,
    ignore_mismatched_sizes=True
)
model_b1.to(device)
model_b1.eval()

os.system("mkdir -p mit-b2-fix")
os.system("curl -L https://huggingface.co/nvidia/mit-b2/resolve/main/config.json -o mit-b2-fix/config.json")
os.system("curl -L https://huggingface.co/nvidia/mit-b2/resolve/main/pytorch_model.bin -o mit-b2-fix/pytorch_model.bin")

model_b2 = SegformerForSemanticSegmentation.from_pretrained(
    "./mit-b2-fix",
    num_labels=2,
    ignore_mismatched_sizes=True
)
model_b2.to(device)
model_b2.eval()

os.system("mkdir -p mit-b3-fix")
os.system("curl -L https://huggingface.co/nvidia/mit-b3/resolve/main/config.json -o mit-b3-fix/config.json")
os.system("curl -L https://huggingface.co/nvidia/mit-b3/resolve/main/pytorch_model.bin -o mit-b3-fix/pytorch_model.bin")

model_b3 = SegformerForSemanticSegmentation.from_pretrained(
    "./mit-b3-fix",
    num_labels=2,
    ignore_mismatched_sizes=True
)
model_b3.to(device)
model_b3.eval()