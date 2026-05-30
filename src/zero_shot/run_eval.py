import os
import cv2
import torch
import numpy as np
from tqdm.auto import tqdm

from src.zero_shot.config import device
from src.zero_shot.dataset import dataloader
from src.zero_shot.models import model_b1, model_b2, model_b3

def calculate_iou_per_zone(pred, target):
    intersection_1 = np.logical_and(pred == 1, target == 1).sum()
    union_1 = np.logical_or(pred == 1, target == 1).sum()
    iou_building = 1.0 if union_1 == 0 else intersection_1 / union_1

    intersection_0 = np.logical_and(pred == 0, target == 0).sum()
    union_0 = np.logical_or(pred == 0, target == 0).sum()
    iou_background = 1.0 if union_0 == 0 else intersection_0 / union_0

    return iou_background, iou_building

def evaluate_segformer(model, dataloader):
    total_iou_bg = 0.0
    total_iou_bld = 0.0
    count = 0

    with torch.no_grad():
        for imgs, masks in tqdm(dataloader):
            imgs = imgs.to(device)
            outputs = model(pixel_values=imgs)

            logits = torch.nn.functional.interpolate(
                outputs.logits, size=(512, 512), mode="bilinear", align_corners=False
            )
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true_masks = masks.numpy()

            for i in range(imgs.size(0)):
                iou_bg, iou_bld = calculate_iou_per_zone(preds[i], true_masks[i])
                total_iou_bg += iou_bg
                total_iou_bld += iou_bld
                count += 1

    mean_iou_bg = total_iou_bg / count
    mean_iou_bld = total_iou_bld / count
    mIoU = (mean_iou_bg + mean_iou_bld) / 2
    print(mIoU)

evaluate_segformer(model_b1, dataloader)
evaluate_segformer(model_b2, dataloader)
evaluate_segformer(model_b3, dataloader)