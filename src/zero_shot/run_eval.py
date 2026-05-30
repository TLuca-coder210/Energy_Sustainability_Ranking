import torch
import numpy as np
from src.zero_shot.config import device
from src.zero_shot.dataset import dataloader
from src.zero_shot.models import model_b1, model_b2, model_b3

def compute_iou_per_class(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(axis=1)
    predicted_set = confusion_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    
    iou = intersection / (union + 1e-10)
    return iou

def evaluate_model(model, dataloader):
    confusion_matrix = np.zeros((2, 2), dtype=np.int64)
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(pixel_values=images)
            logits = outputs.logits
            
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            
            preds = torch.argmax(upsampled_logits, dim=1)
            
            preds_flat = preds.cpu().numpy().flatten()
            masks_flat = masks.cpu().numpy().flatten()
            
            inputs = masks_flat * 2 + preds_flat
            counts = np.bincount(inputs, minlength=4)
            
            confusion_matrix += counts.reshape((2, 2))
            
    ious = compute_iou_per_class(confusion_matrix)
    return ious

models_dict = {
    "SegFormer-B1": model_b1,
    "SegFormer-B2": model_b2,
    "SegFormer-B3": model_b3
}

for name, model in models_dict.items():
    ious = evaluate_model(model, dataloader)
    print(f"--- Rezultate {name} ---")
    print(f"IoU Clasa 0 (Fundal): {ious[0]:.4f}")
    print(f"IoU Clasa 1 (Clădiri): {ious[1]:.4f}")
    print(f"Mean IoU (mIoU): {np.mean(ious):.4f}\n")