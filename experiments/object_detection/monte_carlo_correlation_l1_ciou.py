import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import box_iou, complete_box_iou_loss
from scipy.stats import pearsonr

def generate_random_boxes(num_samples=2000):
    # Ground Truth: Fixed square at center
    gt_box = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)

    # Generate random predictions around the GT
    # We randomize center (x,y) and size (w,h) independently
    # This creates the "shape vs location" confusion L1 suffers from
    pred_boxes = []

    for _ in range(num_samples):
        cx = 100 + np.random.uniform(-40, 40)
        cy = 100 + np.random.uniform(-40, 40)
        w = 100 + np.random.uniform(-60, 60) # High variance in width
        h = 100 + np.random.uniform(-60, 60) # High variance in height

        pred_boxes.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    return gt_box, torch.tensor(pred_boxes, dtype=torch.float32)

# 1. Generate Data
gt, preds = generate_random_boxes(50000)

# 2. Calculate Metrics
# L1 Loss (normalized by image scale approx to keep it readable)
l1_losses = torch.nn.functional.l1_loss(preds, gt.expand_as(preds), reduction='none').mean(dim=1).numpy()

# CIoU Loss
ciou_losses = complete_box_iou_loss(preds, gt.expand_as(preds), reduction='none').numpy()

# Actual IoU (The Truth)
ious = box_iou(preds, gt).squeeze().numpy()

# 3. Calculate Correlation (Pearson R)
# Note: We invert L1/CIoU because Loss goes down as IoU goes up.
# A perfect score would be -1.0.
corr_l1, _ = pearsonr(l1_losses, ious)
corr_ciou, _ = pearsonr(ciou_losses, ious)

# --- Visualization ---
# --- Visualization ---

# Plot L1
plt.figure(figsize=(8, 7))
plt.scatter(ious, l1_losses, alpha=0.05, s=2, c='#FF3B30') # Clean Red
plt.title(f"Pearson Correlation: {corr_l1:.3f}", fontsize=14)
plt.xlabel("IoU")
plt.ylabel("L1 Loss")
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Save L1
l1_output_path = "/mnt/arch_home/fruet/dev/website/gabrielfruet/src/assets/images/experiments/object_detection/l1_correlation.png"
plt.savefig(l1_output_path, dpi=300)
print(f"Saved L1 plot to {l1_output_path}")
plt.close()

# Plot CIoU
plt.figure(figsize=(8, 7))
plt.scatter(ious, ciou_losses, alpha=0.05, s=2, c='#34C759') # Clean Green
plt.title(f"Pearson Correlation: {corr_ciou:.3f}", fontsize=14)
plt.xlabel("IoU")
plt.ylabel("CIoU Loss")
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Save CIoU
ciou_output_path = "/mnt/arch_home/fruet/dev/website/gabrielfruet/src/assets/images/experiments/object_detection/ciou_correlation.png"
plt.savefig(ciou_output_path, dpi=300)
print(f"Saved CIoU plot to {ciou_output_path}")
plt.close()