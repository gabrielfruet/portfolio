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

plt.style.use("dark_background")

def plot_correlation(x, y, title, xlabel, ylabel, color, output_path):
    plt.figure(figsize=(8, 7))
    plt.scatter(x, y, alpha=0.1, s=4, c=color, edgecolors='none') # Increase alpha and size for detail
    plt.title(title, fontsize=16, color='white', fontweight='bold')
    plt.xlabel(xlabel, fontsize=12, color='#A3A3A3')
    plt.ylabel(ylabel, fontsize=12, color='#A3A3A3')
    plt.grid(True, alpha=0.1, linestyle='--', color='white')
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#333333')
    plt.gca().spines['left'].set_color('#333333')
    plt.gca().tick_params(colors='#A3A3A3')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='black', edgecolor='none')
    print(f"Saved plot to {output_path}")
    plt.close()

# Plot L1
plot_correlation(
    ious, l1_losses, 
    f"L1 Loss Correlation (R={corr_l1:.3f})", 
    "IoU (Overlap)", 
    "L1 Loss", 
    '#EF4444', # Red-500
    "/mnt/arch_home/fruet/dev/website/gabrielfruet/src/assets/images/experiments/object_detection/l1_correlation.png"
)

# Plot CIoU
plot_correlation(
    ious, ciou_losses, 
    f"CIoU Loss Correlation (R={corr_ciou:.3f})", 
    "IoU (Overlap)", 
    "CIoU Loss", 
    '#10B981', # Emerald-500
    "/mnt/arch_home/fruet/dev/website/gabrielfruet/src/assets/images/experiments/object_detection/ciou_correlation.png"
)