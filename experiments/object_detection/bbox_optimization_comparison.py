import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import complete_box_iou_loss, box_iou
from PIL import Image
from pathlib import Path

# --- Configuration ---
CANVAS_SIZE = 600
BG_COLOR = (26, 26, 26) 
GT_COLOR = (0, 255, 0)   # Green
L1_COLOR = (255, 50, 50) # Red
CIOU_COLOR = (255, 255, 50) # Yellow
TEXT_COLOR = (255, 255, 255)

DURATION = 7.0
FPS = 10
NUM_FRAMES = int(DURATION * FPS)
OUTPUT_DIR = Path("src/assets/images/experiments/object_detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NORM_FACTOR = float(CANVAS_SIZE)

# --- THE SETUP: ASPECT RATIO TEST ---
# Target: A Perfect Square (100x100)
# We place two identical targets to race L1 vs CIoU side-by-side
target_bboxes_raw = torch.tensor([
    [150.0, 300.0, 100.0, 100.0], # Target for L1 (Left)
    [450.0, 300.0, 100.0, 100.0], # Target for CIoU (Right)
])
target_bboxes = target_bboxes_raw / NORM_FACTOR

# Prediction: WRONG SHAPE (Wide and Short: 180x20)
# This has the same center, so L1 is only driven by edge differences.
# CIoU should detect the aspect ratio is wrong.
start_bboxes_raw = torch.tensor([
    [250.0, 300.0, 200.0, 20.0], # Pred for L1 (Starts Wide)
    [550.0, 300.0, 200.0, 20.0], # Pred for CIoU (Starts Wide)
])
start_bboxes = start_bboxes_raw / NORM_FACTOR

# Init Parameters
bboxes_pred = start_bboxes.clone().detach().requires_grad_(True)

# We optimize them independently to track history
# Index 0 -> L1, Index 1 -> CIoU
optimizer = optim.Adam([bboxes_pred], lr=0.02) 

# --- Helper Functions ---
def xywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    w = torch.clamp(w, min=0.01) # Safety clamp to prevent negative box size
    h = torch.clamp(h, min=0.01)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)

def to_pixel_rect(box_xywh):
    box_xyxy = xywh_to_xyxy(box_xywh)
    coords = (box_xyxy * NORM_FACTOR).detach().clamp(0, CANVAS_SIZE).numpy().astype(int)
    return coords

# --- Main Loop ---
frames = []
print(f"Generating Aspect Ratio Race...")

for i in range(NUM_FRAMES):
    optimizer.zero_grad()
    
    # --- Split the batch to apply different loss functions ---
    # Box 0: L1 Loss
    l1_pred = bboxes_pred[0:1]
    l1_target = target_bboxes[0:1]
    loss_l1 = nn.functional.l1_loss(l1_pred, l1_target) * 5 # Scale up L1 so it's competitive in magnitude
    
    # Box 1: CIoU Loss
    ciou_pred_xyxy = xywh_to_xyxy(bboxes_pred[1:2])
    ciou_target_xyxy = xywh_to_xyxy(target_bboxes[1:2])
    loss_ciou = complete_box_iou_loss(ciou_pred_xyxy, ciou_target_xyxy, reduction='mean')

    # Combine losses (just summing them lets backward work on both)
    total_loss = loss_l1 + loss_ciou
    total_loss.backward()
    optimizer.step()

    # --- Rendering ---
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), BG_COLOR[::-1], dtype=np.uint8)
    
    # Compute current IoU for display
    current_xyxy = xywh_to_xyxy(bboxes_pred)
    target_xyxy = xywh_to_xyxy(target_bboxes)
    ious = box_iou(current_xyxy, target_xyxy).diag()

    # Draw Divider
    cv2.line(canvas, (300, 0), (300, 600), (50, 50, 50), 2)
    
    # Draw Left Side (L1)
    t_rect_l = to_pixel_rect(target_bboxes[0])
    p_rect_l = to_pixel_rect(bboxes_pred[0])
    cv2.rectangle(canvas, (t_rect_l[0], t_rect_l[1]), (t_rect_l[2], t_rect_l[3]), GT_COLOR[::-1], 2)
    cv2.rectangle(canvas, (p_rect_l[0], p_rect_l[1]), (p_rect_l[2], p_rect_l[3]), L1_COLOR[::-1], 3)
    cv2.putText(canvas, "L1 Loss", (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, L1_COLOR[::-1], 2)
    cv2.putText(canvas, f"IoU: {ious[0]:.2f}", (100, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)

    # Draw Right Side (CIoU)
    t_rect_r = to_pixel_rect(target_bboxes[1])
    p_rect_r = to_pixel_rect(bboxes_pred[1])
    cv2.rectangle(canvas, (t_rect_r[0], t_rect_r[1]), (t_rect_r[2], t_rect_r[3]), GT_COLOR[::-1], 2)
    cv2.rectangle(canvas, (p_rect_r[0], p_rect_r[1]), (p_rect_r[2], p_rect_r[3]), CIOU_COLOR[::-1], 3)
    cv2.putText(canvas, "CIoU Loss", (380, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, CIOU_COLOR[::-1], 2)
    cv2.putText(canvas, f"IoU: {ious[1]:.2f}", (400, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)

    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    frames.append(Image.fromarray(canvas_rgb))

# Save GIF
output_path = OUTPUT_DIR / "l1_vs_ciou_aspect_ratio.gif"
frames[0].save(output_path, save_all=True, append_images=frames[1:], optimize=False, duration=1000/FPS, loop=0)
print(f"Saved to {output_path}")