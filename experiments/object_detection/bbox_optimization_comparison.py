import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.ops import box_iou, box_convert, complete_box_iou_loss
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import argparse
import copy
import math

# --- Configuration ---
# Standard output
OUTPUT_DIR = Path("src/assets/images/experiments/object_detection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Colors
GT_COLOR = (0, 255, 0)      # Green
L1_COLOR = (255, 100, 100)  # Reddish
CIOU_COLOR = (50, 255, 255) # Yellow/Cyan
TEXT_COLOR = (255, 255, 255)

# Settings - INCREASED FRAMES/LR to ensure convergence
LR = 1e-3

# --- Helper ---
def load_visdrone_ann(path):
    boxes = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            x, y, w, h = map(float, parts[:4])
            if w > 0 and h > 0:
                boxes.append([x, y, x+w, y+h])
    return torch.tensor(boxes, dtype=torch.float32)

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb

# --- Model ---
# Use a very simple fully convolutional network that can overfit easily
# --- Model ---
# Radically simplified: Learn coordinates directly per pixel.
# This ensures we see the OPTIMIZATION process of the loss function, 
# unhindered by the capacity or dynamics of a Convolutional Neural Network.
class DirectGridModel(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        # learnable logits for every position: [1, 4, h, w]
        # Initialize to 1.0 (approx 8px after denorm, or 2.7 after exp if treated as log)
        # We will use raw values with ReLU to avoid exploding Exp
        self.reg_logits = nn.Parameter(torch.ones(1, 4, h, w) * 1.0) 
        self.ctr_logits = nn.Parameter(torch.zeros(1, 1, h, w)) # 0.5 sigmoid

    def forward(self, x):
        # We ignore input X! We are optimizing the grid slots directly.
        # This makes the visualization pure "Loss Function Optimization"
        
        # Regression: ReLU + offset to ensure positive >= 0.1
        reg = torch.relu(self.reg_logits) + 0.1
        centerness = self.ctr_logits
        return reg, centerness

# --- Coordinates & Targets ---
def get_grid_coords(h, w, stride, device):
    shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
    return shift_x, shift_y 

def targets_for_image(locations, gt_boxes, stride):
    # locations: [N, 2]
    # gt_boxes: [M, 4]
    
    xs, ys = locations[:, 0], locations[:, 1]
    
    # [N, M] distances
    l = xs[:, None] - gt_boxes[None, :, 0]
    t = ys[:, None] - gt_boxes[None, :, 1]
    r = gt_boxes[None, :, 2] - xs[:, None]
    b = gt_boxes[None, :, 3] - ys[:, None]
    
    reg_targets = torch.stack([l, t, r, b], dim=2) # [N, M, 4]
    
    # Condition: point inside box
    is_in_box = reg_targets.min(dim=2)[0] > 0
    
    # Centerness Target
    max_l_r = torch.max(reg_targets[:, :, 0], reg_targets[:, :, 2])
    min_l_r = torch.min(reg_targets[:, :, 0], reg_targets[:, :, 2])
    max_t_b = torch.max(reg_targets[:, :, 1], reg_targets[:, :, 3])
    min_t_b = torch.min(reg_targets[:, :, 1], reg_targets[:, :, 3])
    centerness_targets = torch.sqrt((min_l_r / max_l_r) * (min_t_b / max_t_b))
    
    # Match to Box with Smallest Area (standard FCOS)
    areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    areas = areas[None, :].repeat(locations.shape[0], 1)
    areas[~is_in_box] = float('inf')
    
    min_area, min_area_inds = areas.min(dim=1)
    
    # VALID targets
    pos_mask = min_area < float('inf')
    inds = min_area_inds[pos_mask]
    
    # Just return the targets for positive locations
    # We will ignore negative locations for regression
    return pos_mask, reg_targets[pos_mask, inds], centerness_targets[pos_mask, inds]


def decode_boxes(pred_ltrb, locations):
    x, y = locations[:, 0], locations[:, 1]
    l, t, r, b = pred_ltrb[:, 0], pred_ltrb[:, 1], pred_ltrb[:, 2], pred_ltrb[:, 3]
    x1 = x - l
    y1 = y - t
    x2 = x + r
    y2 = y + b
    return torch.stack([x1, y1, x2, y2], dim=1)

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("ann_path", type=str)
    args = parser.parse_args()
    
    # 1. Load Data
    raw_img, raw_img_rgb = load_image(args.image_path)
    gt_boxes_orig = load_visdrone_ann(args.ann_path).to(DEVICE)
    h_orig, w_orig = raw_img.shape[:2]
    
    # Grid size (Stride 8)
    stride = 8
    feat_h, feat_w = int(h_orig / stride), int(w_orig / stride)
    print(f"Grid: {feat_h}x{feat_w}")
    
    # 2. Prep Tensor (unused for DirectGridModel but kept for API consistency if needed)
    img_tensor = torch.from_numpy(raw_img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.to(DEVICE).unsqueeze(0)
    
    # 3. Models - Direct Optimization
    model_l1 = DirectGridModel(feat_h, feat_w).to(DEVICE)
    model_ciou = DirectGridModel(feat_h, feat_w).to(DEVICE)
    
    # Use SGD or Adam with high LR for visibility
    LR = 0.05
    opt_l1 = optim.Adam(model_l1.parameters(), lr=LR)
    opt_ciou = optim.Adam(model_ciou.parameters(), lr=LR)
    
    # 4. Compute Geometry & Targets
    with torch.no_grad():
        reg_dummy, _ = model_l1(img_tensor)
        feat_h, feat_w = reg_dummy.shape[2], reg_dummy.shape[3]

    print(f"Grid: {feat_h}x{feat_w}")
    shift_x, shift_y = get_grid_coords(feat_h, feat_w, 8, DEVICE) # Stride 8 now
    locations = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=1)
    
    # Generate targets
    pos_mask, reg_targets_gt, centerness_gt = targets_for_image(locations, gt_boxes_orig, 8)
    
    # NORMALIZE TARGETS for stability (divide by stride)
    reg_targets_gt = reg_targets_gt / 8.0
    
    pos_inds = torch.nonzero(pos_mask).squeeze(1)
    num_pos = max(pos_inds.numel(), 1.0)
    
    print(f"Positives: {num_pos}")
    
    frames = []
    
    # Run longer
    for i in range(600):
        # --- L1 Step ---
        reg_pred, center_pred = model_l1(img_tensor)
        reg_pred = reg_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        center_pred = center_pred.permute(0, 2, 3, 1).reshape(-1, 1)
        
        # Train both Regression and Centerness
        # We need centerness for NMS
        loss_cnt = F.binary_cross_entropy_with_logits(center_pred[pos_inds], centerness_gt.unsqueeze(1), reduction='sum') / num_pos
        loss_reg = F.l1_loss(reg_pred[pos_inds], reg_targets_gt, reduction='sum') / num_pos
        
        loss_total = loss_reg + loss_cnt
        
        opt_l1.zero_grad()
        loss_total.backward()
        # torch.nn.utils.clip_grad_norm_(model_l1.parameters(), 1.0)
        opt_l1.step()
        
        # --- CIoU Step ---
        reg_pred_c, center_pred_c = model_ciou(img_tensor)
        reg_pred_c = reg_pred_c.permute(0, 2, 3, 1).reshape(-1, 4)
        center_pred_c = center_pred_c.permute(0, 2, 3, 1).reshape(-1, 1)
        
        loss_cnt_c = F.binary_cross_entropy_with_logits(center_pred_c[pos_inds], centerness_gt.unsqueeze(1), reduction='sum') / num_pos
        
        # Decode: Multiply by stride (8.0)
        boxes_pred_xyxy = decode_boxes(reg_pred_c[pos_inds] * 8.0, locations[pos_inds])
        boxes_gt_xyxy = decode_boxes(reg_targets_gt * 8.0, locations[pos_inds])
        loss_reg_c = complete_box_iou_loss(boxes_pred_xyxy, boxes_gt_xyxy, reduction='sum') / num_pos
        
        loss_total_c = loss_reg_c + loss_cnt_c
        
        opt_ciou.zero_grad()
        loss_total_c.backward()
        # torch.nn.utils.clip_grad_norm_(model_ciou.parameters(), 1.0)
        opt_ciou.step()
        
        # --- Visualization (Every 10 frames) ---
        if i % 10 == 0 or i == 299:
            # Stack vertically: Height is 2x, Width is 1x
            canvas = np.zeros((h_orig * 2, w_orig, 3), dtype=np.uint8)
            
            def draw_side(reg_flat, cen_flat, color):
                img_cp = raw_img.copy()
                # Draw GT
                for b in gt_boxes_orig:
                    x1,y1,x2,y2 = map(int, b)
                    # Use bright GT_COLOR (0, 255, 0) and thickness 2 for visibility
                    cv2.rectangle(img_cp, (x1,y1), (x2,y2), GT_COLOR, 2)
                
                # Draw Predictions with NMS
                with torch.no_grad():
                    # Denormalize
                    pred_boxes = decode_boxes(reg_flat[pos_inds] * 8.0, locations[pos_inds])
                    pred_boxes = torch.nan_to_num(pred_boxes, 0.0)
                    
                    # Scores from centerness
                    scores = torch.sigmoid(cen_flat[pos_inds]).squeeze(1)
                    
                    # NMS
                    keep = torch.ops.torchvision.nms(pred_boxes, scores, 0.9)
                    
                    # Draw filtered boxes (Top 500)
                    for idx in keep:
                         box = pred_boxes[idx]
                         x1,y1,x2,y2 = map(int, box)
                         # Simple clamp
                         x1 = max(0, x1); y1 = max(0, y1); x2 = min(w_orig, x2); y2 = min(h_orig, y2)
                         # Predictions with thickness 2
                         cv2.rectangle(img_cp, (x1,y1), (x2,y2), color, 2)
                return img_cp

            img_l1 = draw_side(reg_pred, center_pred, L1_COLOR)
            cv2.putText(img_l1, f"L1 Loss: {loss_reg.item():.3f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, L1_COLOR, 3)
            
            img_ciou = draw_side(reg_pred_c, center_pred_c, CIOU_COLOR)
            cv2.putText(img_ciou, f"CIoU Loss: {loss_reg_c.item():.3f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, CIOU_COLOR, 3)
            
            canvas[:h_orig, :] = img_l1
            canvas[h_orig:, :] = img_ciou
            
            # Resize for web (reduce file size)
            # Target width 800 (keeps aspect reasonable for vertical stacking)
            scale = 800.0 / canvas.shape[1]
            new_w = 800
            new_h = int(canvas.shape[0] * scale)
            canvas_resized = cv2.resize(canvas, (new_w, new_h))
            
            frames.append(Image.fromarray(cv2.cvtColor(canvas_resized, cv2.COLOR_BGR2RGB)))
            print(f"Frame {i}: L1={loss_reg.item():.3f}, CIoU={loss_reg_c.item():.3f}")

    if frames:
        out_path = OUTPUT_DIR / "fcos_optimization.gif"
        frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()