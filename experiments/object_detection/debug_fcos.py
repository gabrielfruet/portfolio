import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from torchvision.ops import complete_box_iou_loss

# --- Config ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stride = 8

# --- Helpers ---
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

def get_grid_coords(h, w, stride, device):
    shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
    return shift_x, shift_y 

def targets_for_image(locations, gt_boxes, stride):
    xs, ys = locations[:, 0], locations[:, 1]
    
    # [N, M] distances
    # l = x - x1, t = y - y1, r = x2 - x, b = y2 - y
    l = xs[:, None] - gt_boxes[None, :, 0]
    t = ys[:, None] - gt_boxes[None, :, 1]
    r = gt_boxes[None, :, 2] - xs[:, None]
    b = gt_boxes[None, :, 3] - ys[:, None]
    
    reg_targets = torch.stack([l, t, r, b], dim=2) # [N, M, 4]
    
    # Condition: point inside box
    is_in_box = reg_targets.min(dim=2)[0] > 0
    
    areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    areas = areas[None, :].repeat(locations.shape[0], 1)
    areas[~is_in_box] = float('inf')
    
    min_area, min_area_inds = areas.min(dim=1)
    
    pos_mask = min_area < float('inf')
    inds = min_area_inds[pos_mask]
    
    return pos_mask, reg_targets[pos_mask, inds]

def decode_boxes(pred_ltrb, locations):
    x, y = locations[:, 0], locations[:, 1]
    l, t, r, b = pred_ltrb[:, 0], pred_ltrb[:, 1], pred_ltrb[:, 2], pred_ltrb[:, 3]
    x1 = x - l
    y1 = y - t
    x2 = x + r
    y2 = y + b
    return torch.stack([x1, y1, x2, y2], dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("ann_path", type=str)
    args = parser.parse_args()

    print("--- 1. Data Loading ---")
    raw_img, raw_img_rgb = load_image(args.image_path)
    gt_boxes = load_visdrone_ann(args.ann_path).to(DEVICE)
    h, w = raw_img.shape[:2]
    print(f"Image: {h}x{w}")
    print(f"GT Boxes: {gt_boxes.shape}")

    # Grid
    feat_h, feat_w = int(h / stride), int(w / stride)
    print(f"Grid: {feat_h}x{feat_w}")
    shift_x, shift_y = get_grid_coords(feat_h, feat_w, stride, DEVICE)
    locations = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=1)
    
    print("--- 2. Targets ---")
    pos_mask, reg_targets = targets_for_image(locations, gt_boxes, stride)
    pos_inds = torch.nonzero(pos_mask).squeeze(1)
    print(f"Positive Anchors: {pos_inds.numel()}")
    
    if pos_inds.numel() == 0:
        print("ERROR: No positive anchors found! Check coordinate system or box sizes.")
        return

    # Visual Debug of Anchors
    debug_img = raw_img.copy()
    # Draw GT
    for b in gt_boxes:
        x1,y1,x2,y2 = map(int, b)
        cv2.rectangle(debug_img, (x1,y1), (x2,y2), (0, 255, 0), 2)
    
    # Draw Positive Anchors
    pos_locs = locations[pos_mask].cpu().numpy()
    for (px, py) in pos_locs:
        cv2.circle(debug_img, (int(px), int(py)), 2, (0, 0, 255), -1)
        
    cv2.imwrite("debug_anchors.jpg", debug_img)
    print("Saved debug_anchors.jpg")
    
    print("--- 3. Optimization Sanity Check (No Img Model) ---")
    # Initialize a learnable tensor representing predicted offsets for positive anchors ONLY
    # Initialize FAR from target
    # Targets are [l, t, r, b]
    # We normalize targets by stride for stability
    targets_norm = reg_targets / stride
    print(f"Targets Max: {targets_norm.max().item()}, Min: {targets_norm.min().item()}")
    
    # Init pred with mean 1.0 (8 pixels)
    pred_params = nn.Parameter(torch.ones_like(targets_norm) * 1.0) 
    opt = torch.optim.Adam([pred_params], lr=0.1)
    
    print("Optimizing Single Tensor directly against Targets (L1)...")
    for i in range(100):
        loss = torch.nn.functional.l1_loss(pred_params, targets_norm, reduction='mean')
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 20 == 0:
            print(f"Iter {i}: L1 Loss={loss.item():.4f}, Pred Mean={pred_params.mean().item():.3f}")
            
    print("Final L1 Pred Mean:", pred_params.mean().item())
    
    # CIoU Check
    print("\nOptimizing Single Tensor directly against Targets (CIoU)...")
    pred_params_c = nn.Parameter(torch.ones_like(targets_norm) * 1.0)
    opt_c = torch.optim.Adam([pred_params_c], lr=0.1)
    
    # Locations for positives
    locs_pos = locations[pos_mask]
    
    for i in range(100):
        # Decode
        # Must prevent negative predictions for IoU
        pred_clamped = torch.relu(pred_params_c) + 0.001
        
        boxes_pred = decode_boxes(pred_clamped * stride, locs_pos)
        boxes_target = decode_boxes(targets_norm * stride, locs_pos)
        
        loss = complete_box_iou_loss(boxes_pred, boxes_target, reduction='mean')
        
        opt_c.zero_grad()
        loss.backward()
        opt_c.step()
        
        if i % 20 == 0:
            print(f"Iter {i}: CIoU Loss={loss.item():.4f}, Pred Mean={pred_params_c.mean().item():.3f}")

if __name__ == "__main__":
    main()
