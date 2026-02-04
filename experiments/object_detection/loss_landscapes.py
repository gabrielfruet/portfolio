import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torchvision.ops.ciou_loss import complete_box_iou_loss

def get_iou_loss(pred_box, gt_box):
    """
    Standard IoU Loss (1 - IoU)
    Boxes: [cx, cy, w, h]
    """
    # Convert cx, cy, w, h to x1, y1, x2, y2
    pred_x1 = pred_box[0] - pred_box[2] / 2
    pred_y1 = pred_box[1] - pred_box[3] / 2
    pred_x2 = pred_box[0] + pred_box[2] / 2
    pred_y2 = pred_box[1] + pred_box[3] / 2

    gt_x1 = gt_box[0] - gt_box[2] / 2
    gt_y1 = gt_box[1] - gt_box[3] / 2
    gt_x2 = gt_box[0] + gt_box[2] / 2
    gt_y2 = gt_box[1] + gt_box[3] / 2

    # Intersection
    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union
    pred_area = pred_box[2] * pred_box[3]
    gt_area = gt_box[2] * gt_box[3]
    union_area = pred_area + gt_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return 1.0 - iou

def xcycwh_to_xyxy(box):
    """
    Convert from (xc,yc,w,h) to (x1,y1,x2,y2)
    """
    xc = box[:2]
    wh = box[2:]
    xy1 = xc - wh / 2
    xy2 = xc + wh / 2
    return torch.cat([xy1, xy2],dim=0)


def get_ciou_loss(pred_box, gt_box):
    """
    Complete IoU Loss (1 - IoU)
    Boxes: [cx, cy, w, h]
    """
    return complete_box_iou_loss(xcycwh_to_xyxy(pred_box), xcycwh_to_xyxy(gt_box), reduction="sum")


def get_l1_loss(pred_box, gt_box):
    """
    L1 Loss on coordinates
    """
    return torch.nn.functional.l1_loss(pred_box, gt_box)

def plot_loss_landscape(loss_func, title, ax):
    # Setup Ground Truth Box [cx, cy, w, h]
    gt_box = torch.tensor([0.0, 0.0, 1.0, 1.0]) # Center at 0,0, size 1x1

    # Grid for Predicted Box Center (cx, cy)
    # We keep w, h fixed to match GT to isolate position optimization
    range_val = 1.5
    steps = 25
    x = np.linspace(-range_val, range_val, steps)
    y = np.linspace(-range_val, range_val, steps)
    X, Y = np.meshgrid(x, y)

    Z_loss = np.zeros_like(X)
    U_grad = np.zeros_like(X)
    V_grad = np.zeros_like(X)

    for i in range(steps):
        for j in range(steps):
            # Create a learnable tensor for the predicted box
            # Initialized at grid position (X[i,j], Y[i,j])
            pred_box = torch.tensor([X[i, j], Y[i, j], 1.0, 1.0], requires_grad=True)
            
            # Forward pass
            loss = loss_func(pred_box, gt_box)
            
            # Backward pass to get gradients
            loss.backward()
            
            Z_loss[i, j] = loss.item()
            # Negative gradient because we want the direction of descent
            U_grad[i, j] = -pred_box.grad[0].item() 
            V_grad[i, j] = -pred_box.grad[1].item()

    # Visualization
    # 1. Contour (Loss Surface)
    cp = ax.contourf(X, Y, Z_loss, levels=20, cmap='viridis', alpha=0.6)
    
    # 2. Quiver (Gradient Field)
    # Normalize arrows for cleaner look, coloring by magnitude
    magnitude = np.sqrt(U_grad**2 + V_grad**2)
    
    # Avoid division by zero for normalization
    magnitude_safe = magnitude.copy()
    magnitude_safe[magnitude_safe == 0] = 1.0 
    
    ax.quiver(X, Y, U_grad/magnitude_safe, V_grad/magnitude_safe, magnitude, 
              cmap='autumn', scale=20, width=0.005)

    # Draw Ground Truth Box
    rect = plt.Rectangle((-0.5, -0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(-range_val, range_val)
    ax.set_ylim(-range_val, range_val)

def plot_loss_landscape_3d(loss_func, title, ax: plt.Axes):
    gt_box = torch.tensor([0.0, 0.0, 1.0, 1.0]) # Center at 0,0, size 1x1

    # Grid for Predicted Box Center (cx, cy)
    # We keep w, h fixed to match GT to isolate position optimization
    range_val = 1.5
    steps = 40
    x = np.linspace(-range_val, range_val, steps)
    y = np.linspace(-range_val, range_val, steps)
    X, Y = np.meshgrid(x, y)

    Z_loss = np.zeros_like(X)
    U_grad = np.zeros_like(X)
    V_grad = np.zeros_like(X)
    W_grad = np.zeros_like(X)


    for i in range(steps):
        for j in range(steps):
            # Create a learnable tensor for the predicted box
            # Initialized at grid position (X[i,j], Y[i,j])
            pred_box = torch.tensor([X[i, j], Y[i, j], 1.0, 1.0], requires_grad=True)
            
            # Forward pass
            loss = loss_func(pred_box, gt_box)
            
            # Backward pass to get gradients
            loss.backward()
            
            Z_loss[i, j] = loss.item()
            # Negative gradient because we want the direction of descent
            U_grad[i, j] = -pred_box.grad[0].item() 
            V_grad[i, j] = -pred_box.grad[1].item()
            W_grad[i, j] = -(U_grad[i, j]**2 + V_grad[i, j]**2)

    W_grad = np.sqrt(W_grad)
    ax.plot_surface(X, Y, Z_loss, cmap='plasma', alpha=0.9)
    ax.quiver(X, Y, Z_loss, U_grad, V_grad, W_grad, color='black', length=0.1)
    # draw 3d ball at some point
    ax.set_title(title)
    ax.set_xlim(-range_val, range_val)
    ax.set_ylim(-range_val, range_val)
    ax.set_xlabel("cx")
    ax.set_ylabel("cy")
    ax.set_zlabel("Loss")

    


# --- Run Visualization ---
fig1, ax1 = plt.subplots(1, figsize=(6, 6), subplot_kw=dict(projection="3d"))
fig2, ax2 = plt.subplots(1, figsize=(6, 6), subplot_kw=dict(projection="3d"))
fig3, ax3 = plt.subplots(1, figsize=(6, 6), subplot_kw=dict(projection="3d"))

# Plot L1
plot_loss_landscape_3d(get_l1_loss, "L1 Loss Landscape & Gradients", ax1)
# Plot IoU
plot_loss_landscape_3d(get_iou_loss, "IoU Loss Landscape & Gradients", ax2)
plot_loss_landscape_3d(get_ciou_loss, "Complete IoU Loss Landscape & Gradients", ax3)

# plt.show()

images_path = Path("src/assets/images/")
if not images_path.exists():
    raise FileNotFoundError(f"Images path {images_path} does not exist")

subpath = Path("experiments/object_detection/")
(images_path / subpath).mkdir(parents=True, exist_ok=True)

for fig, image_name in zip([fig1, fig2, fig3], ["l1_loss_landscape.png", "iou_loss_landscape.png", "ciou_loss_landscape.png"]):
    fig.savefig(images_path / subpath / image_name, dpi=300)