import torch
from torch.nn import nn

class MultiLabelCombinedLoss(nn.Module):
    """Combined loss function for multi-label medical image segmentation"""
    
    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.3, 
                 iou_weight: float = 0.2, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
    def dice_loss(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor, smooth: float = 1e-6):
        """Dice loss for multi-label segmentation"""
        pred_masks = torch.sigmoid(pred_masks)
        
        # Flatten spatial dimensions
        pred_flat = pred_masks.view(pred_masks.size(0), -1)  # [B, H*W]
        gt_flat = gt_masks.view(gt_masks.size(0), -1)  # [B, H*W]
        
        intersection = (pred_flat * gt_flat).sum(dim=1)  # [B]
        dice_scores = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + gt_flat.sum(dim=1) + smooth)
        
        return 1 - dice_scores.mean()
    
    def focal_loss(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor):
        """Focal loss to handle class imbalance in multi-label setting"""
        pred_sigmoid = torch.sigmoid(pred_masks)
        ce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks, reduction='none')
        pt = torch.where(gt_masks == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def iou_loss(self, pred_iou: torch.Tensor, pred_masks: torch.Tensor, gt_masks: torch.Tensor):
        """IoU prediction loss for multi-label"""
        with torch.no_grad():
            pred_masks_binary = (torch.sigmoid(pred_masks) > 0.5).float()
            
            # Calculate IoU for each sample
            intersection = (pred_masks_binary * gt_masks).sum(dim=(-2, -1))  # [B]
            union = (pred_masks_binary + gt_masks - pred_masks_binary * gt_masks).sum(dim=(-2, -1))  # [B]
            target_iou = intersection / (union + 1e-6)
        
        # pred_iou shape: [B, 1] -> squeeze to [B]
        pred_iou_flat = pred_iou.squeeze(-1)
        return F.mse_loss(pred_iou_flat, target_iou)
    
    def forward(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor, pred_iou: torch.Tensor):
        """Combined loss computation for multi-label segmentation"""
        dice_l = self.dice_loss(pred_masks, gt_masks)
        focal_l = self.focal_loss(pred_masks, gt_masks)
        iou_l = self.iou_loss(pred_iou, pred_masks, gt_masks)
        
        total_loss = (self.dice_weight * dice_l + 
                     self.focal_weight * focal_l + 
                     self.iou_weight * iou_l)
        
        return total_loss, {
            'dice_loss': dice_l.item(),
            'focal_loss': focal_l.item(),
            'iou_loss': iou_l.item(),
            'total_loss': total_loss.item()
        }