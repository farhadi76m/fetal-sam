import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

import os
from tqdm import tqdm

from mobilesam_lora import MobileSAMLoRA
from loss import MultiLabelCombinedLoss
class MobileSAMTrainer:
    """Trainer class for MobileSAM fine-tuning"""
    
    def __init__(self, model: MobileSAMLoRA, train_dataloader, val_dataloader, 
                 device: str = 'cuda', learning_rate: float = 1e-4, 
                 weight_decay: float = 1e-4, num_epochs: int = 50):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.num_epochs = num_epochs
        
        # Loss function
        self.criterion = MultiLabelCombinedLoss()
        
        # Optimizer - only optimize LoRA parameters and unfrozen components
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=num_epochs,
            eta_min=learning_rate * 0.01
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def compute_metrics(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor):
        """Compute segmentation metrics"""
        pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
        
        # IoU
        intersection = (pred_binary * gt_masks).sum()
        union = (pred_binary + gt_masks - pred_binary * gt_masks).sum()
        iou = intersection / (union + 1e-6)
        
        # Dice
        dice = (2 * intersection) / (pred_binary.sum() + gt_masks.sum() + 1e-6)
        
        return {'iou': iou.item(), 'dice': dice.item()}
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_metrics = {'iou': 0, 'dice': 0}
        
        pbar = tqdm(self.train_dataloader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch - adjust based on your dataloader format
            images = batch['image'].to(self.device)
            gt_masks = batch['mask'].to(self.device)
            
            # Optional: get prompts if available in your data
            input_points = batch.get('points', None)
            input_labels = batch.get('labels', None)
            input_boxes = batch.get('boxes', None)
            
            if input_points is not None:
                input_points = input_points.to(self.device)
            if input_labels is not None:
                input_labels = input_labels.to(self.device)
            if input_boxes is not None:
                input_boxes = input_boxes.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_masks, pred_iou = self.model(images, input_points, input_labels, input_boxes)
            
            # Compute loss
            loss, loss_dict = self.criterion(pred_masks, gt_masks, pred_iou)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            metrics = self.compute_metrics(pred_masks, gt_masks)
            
            # Update tracking
            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'iou': f"{metrics['iou']:.4f}",
                'dice': f"{metrics['dice']:.4f}"
            })
        
        # Average metrics
        avg_loss = total_loss / len(self.train_dataloader)
        avg_metrics = {k: v / len(self.train_dataloader) for k, v in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def validate(self):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        total_metrics = {'iou': 0, 'dice': 0}
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc='Validation'):
                images = batch['image'].to(self.device)
                gt_masks = batch['mask'].to(self.device)
                
                input_points = batch.get('points', None)
                input_labels = batch.get('labels', None)
                input_boxes = batch.get('boxes', None)
                
                if input_points is not None:
                    input_points = input_points.to(self.device)
                if input_labels is not None:
                    input_labels = input_labels.to(self.device)
                if input_boxes is not None:
                    input_boxes = input_boxes.to(self.device)
                
                pred_masks, pred_iou = self.model(images, input_points, input_labels, input_boxes)
                loss, _ = self.criterion(pred_masks, gt_masks, pred_iou)
                metrics = self.compute_metrics(pred_masks, gt_masks)
                
                total_loss += loss.item()
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
        
        avg_loss = total_loss / len(self.val_dataloader)
        avg_metrics = {k: v / len(self.val_dataloader) for k, v in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, epoch: int, save_dir: str = './checkpoints'):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        # Save best model
        if len(self.val_losses) > 0 and self.val_losses[-1] < self.best_val_loss:
            self.best_val_loss = self.val_losses[-1]
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
    
    def train(self, save_every: int = 5):
        """Main training loop"""
        print(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step()
            
            # Print metrics
            print(f"Train - Loss: {train_loss:.4f}, IoU: {train_metrics['iou']:.4f}, Dice: {train_metrics['dice']:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1)
        
        print("Training completed!")


# Usage example
def main():
    # Initialize model
    model = MobileSAMLoRA(
        checkpoint="path/to/mobile_sam.pt",
        freeze_prompt_encoder=True,
        freeze_mask_decoder=False
    )
    
    # Initialize trainer (assuming you have your dataloaders)
    trainer = MobileSAMTrainer(
        model=model,
        train_dataloader=train_dataloader,  # Your dataloader
        val_dataloader=val_dataloader,      # Your dataloader
        device='cuda',
        learning_rate=1e-4,
        weight_decay=1e-4,
        num_epochs=50
    )
    
    # Start training
    trainer.train(save_every=5)

if __name__ == "__main__":
    main()