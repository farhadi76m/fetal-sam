import torch
from torch.nn import nn
from peft import LoraConfig, get_peft_model
from MobileSAM.mobile_sam.build_sam import build_sam_vit_t

from typing import Optional

class MobileSAMLoRA(nn.Module):
    """MobileSAM with LoRA fine-tuning for multi-label medical segmentation"""
    
    def __init__(self, checkpoint: str, num_classes: int = 4, freeze_prompt_encoder: bool = True, 
                 freeze_mask_decoder: bool = False):
        super().__init__()
        
        self.num_classes = num_classes
        
        # LoRA configuration targeting attention modules
        target_modules = [
            "layers.1.blocks.0.attn.qkv",
            "layers.1.blocks.0.attn.proj",
            "layers.1.blocks.1.attn.qkv",
            "layers.1.blocks.1.attn.proj",
            "layers.2.blocks.0.attn.qkv",
            "layers.2.blocks.0.attn.proj",
            "layers.2.blocks.1.attn.qkv",
            "layers.2.blocks.1.attn.proj",
            "layers.2.blocks.2.attn.qkv",
            "layers.2.blocks.2.attn.proj",
            "layers.2.blocks.3.attn.qkv",
            "layers.2.blocks.3.attn.proj",
            "layers.2.blocks.4.attn.qkv",
            "layers.2.blocks.4.attn.proj",
            "layers.2.blocks.5.attn.qkv",
            "layers.2.blocks.5.attn.proj",
        ]

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )

        # Load the MobileSAM model
        self.sam_model = build_sam_vit_t(checkpoint=checkpoint)
        
        # Apply LoRA to the image encoder
        self.sam_model.image_encoder = get_peft_model(self.sam_model.image_encoder, lora_config)
        
        # Freeze components based on strategy
        if freeze_prompt_encoder:
            for param in self.sam_model.prompt_encoder.parameters():
                param.requires_grad = False
                
        if freeze_mask_decoder:
            for param in self.sam_model.mask_decoder.parameters():
                param.requires_grad = False
    
    def forward(self, images: torch.Tensor, input_points: Optional[torch.Tensor] = None, 
                input_labels: Optional[torch.Tensor] = None, input_boxes: Optional[torch.Tensor] = None):
        """
        Forward pass for multi-label segmentation
        
        Args:
            images: Input images [B, 3, H, W]
            input_points: Point prompts [B, N, 2]
            input_labels: Point labels [B, N]
            input_boxes: Box prompts [B, N, 4]
        """
        # Get image embeddings through LoRA-adapted encoder
        image_embeddings = self.sam_model.image_encoder(images)
        
        # Prepare prompts
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=(input_points, input_labels) if input_points is not None else None,
            boxes=input_boxes,
            masks=None,
        )
        
        # Generate masks
        masks, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        return masks, iou_predictions