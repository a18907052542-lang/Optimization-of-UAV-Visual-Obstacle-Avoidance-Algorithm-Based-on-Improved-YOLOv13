"""
Training Script for Improved YOLOv13 UAV Obstacle Avoidance
Implements the complete training pipeline with:
- SGD optimizer with momentum 0.937
- Cosine annealing scheduler with warmup
- Mixed precision training
- Gradient accumulation
- Checkpointing and logging
"""

import os
import sys
import time
import argparse
import yaml
import math
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import ImprovedYOLOv13
from losses import DetectionLoss
from data import (
    VisDroneDataset, 
    UAVDTDataset, 
    CombinedUAVDataset,
    create_dataloader,
    collate_fn,
    TrainAugmentation,
    ValAugmentation
)
from utils import DetectionMetrics, evaluate_detection


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class CosineAnnealingWarmupScheduler:
    """
    Cosine annealing scheduler with linear warmup
    Used for learning rate scheduling during training
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_lr: float = 1e-6,
        min_lr: float = 1e-6
    ):
        """
        Initialize scheduler
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            warmup_lr: Initial learning rate during warmup
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        
        # Get base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """Update learning rate"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            if self.current_epoch < self.warmup_epochs:
                # Linear warmup
                alpha = self.current_epoch / self.warmup_epochs
                lr = self.warmup_lr + alpha * (self.base_lrs[i] - self.warmup_lr)
            else:
                # Cosine annealing
                progress = (self.current_epoch - self.warmup_epochs) / (
                    self.total_epochs - self.warmup_epochs
                )
                lr = self.min_lr + 0.5 * (self.base_lrs[i] - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                )
            
            param_group['lr'] = lr
    
    def get_lr(self) -> List[float]:
        """Get current learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]


class EMAModel:
    """
    Exponential Moving Average for model weights
    Helps stabilize training and often improves final performance
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Initialize EMA model
        
        Args:
            model: Model to track
            decay: EMA decay rate
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply EMA weights to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class Trainer:
    """
    Main trainer class for Improved YOLOv13
    Handles training loop, validation, checkpointing, and logging
    """
    
    def __init__(
        self,
        config: Dict,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: str
    ):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use
            output_dir: Directory for outputs
        """
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        train_cfg = config.get('training', {})
        self.epochs = train_cfg.get('epochs', 300)
        self.batch_size = train_cfg.get('batch_size', 16)
        self.accumulation_steps = train_cfg.get('accumulation_steps', 1)
        
        # Setup loss function
        self.criterion = DetectionLoss(
            num_classes=config.get('model', {}).get('num_classes', 10),
            reg_max=config.get('model', {}).get('reg_max', 16)
        )
        
        # Setup optimizer (SGD with momentum as per paper)
        opt_cfg = train_cfg.get('optimizer', {})
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=opt_cfg.get('lr', 0.01),
            momentum=opt_cfg.get('momentum', 0.937),
            weight_decay=opt_cfg.get('weight_decay', 0.0005),
            nesterov=opt_cfg.get('nesterov', True)
        )
        
        # Setup scheduler (cosine annealing with warmup)
        sched_cfg = train_cfg.get('scheduler', {})
        self.scheduler = CosineAnnealingWarmupScheduler(
            self.optimizer,
            warmup_epochs=sched_cfg.get('warmup_epochs', 3),
            total_epochs=self.epochs,
            warmup_lr=sched_cfg.get('warmup_lr', 1e-6),
            min_lr=sched_cfg.get('min_lr', 1e-6)
        )
        
        # Mixed precision training
        self.use_amp = train_cfg.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # EMA
        self.use_ema = train_cfg.get('use_ema', True)
        self.ema = EMAModel(self.model) if self.use_ema else None
        
        # Logging
        self.writer = SummaryWriter(self.output_dir / 'logs')
        
        # Metrics
        self.metrics = DetectionMetrics(
            num_classes=config.get('model', {}).get('num_classes', 10)
        )
        
        # State
        self.current_epoch = 0
        self.best_map = 0.0
        self.global_step = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_box_loss = 0.0
        total_obj_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            images = batch['image'].to(self.device)
            targets = batch['targets']
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets)
                    loss = loss_dict['total_loss'] / self.accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Optimizer step
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    if self.ema is not None:
                        self.ema.update()
            else:
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total_loss'] / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.ema is not None:
                        self.ema.update()
            
            # Accumulate losses
            total_loss += loss_dict['total_loss'].item()
            total_cls_loss += loss_dict.get('cls_loss', torch.tensor(0.0)).item()
            total_box_loss += loss_dict.get('box_loss', torch.tensor(0.0)).item()
            total_obj_loss += loss_dict.get('obj_loss', torch.tensor(0.0)).item()
            num_batches += 1
            
            # Logging
            self.global_step += 1
            
            if batch_idx % 50 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (batch_idx + 1) * (len(self.train_loader) - batch_idx - 1)
                
                print(f"  Batch [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss_dict['total_loss'].item():.4f} "
                      f"LR: {self.scheduler.get_lr()[0]:.6f} "
                      f"ETA: {eta:.0f}s")
        
        # Average losses
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'cls_loss': total_cls_loss / num_batches,
            'box_loss': total_box_loss / num_batches,
            'obj_loss': total_obj_loss / num_batches
        }
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        # Apply EMA weights for validation
        if self.ema is not None:
            self.ema.apply_shadow()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            targets = batch['targets']
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            loss_dict = self.criterion(outputs, targets)
            total_loss += loss_dict['total_loss'].item()
            num_batches += 1
            
            # Get predictions
            predictions = self.model.predict(
                images,
                conf_threshold=0.001,
                nms_threshold=0.65
            )
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
        
        # Restore original weights
        if self.ema is not None:
            self.ema.restore()
        
        # Compute metrics
        metrics = evaluate_detection(
            all_predictions,
            all_targets,
            num_classes=self.config.get('model', {}).get('num_classes', 10)
        )
        
        metrics['val_loss'] = total_loss / num_batches
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': {
                'current_epoch': self.scheduler.current_epoch
            },
            'best_map': self.best_map,
            'config': self.config
        }
        
        if self.ema is not None:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        # Save latest checkpoint
        torch.save(checkpoint, self.output_dir / 'last.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pt')
        
        # Save periodic checkpoint
        if (self.current_epoch + 1) % 50 == 0:
            torch.save(
                checkpoint, 
                self.output_dir / f'epoch_{self.current_epoch + 1}.pt'
            )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_map = checkpoint.get('best_map', 0.0)
        
        if self.ema is not None and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Accumulation steps: {self.accumulation_steps}")
        print(f"Effective batch size: {self.batch_size * self.accumulation_steps}")
        print("-" * 50)
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            print(f"\nEpoch [{epoch + 1}/{self.epochs}]")
            
            # Update learning rate
            self.scheduler.step(epoch)
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_losses['total_loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
            self.writer.add_scalar('mAP/0.5', val_metrics.get('mAP50', 0), epoch)
            self.writer.add_scalar('mAP/0.75', val_metrics.get('mAP75', 0), epoch)
            self.writer.add_scalar('LR', self.scheduler.get_lr()[0], epoch)
            
            # Print results
            print(f"  Train Loss: {train_losses['total_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  mAP@0.5: {val_metrics.get('mAP50', 0):.4f}")
            print(f"  mAP@0.75: {val_metrics.get('mAP75', 0):.4f}")
            print(f"  APS: {val_metrics.get('APS', 0):.4f}")
            print(f"  APM: {val_metrics.get('APM', 0):.4f}")
            print(f"  APL: {val_metrics.get('APL', 0):.4f}")
            print(f"  Epoch time: {epoch_time:.1f}s")
            
            # Save checkpoint
            is_best = val_metrics.get('mAP50', 0) > self.best_map
            if is_best:
                self.best_map = val_metrics.get('mAP50', 0)
                print(f"  New best mAP@0.5: {self.best_map:.4f}")
            
            self.save_checkpoint(is_best)
        
        print("\nTraining completed!")
        print(f"Best mAP@0.5: {self.best_map:.4f}")
        
        self.writer.close()


def create_model(config: Dict) -> nn.Module:
    """Create model from configuration"""
    model_cfg = config.get('model', {})
    
    model = ImprovedYOLOv13(
        num_classes=model_cfg.get('num_classes', 10),
        variant=model_cfg.get('variant', 'small'),
        in_channels=model_cfg.get('in_channels', 3)
    )
    
    return model


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    data_cfg = config.get('data', {})
    train_cfg = config.get('training', {})
    aug_cfg = config.get('augmentation', {})
    
    # Create augmentations
    train_transform = TrainAugmentation(
        img_size=data_cfg.get('img_size', 640),
        mosaic_prob=aug_cfg.get('mosaic_prob', 0.5),
        mixup_prob=aug_cfg.get('mixup_prob', 0.2),
        mixup_alpha=aug_cfg.get('mixup_alpha', 0.2),
        crop_scale=(aug_cfg.get('crop_scale_min', 0.5), 
                    aug_cfg.get('crop_scale_max', 1.0)),
        brightness=aug_cfg.get('brightness', 0.3),
        contrast=aug_cfg.get('contrast', 0.3),
        saturation=aug_cfg.get('saturation', 0.3),
        hue=aug_cfg.get('hue', 0.1)
    )
    
    val_transform = ValAugmentation(
        img_size=data_cfg.get('img_size', 640)
    )
    
    # Create datasets
    train_dataset = CombinedUAVDataset(
        visdrone_path=data_cfg.get('visdrone_path', './data/VisDrone2019'),
        uavdt_path=data_cfg.get('uavdt_path', './data/UAVDT'),
        split='train',
        transform=train_transform
    )
    
    val_dataset = CombinedUAVDataset(
        visdrone_path=data_cfg.get('visdrone_path', './data/VisDrone2019'),
        uavdt_path=data_cfg.get('uavdt_path', './data/UAVDT'),
        split='val',
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=train_cfg.get('batch_size', 16),
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 8),
        pin_memory=True
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=train_cfg.get('batch_size', 16),
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 8),
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description='Train Improved YOLOv13')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='runs/train',
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=str(output_dir)
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
