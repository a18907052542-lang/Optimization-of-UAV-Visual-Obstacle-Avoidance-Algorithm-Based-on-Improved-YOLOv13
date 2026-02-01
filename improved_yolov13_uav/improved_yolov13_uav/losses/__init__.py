"""
Loss functions module for Improved YOLOv13 UAV Obstacle Avoidance
Contains Focal Loss, Distribution Focal Loss, IoU losses, and combined detection loss
"""

from .loss import (
    FocalLoss,
    DistributionFocalLoss,
    IoULoss,
    QualityFocalLoss,
    VarifocalLoss,
    DetectionLoss,
    TaskAlignedAssigner,
    compute_iou_loss,
    compute_cls_loss
)

__all__ = [
    'FocalLoss',
    'DistributionFocalLoss',
    'IoULoss',
    'QualityFocalLoss',
    'VarifocalLoss',
    'DetectionLoss',
    'TaskAlignedAssigner',
    'compute_iou_loss',
    'compute_cls_loss'
]
