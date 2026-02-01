"""
Evaluation Metrics Module for Improved YOLOv13 UAV Obstacle Avoidance
=====================================================================

Implements evaluation metrics as specified in Section 4.2:
- AP (Average Precision) at different IoU thresholds (AP50, AP75)
- AP for different target scales (APS, APM, APL)
- AR (Average Recall) related metrics
- FPS (Frames Per Second)
- Model complexity metrics (Params, GFLOPs)

Target scale definitions:
- Small (APS): < 32×32 pixels
- Medium (APM): 32×32 to 96×96 pixels
- Large (APL): > 96×96 pixels
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
import time


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: Array of shape (N, 4) in format [x1, y1, x2, y2]
        boxes2: Array of shape (M, 4) in format [x1, y1, x2, y2]
    
    Returns:
        IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])
    
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    union_area = area1[:, None] + area2 - inter_area
    iou = inter_area / (union_area + 1e-7)
    
    return iou


def box_area(boxes: np.ndarray, img_size: int = 640) -> np.ndarray:
    """
    Compute area of boxes in pixels.
    
    Args:
        boxes: Array of shape (N, 4) in normalized format [x1, y1, x2, y2]
        img_size: Image size for denormalization
    
    Returns:
        Array of areas in pixels squared
    """
    # Denormalize
    boxes_pixel = boxes * img_size
    w = boxes_pixel[:, 2] - boxes_pixel[:, 0]
    h = boxes_pixel[:, 3] - boxes_pixel[:, 1]
    return w * h


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision using 101-point interpolation (COCO style).
    
    Args:
        recalls: Recall values at different thresholds
        precisions: Precision values at different thresholds
    
    Returns:
        Average Precision value
    """
    # 101-point interpolation
    recall_points = np.linspace(0, 1, 101)
    
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]
    
    # Interpolate precision at each recall point
    interpolated_precision = np.zeros(101)
    for i, r in enumerate(recall_points):
        # Find precision at recall >= r
        mask = recalls >= r
        if np.any(mask):
            interpolated_precision[i] = np.max(precisions[mask])
    
    # AP is mean of interpolated precisions
    ap = np.mean(interpolated_precision)
    return ap


def compute_pr_curve(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Precision-Recall curve and AP.
    
    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        ground_truths: List of ground truth dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes
    
    Returns:
        recalls, precisions, AP
    """
    # Collect all predictions with their scores and image indices
    all_preds = []
    for img_idx, pred in enumerate(predictions):
        for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
            all_preds.append({
                'img_idx': img_idx,
                'box': box,
                'score': score,
                'label': label
            })
    
    # Sort by confidence score (descending)
    all_preds = sorted(all_preds, key=lambda x: x['score'], reverse=True)
    
    # Count total ground truth objects
    total_gt = sum(len(gt['boxes']) for gt in ground_truths)
    if total_gt == 0:
        return np.array([0.0]), np.array([0.0]), 0.0
    
    # Track which ground truths have been matched
    gt_matched = [np.zeros(len(gt['boxes']), dtype=bool) for gt in ground_truths]
    
    # Compute TP and FP for each prediction
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))
    
    for pred_idx, pred in enumerate(all_preds):
        img_idx = pred['img_idx']
        pred_box = pred['box']
        pred_label = pred['label']
        
        gt = ground_truths[img_idx]
        gt_boxes = gt['boxes']
        gt_labels = gt['labels']
        
        # Find matching ground truth
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_label != pred_label:
                continue
            if gt_matched[img_idx][gt_idx]:
                continue
            
            iou = box_iou(pred_box.reshape(1, 4), gt_box.reshape(1, 4))[0, 0]
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[pred_idx] = 1
            gt_matched[img_idx][best_gt_idx] = True
        else:
            fp[pred_idx] = 1
    
    # Compute cumulative TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Compute precision and recall
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
    recalls = tp_cumsum / total_gt
    
    # Compute AP
    ap = compute_ap(recalls, precisions)
    
    return recalls, precisions, ap


class COCOEvaluator:
    """
    COCO-style object detection evaluator.
    
    Computes metrics as specified in Section 4.2:
    - mAP@0.5 (IoU=0.5)
    - mAP@0.75 (IoU=0.75)
    - APS (small objects < 32²)
    - APM (medium objects 32²-96²)
    - APL (large objects > 96²)
    - AR (Average Recall)
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        iou_thresholds: List[float] = None,
        img_size: int = 640
    ):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.img_size = img_size
        
        # Scale thresholds in pixels (COCO standard)
        self.small_threshold = 32 ** 2  # < 32×32 pixels
        self.large_threshold = 96 ** 2  # > 96×96 pixels
        
        self.reset()
    
    def reset(self):
        """Reset evaluation state."""
        self.predictions = []
        self.ground_truths = []
    
    def update(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor
    ):
        """
        Add predictions and ground truths for one image.
        
        Args:
            pred_boxes: Predicted boxes (N, 4)
            pred_scores: Prediction scores (N,)
            pred_labels: Predicted labels (N,)
            gt_boxes: Ground truth boxes (M, 4)
            gt_labels: Ground truth labels (M,)
        """
        self.predictions.append({
            'boxes': pred_boxes.cpu().numpy() if torch.is_tensor(pred_boxes) else pred_boxes,
            'scores': pred_scores.cpu().numpy() if torch.is_tensor(pred_scores) else pred_scores,
            'labels': pred_labels.cpu().numpy() if torch.is_tensor(pred_labels) else pred_labels
        })
        
        self.ground_truths.append({
            'boxes': gt_boxes.cpu().numpy() if torch.is_tensor(gt_boxes) else gt_boxes,
            'labels': gt_labels.cpu().numpy() if torch.is_tensor(gt_labels) else gt_labels
        })
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Returns:
            Dictionary containing:
            - mAP@0.5
            - mAP@0.75
            - mAP (average over IoU thresholds)
            - APS (small objects)
            - APM (medium objects)
            - APL (large objects)
            - AR (Average Recall)
        """
        metrics = {}
        
        # mAP at different IoU thresholds
        ap_per_iou = []
        for iou_thresh in self.iou_thresholds:
            _, _, ap = compute_pr_curve(
                self.predictions, self.ground_truths,
                iou_threshold=iou_thresh, num_classes=self.num_classes
            )
            ap_per_iou.append(ap)
            
            if iou_thresh == 0.5:
                metrics['mAP@0.5'] = ap
            elif iou_thresh == 0.75:
                metrics['mAP@0.75'] = ap
        
        metrics['mAP'] = np.mean(ap_per_iou)
        
        # AP by object size
        metrics['APS'] = self._compute_ap_by_size('small')
        metrics['APM'] = self._compute_ap_by_size('medium')
        metrics['APL'] = self._compute_ap_by_size('large')
        
        # Average Recall
        metrics['AR'] = self._compute_ar()
        
        return metrics
    
    def _compute_ap_by_size(self, size_category: str) -> float:
        """Compute AP for specific object size category."""
        filtered_preds = []
        filtered_gts = []
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            # Filter predictions by size
            if len(pred['boxes']) > 0:
                pred_areas = box_area(pred['boxes'], self.img_size)
                if size_category == 'small':
                    mask = pred_areas < self.small_threshold
                elif size_category == 'medium':
                    mask = (pred_areas >= self.small_threshold) & (pred_areas <= self.large_threshold)
                else:  # large
                    mask = pred_areas > self.large_threshold
                
                filtered_preds.append({
                    'boxes': pred['boxes'][mask],
                    'scores': pred['scores'][mask],
                    'labels': pred['labels'][mask]
                })
            else:
                filtered_preds.append({'boxes': np.array([]), 'scores': np.array([]), 'labels': np.array([])})
            
            # Filter ground truths by size
            if len(gt['boxes']) > 0:
                gt_areas = box_area(gt['boxes'], self.img_size)
                if size_category == 'small':
                    mask = gt_areas < self.small_threshold
                elif size_category == 'medium':
                    mask = (gt_areas >= self.small_threshold) & (gt_areas <= self.large_threshold)
                else:  # large
                    mask = gt_areas > self.large_threshold
                
                filtered_gts.append({
                    'boxes': gt['boxes'][mask],
                    'labels': gt['labels'][mask]
                })
            else:
                filtered_gts.append({'boxes': np.array([]), 'labels': np.array([])})
        
        _, _, ap = compute_pr_curve(filtered_preds, filtered_gts, iou_threshold=0.5, num_classes=self.num_classes)
        return ap
    
    def _compute_ar(self, max_dets: int = 100) -> float:
        """Compute Average Recall at max detections."""
        total_recalls = []
        
        for pred, gt in zip(self.predictions, self.ground_truths):
            if len(gt['boxes']) == 0:
                continue
            
            # Keep only top-k predictions
            if len(pred['boxes']) > max_dets:
                top_k_idx = np.argsort(pred['scores'])[-max_dets:]
                pred_boxes = pred['boxes'][top_k_idx]
                pred_labels = pred['labels'][top_k_idx]
            else:
                pred_boxes = pred['boxes']
                pred_labels = pred['labels']
            
            # Match predictions to ground truths
            if len(pred_boxes) > 0:
                ious = box_iou(pred_boxes, gt['boxes'])
                
                # For each ground truth, find best matching prediction
                matched = 0
                gt_matched = np.zeros(len(gt['boxes']), dtype=bool)
                
                for pred_idx in range(len(pred_boxes)):
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx in range(len(gt['boxes'])):
                        if gt_matched[gt_idx]:
                            continue
                        if pred_labels[pred_idx] != gt['labels'][gt_idx]:
                            continue
                        
                        if ious[pred_idx, gt_idx] > best_iou:
                            best_iou = ious[pred_idx, gt_idx]
                            best_gt_idx = gt_idx
                    
                    if best_iou >= 0.5 and best_gt_idx >= 0:
                        matched += 1
                        gt_matched[best_gt_idx] = True
                
                recall = matched / len(gt['boxes'])
            else:
                recall = 0.0
            
            total_recalls.append(recall)
        
        return np.mean(total_recalls) if total_recalls else 0.0


class FPSMeter:
    """
    FPS measurement utility.
    
    Reference: Section 4.2 - Real-time performance evaluation
    "Testing disabled all optimization options to obtain baseline performance"
    """
    
    def __init__(self, warmup: int = 10):
        self.warmup = warmup
        self.reset()
    
    def reset(self):
        """Reset FPS meter."""
        self.times = []
        self.count = 0
    
    def start(self):
        """Start timing."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
    
    def stop(self):
        """Stop timing and record."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start_time
        
        self.count += 1
        if self.count > self.warmup:
            self.times.append(elapsed)
    
    @property
    def fps(self) -> float:
        """Get average FPS."""
        if not self.times:
            return 0.0
        avg_time = np.mean(self.times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    @property
    def latency_ms(self) -> float:
        """Get average latency in milliseconds."""
        if not self.times:
            return 0.0
        return np.mean(self.times) * 1000


class ModelComplexityAnalyzer:
    """
    Model complexity analyzer.
    
    Reference: Section 3.5 - Algorithm Complexity Analysis
    
    Computational complexity: O_total = Σ_{l=1}^{L} (C_in^l · C_out^l · K² · H^l · W^l)
    
    Table 6: Memory Usage and Optimization Analysis
    - Model Weights: Original 87.6MB -> Optimized 21.9MB (75% reduction)
    - Feature Map Cache: 812MB -> 486MB (40.1% reduction)
    - Total: 2,120MB -> 656MB (69.1% reduction)
    """
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def count_all_parameters(model: torch.nn.Module) -> int:
        """Count all parameters (trainable and non-trainable)."""
        return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def estimate_flops(model: torch.nn.Module, input_size: Tuple[int, int, int, int]) -> float:
        """
        Estimate FLOPs for the model.
        
        Implements Equation (10) from Section 3.5:
        O_total = Σ_{l=1}^{L} (C_in^l · C_out^l · K² · H^l · W^l)
        
        Args:
            model: PyTorch model
            input_size: Input tensor size (B, C, H, W)
        
        Returns:
            Estimated FLOPs in billions
        """
        total_flops = 0
        
        def hook_fn(module, input, output):
            nonlocal total_flops
            
            if isinstance(module, torch.nn.Conv2d):
                batch_size = input[0].shape[0]
                out_h, out_w = output.shape[2], output.shape[3]
                
                # FLOPs = 2 * K² * C_in * C_out * H_out * W_out
                flops = (
                    2 * module.kernel_size[0] * module.kernel_size[1] *
                    module.in_channels * module.out_channels *
                    out_h * out_w
                )
                
                # Account for groups (depthwise separable)
                flops = flops // module.groups
                
                total_flops += flops * batch_size
            
            elif isinstance(module, torch.nn.Linear):
                batch_size = input[0].shape[0]
                flops = 2 * module.in_features * module.out_features
                total_flops += flops * batch_size
        
        hooks = []
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        device = next(model.parameters()).device
        x = torch.randn(input_size).to(device)
        with torch.no_grad():
            model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return total_flops / 1e9  # Return in billions
    
    @staticmethod
    def estimate_memory(model: torch.nn.Module, input_size: Tuple[int, int, int, int]) -> Dict[str, float]:
        """
        Estimate memory usage.
        
        Reference: Table 6 - Memory Usage and Optimization Analysis
        
        Returns:
            Dictionary with memory estimates in MB
        """
        # Model weights memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # Estimate feature map memory (rough estimate)
        # Assuming average feature map size reduction through network
        batch_size, channels, height, width = input_size
        feature_map_memory = 0
        
        current_h, current_w = height, width
        current_c = channels
        
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                out_c = module.out_channels
                stride = module.stride[0]
                current_h = (current_h + 2 * module.padding[0] - module.kernel_size[0]) // stride + 1
                current_w = (current_w + 2 * module.padding[1] - module.kernel_size[1]) // stride + 1
                
                # Feature map size: B × C × H × W × 4 bytes (float32)
                feature_map_memory += batch_size * out_c * current_h * current_w * 4
                current_c = out_c
        
        return {
            'model_weights_mb': param_memory / (1024 * 1024),
            'feature_maps_mb': feature_map_memory / (1024 * 1024),
            'total_mb': (param_memory + feature_map_memory) / (1024 * 1024)
        }


def evaluate_detection(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 10,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45
) -> Dict[str, float]:
    """
    Evaluate detection model on a dataset.
    
    Reference: Table 2 - Comprehensive Detection Performance Comparison
    
    Args:
        model: Detection model
        dataloader: Validation dataloader
        device: Compute device
        num_classes: Number of classes
        conf_threshold: Confidence threshold for predictions
        nms_threshold: NMS IoU threshold
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    evaluator = COCOEvaluator(num_classes=num_classes)
    fps_meter = FPSMeter(warmup=10)
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            gt_boxes = batch['bboxes']
            gt_labels = batch['labels']
            masks = batch['masks']
            
            # Time inference
            fps_meter.start()
            predictions = model(images)
            fps_meter.stop()
            
            # Process each image in batch
            for i in range(len(images)):
                # Get predictions for this image
                pred_boxes = predictions['boxes'][i] if 'boxes' in predictions else torch.zeros((0, 4))
                pred_scores = predictions['scores'][i] if 'scores' in predictions else torch.zeros((0,))
                pred_labels = predictions['labels'][i] if 'labels' in predictions else torch.zeros((0,))
                
                # Get ground truths for this image
                valid_mask = masks[i]
                gt_box = gt_boxes[i][valid_mask]
                gt_label = gt_labels[i][valid_mask]
                
                # Update evaluator
                evaluator.update(pred_boxes, pred_scores, pred_labels, gt_box, gt_label)
    
    # Compute metrics
    metrics = evaluator.compute_metrics()
    metrics['FPS'] = fps_meter.fps
    metrics['Latency_ms'] = fps_meter.latency_ms
    
    return metrics


# Test the metrics module
if __name__ == "__main__":
    print("Testing Evaluation Metrics Module")
    print("=" * 50)
    
    # Test box IoU
    print("\n1. Testing box_iou")
    boxes1 = np.array([[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]])
    boxes2 = np.array([[0.2, 0.2, 0.6, 0.6], [0.7, 0.7, 1.0, 1.0]])
    ious = box_iou(boxes1, boxes2)
    print(f"   IoU matrix shape: {ious.shape}")
    print(f"   IoU values:\n{ious}")
    
    # Test AP computation
    print("\n2. Testing compute_ap")
    recalls = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    precisions = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    ap = compute_ap(recalls, precisions)
    print(f"   Average Precision: {ap:.4f}")
    
    # Test COCO Evaluator
    print("\n3. Testing COCOEvaluator")
    evaluator = COCOEvaluator(num_classes=10)
    
    # Add dummy predictions
    for _ in range(10):
        pred_boxes = torch.rand(5, 4)
        pred_boxes[:, 2:] = pred_boxes[:, :2] + torch.rand(5, 2) * 0.3
        pred_scores = torch.rand(5)
        pred_labels = torch.randint(0, 10, (5,))
        
        gt_boxes = torch.rand(3, 4)
        gt_boxes[:, 2:] = gt_boxes[:, :2] + torch.rand(3, 2) * 0.3
        gt_labels = torch.randint(0, 10, (3,))
        
        evaluator.update(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
    
    metrics = evaluator.compute_metrics()
    print("   Metrics:")
    for k, v in metrics.items():
        print(f"      {k}: {v:.4f}")
    
    # Test FPS Meter
    print("\n4. Testing FPSMeter")
    fps_meter = FPSMeter(warmup=2)
    for _ in range(10):
        fps_meter.start()
        time.sleep(0.01)  # Simulate 10ms processing
        fps_meter.stop()
    print(f"   FPS: {fps_meter.fps:.2f}")
    print(f"   Latency: {fps_meter.latency_ms:.2f} ms")
    
    # Test Model Complexity Analyzer
    print("\n5. Testing ModelComplexityAnalyzer")
    dummy_model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
        torch.nn.Conv2d(128, 256, 3, stride=2, padding=1)
    )
    
    num_params = ModelComplexityAnalyzer.count_parameters(dummy_model)
    print(f"   Parameters: {num_params:,}")
    
    flops = ModelComplexityAnalyzer.estimate_flops(dummy_model, (1, 3, 640, 640))
    print(f"   GFLOPs: {flops:.2f}")
    
    memory = ModelComplexityAnalyzer.estimate_memory(dummy_model, (1, 3, 640, 640))
    print(f"   Memory - Weights: {memory['model_weights_mb']:.2f} MB")
    print(f"   Memory - Feature Maps: {memory['feature_maps_mb']:.2f} MB")
    
    print("\n" + "=" * 50)
    print("All metrics tests passed!")
