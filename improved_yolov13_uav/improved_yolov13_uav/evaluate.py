"""
Evaluation Script for Improved YOLOv13 UAV Obstacle Avoidance
Implements comprehensive evaluation including:
- mAP at different IoU thresholds
- Scale-based metrics (APS, APM, APL)
- Per-class performance analysis
- Complex scenario evaluation (night, fog, occlusion)
- Speed and latency benchmarking
"""

import os
import sys
import time
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import ImprovedYOLOv13
from data import (
    VisDroneDataset,
    UAVDTDataset,
    CombinedUAVDataset,
    create_dataloader,
    ValAugmentation
)
from utils import (
    DetectionMetrics,
    ConfusionMatrix,
    compute_ap,
    compute_map,
    box_iou,
    xywh2xyxy,
    non_max_suppression
)


class ComplexScenarioEvaluator:
    """
    Evaluator for complex scenario performance
    Tests algorithm robustness under various challenging conditions
    """
    
    # VisDrone category names
    VISDRONE_CLASSES = [
        'pedestrian', 'people', 'bicycle', 'car', 'van',
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
    
    # Scenario categories
    SCENARIOS = {
        'sunny': {'weather': 'clear', 'time': 'day'},
        'cloudy': {'weather': 'cloudy', 'time': 'day'},
        'foggy': {'weather': 'fog', 'time': 'day'},
        'rainy': {'weather': 'rain', 'time': 'day'},
        'night': {'weather': 'clear', 'time': 'night'},
        'crowded': {'density': 'high'}
    }
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int = 10,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45
    ):
        """
        Initialize evaluator
        
        Args:
            model: Model to evaluate
            device: Compute device
            num_classes: Number of object classes
            conf_threshold: Confidence threshold for detection
            nms_threshold: NMS IoU threshold
        """
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        self.model.eval()
    
    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate on entire dataset
        
        Args:
            dataloader: Data loader
            verbose: Print progress
            
        Returns:
            Dictionary of metrics
        """
        all_predictions = []
        all_targets = []
        all_image_ids = []
        
        iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader
        
        for batch in iterator:
            images = batch['image'].to(self.device)
            targets = batch['targets']
            image_ids = batch.get('image_id', list(range(len(images))))
            
            # Get predictions
            predictions = self.model.predict(
                images,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold
            )
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            all_image_ids.extend(image_ids)
        
        # Compute metrics
        metrics = self._compute_coco_metrics(all_predictions, all_targets)
        
        return metrics
    
    def _compute_coco_metrics(
        self,
        predictions: List,
        targets: List
    ) -> Dict[str, float]:
        """
        Compute COCO-style metrics
        
        Args:
            predictions: List of predictions
            targets: List of ground truth targets
            
        Returns:
            Dictionary of metrics
        """
        # Initialize accumulators
        all_ious = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        # Per-class AP storage
        class_aps = {iou: defaultdict(list) for iou in all_ious}
        
        # Scale-based storage (APS, APM, APL)
        # APS: < 32x32, APM: 32x32 to 96x96, APL: > 96x96
        scale_aps = {
            'small': defaultdict(list),
            'medium': defaultdict(list),
            'large': defaultdict(list)
        }
        
        for preds, gts in zip(predictions, targets):
            if len(gts) == 0:
                continue
            
            # Convert targets to numpy if needed
            if isinstance(gts, torch.Tensor):
                gts = gts.cpu().numpy()
            
            for iou_thresh in all_ious:
                # Match predictions to ground truth
                matches = self._match_predictions(preds, gts, iou_thresh)
                
                # Accumulate per-class results
                for cls_id in range(self.num_classes):
                    cls_preds = [p for p in preds if int(p[5]) == cls_id]
                    cls_gts = gts[gts[:, 0] == cls_id] if len(gts) > 0 else []
                    
                    if len(cls_gts) == 0 and len(cls_preds) == 0:
                        continue
                    
                    # Compute precision-recall for this class
                    tp, fp, num_gt = self._compute_tp_fp(
                        cls_preds, cls_gts, iou_thresh
                    )
                    
                    if num_gt > 0:
                        ap = compute_ap(tp, fp, num_gt)
                        class_aps[iou_thresh][cls_id].append(ap)
            
            # Scale-based metrics (at IoU=0.5)
            for gt in gts:
                area = gt[3] * gt[4]  # width * height
                
                if area < 32 * 32:
                    scale = 'small'
                elif area < 96 * 96:
                    scale = 'medium'
                else:
                    scale = 'large'
                
                cls_id = int(gt[0])
                cls_preds = [p for p in preds if int(p[5]) == cls_id]
                
                # Check if this GT is detected
                detected = False
                for pred in cls_preds:
                    iou = self._compute_iou(pred[:4], gt[1:5])
                    if iou >= 0.5:
                        detected = True
                        break
                
                scale_aps[scale][cls_id].append(1.0 if detected else 0.0)
        
        # Aggregate metrics
        metrics = {}
        
        # mAP at different IoU thresholds
        for iou in all_ious:
            class_mean_aps = []
            for cls_id in range(self.num_classes):
                if class_aps[iou][cls_id]:
                    class_mean_aps.append(np.mean(class_aps[iou][cls_id]))
            
            if class_mean_aps:
                metrics[f'mAP@{iou}'] = np.mean(class_mean_aps)
        
        # Standard metrics
        metrics['mAP50'] = metrics.get('mAP@0.5', 0.0)
        metrics['mAP75'] = metrics.get('mAP@0.75', 0.0)
        
        # mAP (average over IoU 0.5:0.95)
        map_values = [metrics.get(f'mAP@{iou}', 0.0) for iou in all_ious]
        metrics['mAP'] = np.mean(map_values)
        
        # Scale-based metrics
        for scale in ['small', 'medium', 'large']:
            scale_values = []
            for cls_id in range(self.num_classes):
                if scale_aps[scale][cls_id]:
                    scale_values.extend(scale_aps[scale][cls_id])
            
            if scale_values:
                key = {'small': 'APS', 'medium': 'APM', 'large': 'APL'}[scale]
                metrics[key] = np.mean(scale_values)
        
        return metrics
    
    def _match_predictions(
        self,
        predictions: List,
        targets: np.ndarray,
        iou_threshold: float
    ) -> Dict:
        """Match predictions to ground truth"""
        matches = {}
        
        for pred_idx, pred in enumerate(predictions):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(targets):
                if int(pred[5]) != int(gt[0]):  # Class mismatch
                    continue
                
                iou = self._compute_iou(pred[:4], gt[1:5])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                matches[pred_idx] = best_gt_idx
        
        return matches
    
    def _compute_tp_fp(
        self,
        predictions: List,
        targets: np.ndarray,
        iou_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Compute true positives and false positives"""
        num_gt = len(targets)
        num_pred = len(predictions)
        
        if num_pred == 0:
            return np.array([]), np.array([]), num_gt
        
        # Sort predictions by confidence
        predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
        
        tp = np.zeros(num_pred)
        fp = np.zeros(num_pred)
        matched_gt = set()
        
        for pred_idx, pred in enumerate(predictions):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(targets):
                if gt_idx in matched_gt:
                    continue
                
                iou = self._compute_iou(pred[:4], gt[1:5])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                tp[pred_idx] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[pred_idx] = 1
        
        return tp, fp, num_gt
    
    def _compute_iou(
        self,
        box1: np.ndarray,
        box2: np.ndarray
    ) -> float:
        """Compute IoU between two boxes"""
        # Convert to xyxy format if needed
        if len(box1) == 4 and len(box2) == 4:
            # Assume box1 is [x1, y1, x2, y2] and box2 is [cx, cy, w, h]
            x1 = max(box1[0], box2[0] - box2[2] / 2)
            y1 = max(box1[1], box2[1] - box2[3] / 2)
            x2 = min(box1[2], box2[0] + box2[2] / 2)
            y2 = min(box1[3], box2[1] + box2[3] / 2)
        else:
            return 0.0
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = box2[2] * box2[3]
        
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    @torch.no_grad()
    def evaluate_by_scenario(
        self,
        dataset,
        scenario_annotations: Dict[int, str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance by scenario type
        
        Args:
            dataset: Evaluation dataset
            scenario_annotations: Mapping of image_id to scenario type
            
        Returns:
            Dictionary of metrics per scenario
        """
        scenario_results = defaultdict(list)
        
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image_id = sample.get('image_id', idx)
            scenario = scenario_annotations.get(image_id, 'unknown')
            
            # Get prediction
            image = sample['image'].unsqueeze(0).to(self.device)
            predictions = self.model.predict(
                image,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold
            )
            
            # Compute metrics for this image
            targets = sample['targets']
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu().numpy()
            
            # Simple mAP@0.5 for this image
            tp = 0
            for pred in predictions[0]:
                for gt in targets:
                    if int(pred[5]) == int(gt[0]):
                        iou = self._compute_iou(pred[:4], gt[1:5])
                        if iou >= 0.5:
                            tp += 1
                            break
            
            precision = tp / len(predictions[0]) if predictions[0] else 0
            recall = tp / len(targets) if len(targets) > 0 else 0
            
            scenario_results[scenario].append({
                'precision': precision,
                'recall': recall,
                'num_predictions': len(predictions[0]),
                'num_targets': len(targets)
            })
        
        # Aggregate by scenario
        metrics_by_scenario = {}
        for scenario, results in scenario_results.items():
            total_tp = sum(r['precision'] * r['num_predictions'] for r in results)
            total_pred = sum(r['num_predictions'] for r in results)
            total_gt = sum(r['num_targets'] for r in results)
            
            metrics_by_scenario[scenario] = {
                'precision': total_tp / total_pred if total_pred > 0 else 0,
                'recall': total_tp / total_gt if total_gt > 0 else 0,
                'num_images': len(results)
            }
        
        return metrics_by_scenario
    
    @torch.no_grad()
    def evaluate_by_altitude(
        self,
        dataset,
        altitude_annotations: Dict[int, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance by flight altitude
        
        Args:
            dataset: Evaluation dataset
            altitude_annotations: Mapping of image_id to altitude (meters)
            
        Returns:
            Dictionary of metrics per altitude range
        """
        altitude_ranges = [
            (0, 30, '0-30m'),
            (30, 60, '30-60m'),
            (60, 90, '60-90m'),
            (90, 150, '90-150m')
        ]
        
        altitude_results = defaultdict(list)
        
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image_id = sample.get('image_id', idx)
            altitude = altitude_annotations.get(image_id, 60)  # Default 60m
            
            # Determine altitude range
            altitude_range = 'unknown'
            for low, high, name in altitude_ranges:
                if low <= altitude < high:
                    altitude_range = name
                    break
            
            # Get prediction
            image = sample['image'].unsqueeze(0).to(self.device)
            predictions = self.model.predict(
                image,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold
            )
            
            targets = sample['targets']
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu().numpy()
            
            # Compute metrics
            tp = 0
            for pred in predictions[0]:
                for gt in targets:
                    if int(pred[5]) == int(gt[0]):
                        iou = self._compute_iou(pred[:4], gt[1:5])
                        if iou >= 0.5:
                            tp += 1
                            break
            
            altitude_results[altitude_range].append({
                'tp': tp,
                'num_predictions': len(predictions[0]),
                'num_targets': len(targets)
            })
        
        # Aggregate
        metrics_by_altitude = {}
        for altitude_range, results in altitude_results.items():
            total_tp = sum(r['tp'] for r in results)
            total_pred = sum(r['num_predictions'] for r in results)
            total_gt = sum(r['num_targets'] for r in results)
            
            metrics_by_altitude[altitude_range] = {
                'precision': total_tp / total_pred if total_pred > 0 else 0,
                'recall': total_tp / total_gt if total_gt > 0 else 0,
                'mAP50': total_tp / total_gt if total_gt > 0 else 0,
                'num_images': len(results)
            }
        
        return metrics_by_altitude
    
    @torch.no_grad()
    def evaluate_by_occlusion(
        self,
        dataset,
        occlusion_annotations: Dict[int, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance by occlusion level
        
        Args:
            dataset: Evaluation dataset
            occlusion_annotations: Mapping of image_id to list of occlusion ratios
            
        Returns:
            Dictionary of metrics per occlusion level
        """
        occlusion_levels = [
            (0, 0, 'none'),           # 100% visible
            (0, 0.3, 'light'),        # 70-100% visible
            (0.3, 0.6, 'moderate'),   # 40-70% visible
            (0.6, 1.0, 'heavy')       # <40% visible
        ]
        
        occlusion_results = defaultdict(list)
        
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image_id = sample.get('image_id', idx)
            
            # Get prediction
            image = sample['image'].unsqueeze(0).to(self.device)
            predictions = self.model.predict(
                image,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold
            )
            
            targets = sample['targets']
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu().numpy()
            
            # Get occlusion ratios for this image
            occlusions = occlusion_annotations.get(image_id, [0] * len(targets))
            
            for gt_idx, gt in enumerate(targets):
                occ_ratio = occlusions[gt_idx] if gt_idx < len(occlusions) else 0
                
                # Determine occlusion level
                occ_level = 'none'
                for low, high, name in occlusion_levels:
                    if low <= occ_ratio < high:
                        occ_level = name
                        break
                
                # Check if this target is detected
                detected = False
                for pred in predictions[0]:
                    if int(pred[5]) == int(gt[0]):
                        iou = self._compute_iou(pred[:4], gt[1:5])
                        if iou >= 0.5:
                            detected = True
                            break
                
                occlusion_results[occ_level].append(1 if detected else 0)
        
        # Aggregate
        metrics_by_occlusion = {}
        for occ_level, results in occlusion_results.items():
            metrics_by_occlusion[occ_level] = {
                'recall': np.mean(results) if results else 0,
                'num_targets': len(results)
            }
        
        return metrics_by_occlusion


class SpeedBenchmark:
    """
    Benchmark inference speed and latency
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        input_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize benchmarker
        
        Args:
            model: Model to benchmark
            device: Compute device
            input_size: Input image size (H, W)
        """
        self.model = model
        self.device = device
        self.input_size = input_size
        
        self.model.eval()
    
    @torch.no_grad()
    def benchmark(
        self,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        batch_size: int = 1
    ) -> Dict[str, float]:
        """
        Run speed benchmark
        
        Args:
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            batch_size: Batch size for inference
            
        Returns:
            Dictionary of timing metrics
        """
        # Create dummy input
        dummy_input = torch.randn(
            batch_size, 3, self.input_size[0], self.input_size[1]
        ).to(self.device)
        
        # Warmup
        for _ in range(warmup_iterations):
            _ = self.model(dummy_input)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            
            _ = self.model(dummy_input)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        return {
            'mean_latency_ms': np.mean(times),
            'std_latency_ms': np.std(times),
            'min_latency_ms': np.min(times),
            'max_latency_ms': np.max(times),
            'fps': 1000 / np.mean(times) * batch_size,
            'throughput_images_per_sec': 1000 / np.mean(times) * batch_size
        }
    
    @torch.no_grad()
    def benchmark_components(
        self,
        num_iterations: int = 50
    ) -> Dict[str, float]:
        """
        Benchmark individual model components
        
        Returns:
            Dictionary of component timings
        """
        dummy_input = torch.randn(
            1, 3, self.input_size[0], self.input_size[1]
        ).to(self.device)
        
        component_times = {}
        
        # This requires model to have accessible components
        # Simplified version for demonstration
        
        # Full forward pass
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self.model(dummy_input)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        component_times['total'] = np.mean(times)
        
        return component_times


def load_model(
    checkpoint_path: str,
    config: Dict,
    device: torch.device
) -> nn.Module:
    """Load model from checkpoint"""
    model_cfg = config.get('model', {})
    
    model = ImprovedYOLOv13(
        num_classes=model_cfg.get('num_classes', 10),
        variant=model_cfg.get('variant', 'small'),
        in_channels=model_cfg.get('in_channels', 3)
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def main():
    """Main evaluation entry point"""
    parser = argparse.ArgumentParser(description='Evaluate Improved YOLOv13')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='./data',
        help='Path to data directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='runs/eval',
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.25,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.45,
        help='NMS threshold'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run speed benchmark'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, config, device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create validation transform
    val_transform = ValAugmentation(
        img_size=config.get('data', {}).get('img_size', 640)
    )
    
    # Create dataset
    print("Loading dataset...")
    data_cfg = config.get('data', {})
    val_dataset = CombinedUAVDataset(
        visdrone_path=data_cfg.get('visdrone_path', f'{args.data}/VisDrone2019'),
        uavdt_path=data_cfg.get('uavdt_path', f'{args.data}/UAVDT'),
        split='val',
        transform=val_transform
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create evaluator
    evaluator = ComplexScenarioEvaluator(
        model=model,
        device=device,
        num_classes=config.get('model', {}).get('num_classes', 10),
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold
    )
    
    # Run evaluation
    print("\nRunning evaluation...")
    metrics = evaluator.evaluate_dataset(val_loader)
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"mAP@0.5: {metrics.get('mAP50', 0):.4f}")
    print(f"mAP@0.75: {metrics.get('mAP75', 0):.4f}")
    print(f"mAP@0.5:0.95: {metrics.get('mAP', 0):.4f}")
    print(f"APS (small): {metrics.get('APS', 0):.4f}")
    print(f"APM (medium): {metrics.get('APM', 0):.4f}")
    print(f"APL (large): {metrics.get('APL', 0):.4f}")
    
    # Run speed benchmark if requested
    if args.benchmark:
        print("\n" + "=" * 50)
        print("Speed Benchmark")
        print("=" * 50)
        
        benchmarker = SpeedBenchmark(
            model=model,
            device=device,
            input_size=(640, 640)
        )
        
        # Benchmark different batch sizes
        for bs in [1, 4, 8, 16]:
            speed_metrics = benchmarker.benchmark(
                num_iterations=100,
                warmup_iterations=10,
                batch_size=bs
            )
            
            print(f"\nBatch size {bs}:")
            print(f"  Mean latency: {speed_metrics['mean_latency_ms']:.2f} ms")
            print(f"  FPS: {speed_metrics['fps']:.1f}")
            print(f"  Throughput: {speed_metrics['throughput_images_per_sec']:.1f} images/sec")
            
            metrics[f'fps_bs{bs}'] = speed_metrics['fps']
            metrics[f'latency_bs{bs}'] = speed_metrics['mean_latency_ms']
    
    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
