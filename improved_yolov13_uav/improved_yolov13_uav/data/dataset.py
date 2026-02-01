"""
Dataset Module for Improved YOLOv13 UAV Obstacle Avoidance
==========================================================

Implements dataset loading for:
- VisDrone2019: 10,209 images, 288 video sequences, 540,816 annotated instances, 10 categories
- UAVDT: 80,000 frames, 100 video sequences, 841,506 annotated instances, 3 categories

Reference: Section 4.1 - Table 1: Detailed Statistics of Experimental Datasets
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Any
import random
from pathlib import Path

from .augmentation import (
    MosaicAugmentation, MixUpAugmentation, get_train_transforms, get_val_transforms,
    Normalize, ColorJitter, RandomCrop, RandomHorizontalFlip, LetterBox
)


# VisDrone2019 class names (10 categories)
VISDRONE_CLASSES = [
    'pedestrian',      # 0
    'people',          # 1
    'bicycle',         # 2
    'car',             # 3
    'van',             # 4
    'truck',           # 5
    'tricycle',        # 6
    'awning-tricycle', # 7
    'bus',             # 8
    'motor'            # 9
]

# UAVDT class names (3 categories)
UAVDT_CLASSES = [
    'car',    # 0
    'truck',  # 1
    'bus'     # 2
]


class VisDroneDataset(Dataset):
    """
    VisDrone2019 Dataset
    
    Dataset Statistics (Table 1):
    - Image Count: 10,209
    - Video Sequences: 288
    - Annotated Instances: 540,816
    - Categories: 10
    - Resolution: 2000×1500
    - Small Target Ratio: 62.30%
    - Scene Type: Urban/Suburban
    
    Training set: 6,471 images with 343,207 annotated instances
    Validation set: 548 images with 37,989 instances
    Test set: 1,610 images with 90,620 instances
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        img_size: int = 640,
        augment: bool = True,
        mosaic: bool = True,
        mixup: bool = True,
        mosaic_prob: float = 1.0,
        mixup_prob: float = 0.2
    ):
        """
        Initialize VisDrone dataset.
        
        Args:
            root: Root directory of VisDrone dataset
            split: 'train', 'val', or 'test'
            img_size: Target image size
            augment: Whether to apply augmentation
            mosaic: Whether to use mosaic augmentation
            mixup: Whether to use mixup augmentation
            mosaic_prob: Probability of applying mosaic
            mixup_prob: Probability of applying mixup
        """
        super().__init__()
        
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        self.mosaic = mosaic
        self.mixup = mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        
        self.classes = VISDRONE_CLASSES
        self.num_classes = len(self.classes)
        
        # Set paths based on split
        if split == 'train':
            self.img_dir = self.root / 'VisDrone2019-DET-train' / 'images'
            self.ann_dir = self.root / 'VisDrone2019-DET-train' / 'annotations'
        elif split == 'val':
            self.img_dir = self.root / 'VisDrone2019-DET-val' / 'images'
            self.ann_dir = self.root / 'VisDrone2019-DET-val' / 'annotations'
        else:  # test
            self.img_dir = self.root / 'VisDrone2019-DET-test-dev' / 'images'
            self.ann_dir = self.root / 'VisDrone2019-DET-test-dev' / 'annotations'
        
        # Load image list
        self.img_files = self._load_image_list()
        
        # Initialize augmentations
        if self.augment:
            self.mosaic_aug = MosaicAugmentation(img_size=img_size)
            self.mixup_aug = MixUpAugmentation(alpha=0.2)  # Paper: coefficient 0.2
            self.transforms = get_train_transforms(img_size)
        else:
            self.transforms = get_val_transforms(img_size)
        
        self.normalize = Normalize()
    
    def _load_image_list(self) -> List[str]:
        """Load list of image files."""
        if self.img_dir.exists():
            img_files = sorted([
                str(f) for f in self.img_dir.glob('*.jpg')
            ])
        else:
            # Create dummy data for testing
            img_files = [f'dummy_{i}.jpg' for i in range(100)]
        return img_files
    
    def _parse_visdrone_annotation(self, ann_path: str, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse VisDrone annotation file.
        
        VisDrone annotation format:
        <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
        
        Categories: ignored(0), pedestrian(1), people(2), bicycle(3), car(4), van(5), 
                   truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10)
        
        Note: We map categories 1-10 to 0-9 (ignoring category 0)
        """
        bboxes = []
        labels = []
        
        ann_path = Path(ann_path)
        if ann_path.exists():
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                        category = int(parts[5])
                        
                        # Skip ignored regions (category 0) and non-target categories (11)
                        if category == 0 or category > 10:
                            continue
                        
                        # Convert to normalized xyxy format
                        x1 = x / img_w
                        y1 = y / img_h
                        x2 = (x + w) / img_w
                        y2 = (y + h) / img_h
                        
                        # Clip to valid range
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(1, x2), min(1, y2)
                        
                        if x2 > x1 and y2 > y1:  # Valid box
                            bboxes.append([x1, y1, x2, y2])
                            labels.append(category - 1)  # Map 1-10 to 0-9
        
        return np.array(bboxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        # Mosaic augmentation
        if self.augment and self.mosaic and random.random() < self.mosaic_prob:
            # Get 4 random indices
            indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
            images, bboxes_list, labels_list = [], [], []
            
            for i in indices:
                img, bboxes, labels = self._load_sample(i)
                images.append(img)
                bboxes_list.append(bboxes)
                labels_list.append(labels)
            
            image, bboxes, labels = self.mosaic_aug(images, bboxes_list, labels_list)
            
            # Apply MixUp after Mosaic
            if self.mixup and random.random() < self.mixup_prob:
                idx2 = random.randint(0, len(self) - 1)
                img2, bboxes2, labels2 = self._load_sample(idx2)
                img2, bboxes2, labels2 = self.transforms(img2, bboxes2, labels2)
                image, bboxes, labels = self.mixup_aug(
                    image, bboxes, labels,
                    img2, bboxes2, labels2
                )
        else:
            image, bboxes, labels = self._load_sample(idx)
            image, bboxes, labels = self.transforms(image, bboxes, labels)
        
        # Normalize
        image = self.normalize(image)
        
        # Convert to torch tensors
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        bboxes = torch.from_numpy(bboxes).float() if len(bboxes) > 0 else torch.zeros((0, 4))
        labels = torch.from_numpy(labels).long() if len(labels) > 0 else torch.zeros((0,), dtype=torch.long)
        
        return {
            'image': image,
            'bboxes': bboxes,
            'labels': labels,
            'img_id': idx
        }
    
    def _load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a single image and its annotations."""
        img_path = self.img_files[idx]
        
        # Load image
        if os.path.exists(img_path):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Create dummy image for testing
            image = np.random.randint(0, 255, (1500, 2000, 3), dtype=np.uint8)
        
        img_h, img_w = image.shape[:2]
        
        # Load annotations
        ann_path = str(img_path).replace('images', 'annotations').replace('.jpg', '.txt')
        bboxes, labels = self._parse_visdrone_annotation(ann_path, img_w, img_h)
        
        return image, bboxes, labels


class UAVDTDataset(Dataset):
    """
    UAVDT Benchmark Dataset
    
    Dataset Statistics (Table 1):
    - Image Count: 80,000
    - Video Sequences: 100
    - Annotated Instances: 841,506
    - Categories: 3 (car, truck, bus)
    - Resolution: 1024×540
    - Small Target Ratio: 38.90%
    - Scene Type: Traffic Scenes
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        img_size: int = 640,
        augment: bool = True,
        mosaic: bool = True,
        mixup: bool = True,
        mosaic_prob: float = 1.0,
        mixup_prob: float = 0.2
    ):
        super().__init__()
        
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        self.mosaic = mosaic
        self.mixup = mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        
        self.classes = UAVDT_CLASSES
        self.num_classes = len(self.classes)
        
        # Load image list
        self.img_files = self._load_image_list()
        
        # Initialize augmentations
        if self.augment:
            self.mosaic_aug = MosaicAugmentation(img_size=img_size)
            self.mixup_aug = MixUpAugmentation(alpha=0.2)
            self.transforms = get_train_transforms(img_size)
        else:
            self.transforms = get_val_transforms(img_size)
        
        self.normalize = Normalize()
    
    def _load_image_list(self) -> List[str]:
        """Load list of image files."""
        if self.root.exists():
            img_files = []
            for seq_dir in sorted(self.root.glob('UAV-benchmark-M/*/img1')):
                img_files.extend(sorted([str(f) for f in seq_dir.glob('*.jpg')]))
        else:
            img_files = [f'dummy_{i}.jpg' for i in range(100)]
        return img_files
    
    def _parse_uavdt_annotation(self, img_path: str, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse UAVDT annotation file.
        
        UAVDT annotation format (gt_whole.txt):
        <frame_id>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<out-of-view>,<occlusion>,<object_category>
        
        Categories: car(1), truck(2), bus(3)
        """
        bboxes = []
        labels = []
        
        # Get sequence directory and frame number
        img_path = Path(img_path)
        frame_num = int(img_path.stem)
        gt_path = img_path.parent.parent / 'gt' / 'gt_whole.txt'
        
        if gt_path.exists():
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 9:
                        frame_id = int(parts[0])
                        if frame_id == frame_num:
                            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                            category = int(parts[8])
                            
                            if category < 1 or category > 3:
                                continue
                            
                            # Convert to normalized xyxy
                            x1 = x / img_w
                            y1 = y / img_h
                            x2 = (x + w) / img_w
                            y2 = (y + h) / img_h
                            
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(1, x2), min(1, y2)
                            
                            if x2 > x1 and y2 > y1:
                                bboxes.append([x1, y1, x2, y2])
                                labels.append(category - 1)  # Map 1-3 to 0-2
        
        return np.array(bboxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self.img_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        if self.augment and self.mosaic and random.random() < self.mosaic_prob:
            indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
            images, bboxes_list, labels_list = [], [], []
            
            for i in indices:
                img, bboxes, labels = self._load_sample(i)
                images.append(img)
                bboxes_list.append(bboxes)
                labels_list.append(labels)
            
            image, bboxes, labels = self.mosaic_aug(images, bboxes_list, labels_list)
            
            if self.mixup and random.random() < self.mixup_prob:
                idx2 = random.randint(0, len(self) - 1)
                img2, bboxes2, labels2 = self._load_sample(idx2)
                img2, bboxes2, labels2 = self.transforms(img2, bboxes2, labels2)
                image, bboxes, labels = self.mixup_aug(
                    image, bboxes, labels,
                    img2, bboxes2, labels2
                )
        else:
            image, bboxes, labels = self._load_sample(idx)
            image, bboxes, labels = self.transforms(image, bboxes, labels)
        
        image = self.normalize(image)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        bboxes = torch.from_numpy(bboxes).float() if len(bboxes) > 0 else torch.zeros((0, 4))
        labels = torch.from_numpy(labels).long() if len(labels) > 0 else torch.zeros((0,), dtype=torch.long)
        
        return {
            'image': image,
            'bboxes': bboxes,
            'labels': labels,
            'img_id': idx
        }
    
    def _load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a single image and its annotations."""
        img_path = self.img_files[idx]
        
        if os.path.exists(img_path):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.random.randint(0, 255, (540, 1024, 3), dtype=np.uint8)
        
        img_h, img_w = image.shape[:2]
        bboxes, labels = self._parse_uavdt_annotation(img_path, img_w, img_h)
        
        return image, bboxes, labels


class CombinedDataset(Dataset):
    """
    Combined Dataset (VisDrone2019 + UAVDT)
    
    Dataset Statistics (Table 1):
    - Image Count: 90,209
    - Video Sequences: 388
    - Annotated Instances: 1,382,322
    - Categories: 10
    - Resolution: Various
    - Small Target Ratio: 50.60%
    - Scene Type: Comprehensive
    """
    
    def __init__(
        self,
        visdrone_root: str,
        uavdt_root: str,
        split: str = 'train',
        img_size: int = 640,
        augment: bool = True,
        mosaic: bool = True,
        mixup: bool = True
    ):
        super().__init__()
        
        self.visdrone = VisDroneDataset(
            visdrone_root, split, img_size, augment, mosaic, mixup
        )
        self.uavdt = UAVDTDataset(
            uavdt_root, split, img_size, augment, mosaic, mixup
        )
        
        self.visdrone_len = len(self.visdrone)
        self.uavdt_len = len(self.uavdt)
        
        # Use VisDrone classes as unified class set
        self.classes = VISDRONE_CLASSES
        self.num_classes = len(self.classes)
        
        # UAVDT to VisDrone class mapping
        # UAVDT: car(0), truck(1), bus(2) -> VisDrone: car(3), truck(5), bus(8)
        self.uavdt_to_visdrone = {0: 3, 1: 5, 2: 8}
    
    def __len__(self) -> int:
        return self.visdrone_len + self.uavdt_len
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < self.visdrone_len:
            return self.visdrone[idx]
        else:
            sample = self.uavdt[idx - self.visdrone_len]
            # Map UAVDT labels to VisDrone labels
            for i, label in enumerate(sample['labels']):
                sample['labels'][i] = self.uavdt_to_visdrone[label.item()]
            return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for variable-length annotations."""
    images = torch.stack([item['image'] for item in batch])
    
    # Pad bboxes and labels to same length
    max_boxes = max(len(item['bboxes']) for item in batch)
    
    batch_bboxes = []
    batch_labels = []
    batch_masks = []
    
    for item in batch:
        num_boxes = len(item['bboxes'])
        if num_boxes > 0:
            padded_bboxes = torch.zeros((max_boxes, 4))
            padded_bboxes[:num_boxes] = item['bboxes']
            padded_labels = torch.zeros(max_boxes, dtype=torch.long)
            padded_labels[:num_boxes] = item['labels']
            mask = torch.zeros(max_boxes, dtype=torch.bool)
            mask[:num_boxes] = True
        else:
            padded_bboxes = torch.zeros((max_boxes, 4))
            padded_labels = torch.zeros(max_boxes, dtype=torch.long)
            mask = torch.zeros(max_boxes, dtype=torch.bool)
        
        batch_bboxes.append(padded_bboxes)
        batch_labels.append(padded_labels)
        batch_masks.append(mask)
    
    return {
        'images': images,
        'bboxes': torch.stack(batch_bboxes),
        'labels': torch.stack(batch_labels),
        'masks': torch.stack(batch_masks),
        'img_ids': [item['img_id'] for item in batch]
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """Create a DataLoader with custom collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True if shuffle else False
    )


# Initialize __init__.py for data module
if __name__ == "__main__":
    print("Testing Dataset Module")
    print("=" * 50)
    
    # Test VisDrone dataset (with dummy data)
    print("\n1. Testing VisDroneDataset")
    visdrone = VisDroneDataset(
        root='./data/visdrone',
        split='train',
        img_size=640,
        augment=True,
        mosaic=True,
        mixup=True
    )
    print(f"   Classes: {visdrone.num_classes}")
    print(f"   Dataset length: {len(visdrone)}")
    
    # Test sample
    sample = visdrone[0]
    print(f"   Sample image shape: {sample['image'].shape}")
    print(f"   Sample bboxes shape: {sample['bboxes'].shape}")
    print(f"   Sample labels shape: {sample['labels'].shape}")
    
    # Test UAVDT dataset
    print("\n2. Testing UAVDTDataset")
    uavdt = UAVDTDataset(
        root='./data/uavdt',
        split='train',
        img_size=640,
        augment=True
    )
    print(f"   Classes: {uavdt.num_classes}")
    print(f"   Dataset length: {len(uavdt)}")
    
    # Test DataLoader
    print("\n3. Testing DataLoader")
    dataloader = create_dataloader(visdrone, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    print(f"   Batch images shape: {batch['images'].shape}")
    print(f"   Batch bboxes shape: {batch['bboxes'].shape}")
    print(f"   Batch labels shape: {batch['labels'].shape}")
    print(f"   Batch masks shape: {batch['masks'].shape}")
    
    print("\n" + "=" * 50)
    print("All dataset tests passed!")
