"""
Data Augmentation Module for Improved YOLOv13 UAV Obstacle Avoidance
====================================================================

Implements augmentation strategies as specified in the paper:
- Mosaic augmentation: stitches 4 images into one
- MixUp: linear interpolation with coefficient 0.2
- Random cropping: range 0.5 to 1.0 times original dimensions
- Color jittering: brightness (±0.3), contrast (±0.3), saturation (±0.3), hue (±0.1)

Reference: Section 4.1 - Experimental Setup and Datasets
"""

import cv2
import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Any
import torch


class ColorJitter:
    """
    Color jittering augmentation.
    
    Parameters from paper (Section 4.1):
    - Brightness: ±0.3
    - Contrast: ±0.3
    - Saturation: ±0.3
    - Hue: ±0.1
    """
    
    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.1
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply color jittering to image."""
        img = image.copy().astype(np.float32)
        
        # Brightness adjustment
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = img * factor
        
        # Contrast adjustment
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = img.mean()
            img = (img - mean) * factor + mean
        
        # Convert to HSV for saturation and hue
        img = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Saturation adjustment
        if self.saturation > 0:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            hsv[:, :, 1] = hsv[:, :, 1] * factor
        
        # Hue adjustment
        if self.hue > 0:
            delta = random.uniform(-self.hue, self.hue) * 180
            hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180
        
        hsv = np.clip(hsv, 0, [180, 255, 255]).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return img


class RandomCrop:
    """
    Random cropping augmentation.
    
    Parameters from paper (Section 4.1):
    - Crop range: 0.5 to 1.0 times original dimensions
    """
    
    def __init__(self, min_scale: float = 0.5, max_scale: float = 1.0):
        self.min_scale = min_scale
        self.max_scale = max_scale
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply random cropping to image and adjust bounding boxes.
        
        Args:
            image: Input image (H, W, C)
            bboxes: Bounding boxes in format (x1, y1, x2, y2), normalized [0, 1]
            labels: Class labels
        
        Returns:
            Cropped image, adjusted bboxes, filtered labels
        """
        h, w = image.shape[:2]
        
        # Random scale
        scale = random.uniform(self.min_scale, self.max_scale)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Random position
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        
        # Crop image
        cropped = image[top:top + new_h, left:left + new_w]
        
        # Adjust bounding boxes
        if len(bboxes) > 0:
            # Convert to absolute coordinates
            bboxes_abs = bboxes.copy()
            bboxes_abs[:, [0, 2]] *= w
            bboxes_abs[:, [1, 3]] *= h
            
            # Adjust for crop
            bboxes_abs[:, [0, 2]] -= left
            bboxes_abs[:, [1, 3]] -= top
            
            # Clip to new image boundaries
            bboxes_abs[:, [0, 2]] = np.clip(bboxes_abs[:, [0, 2]], 0, new_w)
            bboxes_abs[:, [1, 3]] = np.clip(bboxes_abs[:, [1, 3]], 0, new_h)
            
            # Filter out boxes that are too small or outside
            valid_w = bboxes_abs[:, 2] - bboxes_abs[:, 0]
            valid_h = bboxes_abs[:, 3] - bboxes_abs[:, 1]
            valid_mask = (valid_w > 1) & (valid_h > 1)
            
            bboxes_abs = bboxes_abs[valid_mask]
            labels = labels[valid_mask]
            
            # Convert back to normalized coordinates
            if len(bboxes_abs) > 0:
                bboxes_abs[:, [0, 2]] /= new_w
                bboxes_abs[:, [1, 3]] /= new_h
            
            bboxes = bboxes_abs
        
        return cropped, bboxes, labels


class MosaicAugmentation:
    """
    Mosaic augmentation - stitches 4 images into one.
    
    Reference: Section 4.1
    "Mosaic augmentation stitches four images into one to increase target
    quantity and scene diversity within batches."
    """
    
    def __init__(self, img_size: int = 640, mosaic_border: List[int] = [-320, -320]):
        self.img_size = img_size
        self.mosaic_border = mosaic_border
    
    def __call__(
        self,
        images: List[np.ndarray],
        bboxes_list: List[np.ndarray],
        labels_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create mosaic from 4 images.
        
        Args:
            images: List of 4 images
            bboxes_list: List of bounding box arrays for each image
            labels_list: List of label arrays for each image
        
        Returns:
            Mosaic image, combined bboxes, combined labels
        """
        assert len(images) == 4, "Mosaic requires exactly 4 images"
        
        s = self.img_size
        yc = int(random.uniform(s * 0.5, s * 1.5))
        xc = int(random.uniform(s * 0.5, s * 1.5))
        
        # Create mosaic image
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        
        combined_bboxes = []
        combined_labels = []
        
        for i, (img, bboxes, labels) in enumerate(zip(images, bboxes_list, labels_list)):
            h, w = img.shape[:2]
            
            # Place image in mosaic
            if i == 0:  # Top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # Top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # Bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # Bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # Adjust bounding boxes
            padw = x1a - x1b
            padh = y1a - y1b
            
            if len(bboxes) > 0:
                bboxes_abs = bboxes.copy()
                bboxes_abs[:, [0, 2]] = bboxes_abs[:, [0, 2]] * w + padw
                bboxes_abs[:, [1, 3]] = bboxes_abs[:, [1, 3]] * h + padh
                
                combined_bboxes.append(bboxes_abs)
                combined_labels.append(labels)
        
        # Combine all boxes and labels
        if combined_bboxes:
            combined_bboxes = np.concatenate(combined_bboxes, axis=0)
            combined_labels = np.concatenate(combined_labels, axis=0)
        else:
            combined_bboxes = np.zeros((0, 4))
            combined_labels = np.zeros((0,))
        
        # Clip boxes to mosaic boundaries
        combined_bboxes[:, [0, 2]] = np.clip(combined_bboxes[:, [0, 2]], 0, 2 * s)
        combined_bboxes[:, [1, 3]] = np.clip(combined_bboxes[:, [1, 3]], 0, 2 * s)
        
        # Crop to target size
        mosaic_img = mosaic_img[s // 2:s // 2 + s, s // 2:s // 2 + s]
        
        # Adjust boxes for cropping
        combined_bboxes[:, [0, 2]] -= s // 2
        combined_bboxes[:, [1, 3]] -= s // 2
        
        # Clip again
        combined_bboxes[:, [0, 2]] = np.clip(combined_bboxes[:, [0, 2]], 0, s)
        combined_bboxes[:, [1, 3]] = np.clip(combined_bboxes[:, [1, 3]], 0, s)
        
        # Filter valid boxes
        valid_w = combined_bboxes[:, 2] - combined_bboxes[:, 0]
        valid_h = combined_bboxes[:, 3] - combined_bboxes[:, 1]
        valid_mask = (valid_w > 2) & (valid_h > 2)
        
        combined_bboxes = combined_bboxes[valid_mask]
        combined_labels = combined_labels[valid_mask]
        
        # Normalize to [0, 1]
        combined_bboxes[:, [0, 2]] /= s
        combined_bboxes[:, [1, 3]] /= s
        
        return mosaic_img, combined_bboxes, combined_labels


class MixUpAugmentation:
    """
    MixUp augmentation - linear interpolation of images and labels.
    
    Reference: Section 4.1
    "MixUp performs linear interpolation of images and labels with a
    mixing coefficient of 0.2."
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter for mixing coefficient
        """
        self.alpha = alpha
    
    def __call__(
        self,
        image1: np.ndarray,
        bboxes1: np.ndarray,
        labels1: np.ndarray,
        image2: np.ndarray,
        bboxes2: np.ndarray,
        labels2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply MixUp augmentation to two images.
        
        Args:
            image1, bboxes1, labels1: First image and annotations
            image2, bboxes2, labels2: Second image and annotations
        
        Returns:
            Mixed image, combined bboxes, combined labels
        """
        # Sample mixing coefficient from beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Ensure lam >= 0.5 (image1 dominates)
        lam = max(lam, 1 - lam)
        
        # Resize images to same size
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        
        if (h1, w1) != (h2, w2):
            image2 = cv2.resize(image2, (w1, h1))
        
        # Mix images
        mixed_img = (lam * image1.astype(np.float32) + 
                    (1 - lam) * image2.astype(np.float32)).astype(np.uint8)
        
        # Combine bounding boxes and labels
        combined_bboxes = np.concatenate([bboxes1, bboxes2], axis=0) if len(bboxes1) > 0 and len(bboxes2) > 0 else (bboxes1 if len(bboxes1) > 0 else bboxes2)
        combined_labels = np.concatenate([labels1, labels2], axis=0) if len(labels1) > 0 and len(labels2) > 0 else (labels1 if len(labels1) > 0 else labels2)
        
        return mixed_img, combined_bboxes, combined_labels


class RandomHorizontalFlip:
    """Random horizontal flip augmentation."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply random horizontal flip."""
        if random.random() < self.p:
            image = cv2.flip(image, 1)  # Horizontal flip
            
            if len(bboxes) > 0:
                # Flip x coordinates
                bboxes[:, [0, 2]] = 1.0 - bboxes[:, [2, 0]]
        
        return image, bboxes, labels


class RandomScale:
    """Random scaling augmentation."""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2)):
        self.scale_range = scale_range
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply random scaling."""
        scale = random.uniform(*self.scale_range)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        image = cv2.resize(image, (new_w, new_h))
        
        return image, bboxes, labels


class Normalize:
    """Normalize image to [0, 1] range."""
    
    def __init__(self, mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225]):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Normalize image."""
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image


class Resize:
    """Resize image to target size."""
    
    def __init__(self, size: Tuple[int, int] = (640, 640)):
        self.size = size
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resize image (bboxes are normalized so they don't need adjustment)."""
        image = cv2.resize(image, self.size)
        return image, bboxes, labels


class LetterBox:
    """Letterbox resize maintaining aspect ratio with padding."""
    
    def __init__(self, size: Tuple[int, int] = (640, 640), color: int = 114):
        self.size = size
        self.color = color
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply letterbox resize."""
        h, w = image.shape[:2]
        target_h, target_w = self.size
        
        # Calculate scale
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        image = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.full((target_h, target_w, 3), self.color, dtype=np.uint8)
        
        # Calculate padding
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        # Place resized image
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = image
        
        # Adjust bounding boxes
        if len(bboxes) > 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * new_w / target_w + pad_w / target_w
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * new_h / target_h + pad_h / target_h
        
        return padded, bboxes, labels


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply all transforms sequentially."""
        for transform in self.transforms:
            if hasattr(transform, '__call__'):
                result = transform(image, bboxes, labels) if hasattr(transform, 'transforms') or hasattr(transform, 'p') or hasattr(transform, 'min_scale') or hasattr(transform, 'size') or hasattr(transform, 'scale_range') else (transform(image), bboxes, labels)
                if isinstance(result, tuple):
                    image, bboxes, labels = result
                else:
                    image = result
        
        return image, bboxes, labels


def get_train_transforms(img_size: int = 640) -> Compose:
    """
    Get training transforms as specified in paper Section 4.1.
    
    Returns:
        Composed transforms for training
    """
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomCrop(min_scale=0.5, max_scale=1.0),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        LetterBox(size=(img_size, img_size)),
    ])


def get_val_transforms(img_size: int = 640) -> Compose:
    """
    Get validation transforms.
    
    Returns:
        Composed transforms for validation
    """
    return Compose([
        LetterBox(size=(img_size, img_size)),
    ])


# Test the augmentation module
if __name__ == "__main__":
    print("Testing Data Augmentation Module")
    print("=" * 50)
    
    # Create dummy image
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bboxes = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.7, 0.8]])
    dummy_labels = np.array([0, 1])
    
    # Test Color Jitter
    print("\n1. Testing ColorJitter (brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)")
    color_jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    jittered = color_jitter(dummy_img)
    print(f"   Input shape: {dummy_img.shape}, Output shape: {jittered.shape}")
    
    # Test Random Crop
    print("\n2. Testing RandomCrop (scale_range: 0.5-1.0)")
    random_crop = RandomCrop(min_scale=0.5, max_scale=1.0)
    cropped_img, cropped_bboxes, cropped_labels = random_crop(dummy_img, dummy_bboxes.copy(), dummy_labels.copy())
    print(f"   Input shape: {dummy_img.shape}, Output shape: {cropped_img.shape}")
    print(f"   Boxes: {len(dummy_bboxes)} -> {len(cropped_bboxes)}")
    
    # Test Mosaic
    print("\n3. Testing MosaicAugmentation")
    mosaic = MosaicAugmentation(img_size=640)
    images = [dummy_img.copy() for _ in range(4)]
    bboxes_list = [dummy_bboxes.copy() for _ in range(4)]
    labels_list = [dummy_labels.copy() for _ in range(4)]
    mosaic_img, mosaic_bboxes, mosaic_labels = mosaic(images, bboxes_list, labels_list)
    print(f"   Output shape: {mosaic_img.shape}")
    print(f"   Combined boxes: {len(mosaic_bboxes)}")
    
    # Test MixUp
    print("\n4. Testing MixUpAugmentation (alpha=0.2)")
    mixup = MixUpAugmentation(alpha=0.2)
    mixed_img, mixed_bboxes, mixed_labels = mixup(
        dummy_img, dummy_bboxes, dummy_labels,
        dummy_img.copy(), dummy_bboxes.copy(), dummy_labels.copy()
    )
    print(f"   Output shape: {mixed_img.shape}")
    print(f"   Combined boxes: {len(mixed_bboxes)}")
    
    # Test LetterBox
    print("\n5. Testing LetterBox (size=640x640)")
    letterbox = LetterBox(size=(640, 640))
    lb_img, lb_bboxes, lb_labels = letterbox(dummy_img, dummy_bboxes.copy(), dummy_labels.copy())
    print(f"   Input shape: {dummy_img.shape}, Output shape: {lb_img.shape}")
    
    # Test composed transforms
    print("\n6. Testing Training Transforms (composed)")
    train_transforms = get_train_transforms(img_size=640)
    trans_img, trans_bboxes, trans_labels = train_transforms(dummy_img, dummy_bboxes.copy(), dummy_labels.copy())
    print(f"   Output shape: {trans_img.shape}")
    
    print("\n" + "=" * 50)
    print("All augmentation tests passed!")
