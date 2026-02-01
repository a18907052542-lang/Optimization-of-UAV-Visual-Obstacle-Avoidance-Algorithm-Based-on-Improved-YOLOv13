"""
改进的YOLOv13完整网络架构
论文: Optimization of UAV Visual Obstacle Avoidance Algorithm Based on Improved YOLOv13 in Complex Scenarios

核心改进:
1. DS-C3k2模块: 使用深度可分离卷积替换大核卷积
2. P2检测层: 增强微小目标感知能力
3. 可变形卷积: 自适应目标几何形变
4. 空间-通道联合注意力: 增强关键特征表达
5. BiFPN + FullPAD: 多尺度特征融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from models.common import (
    Conv, DepthwiseSeparableConv, DS_C3k2, C2f, C2f_Ghost,
    SPPF, AdaptiveSPPF, Focus, Concat, initialize_weights
)
from models.attention import SpatialChannelAttention, CBAM, CoordAttention
from models.dcn import DeformableConvBlock, DeformableBottleneck
from models.neck import EnhancedBiFPN, FourScaleFPN, ImprovedNeck, PANet
from models.head import MultiScaleDetectionHead, DecoupledHead


class ImprovedYOLOv13Backbone(nn.Module):
    """
    改进的YOLOv13骨干网络
    
    特点:
    - 使用DS-C3k2模块替换原始C3k2
    - 嵌入自适应空间金字塔池化层
    - 输出四个尺度的特征: C2(1/4), C3(1/8), C4(1/16), C5(1/32)
    """
    
    def __init__(self, in_channels=3, base_channels=64, depth_multiple=1.0, width_multiple=1.0):
        """
        Args:
            in_channels: 输入通道数(RGB为3)
            base_channels: 基础通道数
            depth_multiple: 深度倍数
            width_multiple: 宽度倍数
        """
        super().__init__()
        
        # 根据倍数调整通道数和深度
        def make_divisible(x, divisor=8):
            return max(divisor, int(x + divisor / 2) // divisor * divisor)
        
        c1 = make_divisible(base_channels * width_multiple)
        c2 = make_divisible(base_channels * 2 * width_multiple)
        c3 = make_divisible(base_channels * 4 * width_multiple)
        c4 = make_divisible(base_channels * 8 * width_multiple)
        
        n = max(round(3 * depth_multiple), 1)  # 重复次数
        
        # Stem: 初始卷积
        self.stem = nn.Sequential(
            Conv(in_channels, c1 // 2, 3, 2),  # 640 -> 320
            Conv(c1 // 2, c1, 3, 2),            # 320 -> 160
        )
        
        # Stage 1: C2 (1/4分辨率)
        self.stage1 = nn.Sequential(
            DS_C3k2(c1, c1, n=n, shortcut=True),
            SpatialChannelAttention(c1, reduction_ratio=16)
        )
        
        # Stage 2: C3 (1/8分辨率)
        self.stage2 = nn.Sequential(
            Conv(c1, c2, 3, 2),  # 160 -> 80
            DS_C3k2(c2, c2, n=n, shortcut=True),
            DS_C3k2(c2, c2, n=n, shortcut=True),
        )
        
        # Stage 3: C4 (1/16分辨率)
        self.stage3 = nn.Sequential(
            Conv(c2, c3, 3, 2),  # 80 -> 40
            DS_C3k2(c3, c3, n=n, shortcut=True),
            DS_C3k2(c3, c3, n=n, shortcut=True),
            DeformableBottleneck(c3, c3),  # 可变形卷积
        )
        
        # Stage 4: C5 (1/32分辨率)
        self.stage4 = nn.Sequential(
            Conv(c3, c4, 3, 2),  # 40 -> 20
            DS_C3k2(c4, c4, n=n, shortcut=True),
            AdaptiveSPPF(c4, c4),  # 自适应空间金字塔池化
        )
        
        # 通道数记录
        self.out_channels = [c1, c2, c3, c4]
        
        # 初始化权重
        initialize_weights(self)
    
    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            features: [C2, C3, C4, C5] 多尺度特征
        """
        # Stem
        x = self.stem(x)  # [B, C1, H/4, W/4]
        
        # Stage 1: C2
        c2 = self.stage1(x)  # [B, C1, H/4, W/4]
        
        # Stage 2: C3
        c3 = self.stage2(c2)  # [B, C2, H/8, W/8]
        
        # Stage 3: C4
        c4 = self.stage3(c3)  # [B, C3, H/16, W/16]
        
        # Stage 4: C5
        c5 = self.stage4(c4)  # [B, C4, H/32, W/32]
        
        return [c2, c3, c4, c5]


class FeatureEnhancementModule(nn.Module):
    """
    复杂场景特征增强模块
    论文3.3节图4
    
    包含四个协同工作的子组件:
    1. 可变形卷积单元
    2. 空间注意力模块
    3. 通道注意力模块
    4. 多尺度特征聚合单元
    """
    
    def __init__(self, channels, reduction_ratio=16):
        """
        Args:
            channels: 输入/输出通道数
            reduction_ratio: 注意力缩减比例
        """
        super().__init__()
        
        # 1. 可变形卷积单元
        self.dcn = DeformableConvBlock(channels, channels, 3, 1, 1)
        
        # 2. 空间-通道联合注意力
        self.attention = SpatialChannelAttention(channels, reduction_ratio)
        
        # 3. 多尺度特征聚合
        self.multi_scale = MultiScaleFeatureAggregation(channels)
        
        # 4. 融合层
        self.fusion = nn.Sequential(
            Conv(channels * 3, channels, 1, 1),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        # 残差连接
        self.residual = nn.Identity()
    
    def forward(self, x):
        """
        特征增强过程
        """
        identity = self.residual(x)
        
        # 可变形卷积分支
        dcn_out = self.dcn(x)
        
        # 注意力分支
        attn_out = self.attention(x)
        
        # 多尺度分支
        ms_out = self.multi_scale(x)
        
        # 融合
        fused = self.fusion(torch.cat([dcn_out, attn_out, ms_out], dim=1))
        
        # 残差连接
        return fused + identity


class MultiScaleFeatureAggregation(nn.Module):
    """
    多尺度特征聚合单元
    论文3.3节: 通过不同感受野尺度的并行分支捕获多粒度上下文信息
    
    包含:
    - 1×1卷积: 保留原始特征
    - 3×3卷积: 捕获局部模式
    - 5×7膨胀卷积(膨胀率2): 扩展感受野
    - 7×7深度可分离卷积: 减少计算成本
    """
    
    def __init__(self, channels):
        super().__init__()
        
        # 分支1: 1×1卷积保留原始特征
        self.branch1 = Conv(channels, channels // 4, 1, 1)
        
        # 分支2: 3×3卷积捕获局部模式
        self.branch2 = Conv(channels, channels // 4, 3, 1)
        
        # 分支3: 膨胀卷积扩展感受野
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, (5, 7), padding=(4, 6), dilation=2, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.SiLU()
        )
        
        # 分支4: 7×7深度可分离卷积
        self.branch4 = DepthwiseSeparableConv(channels, channels // 4, 7, 1, 3)
        
        # 融合1×1卷积
        self.fusion = Conv(channels, channels, 1, 1)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # 拼接并融合
        return self.fusion(torch.cat([b1, b2, b3, b4], dim=1))


class ImprovedYOLOv13(nn.Module):
    """
    改进的YOLOv13完整网络
    
    架构:
    - Backbone: DS-C3k2结构 + 自适应SPPF
    - Neck: 增强型BiFPN + FullPAD
    - Head: 四尺度解耦检测头
    
    核心创新:
    1. P2高分辨率检测层
    2. 可变形卷积增强
    3. 空间-通道联合注意力
    4. INT8量化友好设计
    """
    
    def __init__(self, 
                 num_classes: int = 10,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 depth_multiple: float = 1.0,
                 width_multiple: float = 1.0,
                 use_attention: bool = True,
                 use_dcn: bool = True):
        """
        Args:
            num_classes: 类别数量(VisDrone2019为10)
            in_channels: 输入通道数
            base_channels: 基础通道数
            depth_multiple: 深度倍数
            width_multiple: 宽度倍数
            use_attention: 是否使用注意力机制
            use_dcn: 是否使用可变形卷积
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_dcn = use_dcn
        
        # 1. Backbone
        self.backbone = ImprovedYOLOv13Backbone(
            in_channels=in_channels,
            base_channels=base_channels,
            depth_multiple=depth_multiple,
            width_multiple=width_multiple
        )
        
        # 获取backbone输出通道
        backbone_channels = self.backbone.out_channels  # [C1, C2, C3, C4]
        
        # 2. 特征增强模块（可选）
        if use_attention or use_dcn:
            self.feature_enhance = nn.ModuleList([
                FeatureEnhancementModule(c) for c in backbone_channels
            ])
        else:
            self.feature_enhance = None
        
        # 3. Neck: 四尺度BiFPN
        self.neck = ImprovedNeck(
            in_channels=backbone_channels,
            out_channels=base_channels * 4,  # 统一通道数
            use_bifpn=True,
            use_attention=use_attention
        )
        
        # 4. Detection Head: 多尺度解耦检测头
        self.head = MultiScaleDetectionHead(
            num_classes=num_classes,
            in_channels=backbone_channels,
            strides=(4, 8, 16, 32),
            reg_max=16,
            use_lightweight_p2=True
        )
        
        # 初始化
        initialize_weights(self)
    
    def forward(self, x: torch.Tensor) -> Dict:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
        
        Returns:
            outputs: 包含多尺度预测的字典
        """
        # Backbone特征提取
        features = self.backbone(x)  # [C2, C3, C4, C5]
        
        # 特征增强（可选）
        if self.feature_enhance is not None:
            features = [enhance(feat) for enhance, feat in zip(self.feature_enhance, features)]
        
        # Neck多尺度融合
        features = self.neck(features)  # [P2, P3, P4, P5]
        
        # Detection Head
        outputs = self.head(features)
        
        return outputs
    
    def predict(self, x: torch.Tensor, conf_threshold: float = 0.25,
                nms_threshold: float = 0.45) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测并进行后处理
        
        Args:
            x: 输入图像
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
        
        Returns:
            boxes: 边界框
            scores: 置信度分数
            labels: 类别标签
        """
        # 前向传播
        outputs = self.forward(x)
        
        # 解码
        img_size = x.shape[2:]
        boxes, cls_scores, objectness = self.head.decode(outputs, img_size)
        
        # 合并置信度
        scores = cls_scores * objectness.unsqueeze(-1)
        
        # NMS
        results = self._nms(boxes, scores, conf_threshold, nms_threshold)
        
        return results
    
    def _nms(self, boxes, scores, conf_threshold, nms_threshold):
        """非极大值抑制"""
        batch_size = boxes.shape[0]
        results = []
        
        for i in range(batch_size):
            box = boxes[i]  # [N, 4]
            score = scores[i]  # [N, C]
            
            # 获取最大类别分数
            max_scores, labels = score.max(dim=1)
            
            # 置信度过滤
            mask = max_scores > conf_threshold
            box = box[mask]
            max_scores = max_scores[mask]
            labels = labels[mask]
            
            if len(box) == 0:
                results.append((torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.long)))
                continue
            
            # 类别级NMS
            keep_boxes = []
            keep_scores = []
            keep_labels = []
            
            for cls in range(self.num_classes):
                cls_mask = labels == cls
                if not cls_mask.any():
                    continue
                
                cls_boxes = box[cls_mask]
                cls_scores = max_scores[cls_mask]
                
                # NMS
                keep = self._box_nms(cls_boxes, cls_scores, nms_threshold)
                
                keep_boxes.append(cls_boxes[keep])
                keep_scores.append(cls_scores[keep])
                keep_labels.append(torch.full((len(keep),), cls, dtype=torch.long, device=box.device))
            
            if keep_boxes:
                results.append((
                    torch.cat(keep_boxes),
                    torch.cat(keep_scores),
                    torch.cat(keep_labels)
                ))
            else:
                results.append((torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.long)))
        
        return results
    
    @staticmethod
    def _box_nms(boxes, scores, threshold):
        """单类别NMS"""
        if len(boxes) == 0:
            return torch.empty(0, dtype=torch.long)
        
        # 按分数排序
        _, order = scores.sort(descending=True)
        
        keep = []
        while len(order) > 0:
            idx = order[0].item()
            keep.append(idx)
            
            if len(order) == 1:
                break
            
            # 计算IoU
            remaining = order[1:]
            ious = ImprovedYOLOv13._box_iou(boxes[idx:idx+1], boxes[remaining])
            
            # 保留IoU小于阈值的框
            mask = ious.squeeze(0) < threshold
            order = remaining[mask]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    
    @staticmethod
    def _box_iou(boxes1, boxes2):
        """计算IoU"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        iou = inter / (area1[:, None] + area2 - inter + 1e-6)
        return iou


# 不同规模的模型配置
def improved_yolov13_n(num_classes=10, **kwargs):
    """Nano版本 - 最轻量"""
    return ImprovedYOLOv13(
        num_classes=num_classes,
        base_channels=32,
        depth_multiple=0.33,
        width_multiple=0.25,
        **kwargs
    )


def improved_yolov13_s(num_classes=10, **kwargs):
    """Small版本 - 论文主要实验配置"""
    return ImprovedYOLOv13(
        num_classes=num_classes,
        base_channels=64,
        depth_multiple=0.33,
        width_multiple=0.5,
        **kwargs
    )


def improved_yolov13_m(num_classes=10, **kwargs):
    """Medium版本"""
    return ImprovedYOLOv13(
        num_classes=num_classes,
        base_channels=64,
        depth_multiple=0.67,
        width_multiple=0.75,
        **kwargs
    )


def improved_yolov13_l(num_classes=10, **kwargs):
    """Large版本"""
    return ImprovedYOLOv13(
        num_classes=num_classes,
        base_channels=64,
        depth_multiple=1.0,
        width_multiple=1.0,
        **kwargs
    )


def improved_yolov13_x(num_classes=10, **kwargs):
    """XLarge版本 - 最高精度"""
    return ImprovedYOLOv13(
        num_classes=num_classes,
        base_channels=64,
        depth_multiple=1.33,
        width_multiple=1.25,
        **kwargs
    )


if __name__ == "__main__":
    print("Testing Improved YOLOv13 Network...")
    
    # 测试不同规模的模型
    models = {
        'nano': improved_yolov13_n,
        'small': improved_yolov13_s,
        'medium': improved_yolov13_m,
        'large': improved_yolov13_l,
        'xlarge': improved_yolov13_x
    }
    
    x = torch.randn(1, 3, 640, 640)
    
    for name, model_fn in models.items():
        model = model_fn(num_classes=10)
        model.eval()
        
        # 计算参数量
        params = sum(p.numel() for p in model.parameters()) / 1e6
        
        # 前向传播
        with torch.no_grad():
            outputs = model(x)
        
        print(f"\n{name.upper()} Model:")
        print(f"  Parameters: {params:.2f}M")
        print(f"  Output scales: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"    Scale {i}: cls={out['cls'].shape}, reg={out['reg'].shape}")
    
    # 详细测试Small模型
    print("\n" + "="*50)
    print("Detailed test of Small model:")
    print("="*50)
    
    model = improved_yolov13_s(num_classes=10)
    model.eval()
    
    # 测试预测
    with torch.no_grad():
        results = model.predict(x, conf_threshold=0.25, nms_threshold=0.45)
    
    print(f"\nPrediction results:")
    for i, (boxes, scores, labels) in enumerate(results):
        print(f"  Batch {i}: {len(boxes)} detections")
    
    # 计算FLOPs（简化估计）
    print(f"\nModel Summary:")
    print(f"  Input size: {x.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 验证各模块
    print("\n" + "="*50)
    print("Module verification:")
    print("="*50)
    
    # Backbone
    backbone_out = model.backbone(x)
    print(f"\nBackbone outputs:")
    for i, feat in enumerate(backbone_out):
        print(f"  C{i+2}: {feat.shape}")
    
    # 特征增强
    if model.feature_enhance is not None:
        enhanced = [enhance(feat) for enhance, feat in zip(model.feature_enhance, backbone_out)]
        print(f"\nFeature enhancement outputs:")
        for i, feat in enumerate(enhanced):
            print(f"  Enhanced C{i+2}: {feat.shape}")
    
    # Neck
    neck_out = model.neck(backbone_out)
    print(f"\nNeck outputs:")
    for i, feat in enumerate(neck_out):
        print(f"  P{i+2}: {feat.shape}")
    
    print("\nAll tests passed!")
