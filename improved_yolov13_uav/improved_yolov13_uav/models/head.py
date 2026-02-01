"""
检测头模块
基于论文3.2节的任务解耦设计

检测头采用任务解耦设计，将分类和定位任务分离到独立分支进行处理，
每个分支使用专门优化的卷积层序列。

分类分支采用Focal Loss处理正负样本不平衡:
公式(3): L_cls = -α_t(1-p_t)^γ log(p_t)

定位分支采用Distribution Focal Loss进行边界框回归:
公式(4): L_loc = -((y_{l+1}-y)log(ŷ_l) + (y-y_l)log(ŷ_{l+1}))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.common import Conv, DepthwiseSeparableConv


class DecoupledHead(nn.Module):
    """
    解耦检测头
    将分类和定位任务分离到独立分支
    """
    
    def __init__(self, num_classes, in_channels, num_anchors=1, reg_max=16):
        """
        Args:
            num_classes: 类别数
            in_channels: 输入通道数
            num_anchors: 每个位置的anchor数量（anchor-free为1）
            reg_max: 分布式focal loss的最大值
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.reg_max = reg_max
        
        # 分类分支
        self.cls_convs = nn.Sequential(
            Conv(in_channels, in_channels, 3, 1),
            Conv(in_channels, in_channels, 3, 1)
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes * num_anchors, 1)
        
        # 定位分支
        self.reg_convs = nn.Sequential(
            Conv(in_channels, in_channels, 3, 1),
            Conv(in_channels, in_channels, 3, 1)
        )
        # 输出4个分布（left, top, right, bottom），每个分布有reg_max+1个值
        self.reg_pred = nn.Conv2d(in_channels, 4 * (reg_max + 1) * num_anchors, 1)
        
        # 目标置信度分支
        self.obj_pred = nn.Conv2d(in_channels, num_anchors, 1)
        
        # 初始化
        self._initialize_biases()
    
    def _initialize_biases(self):
        """初始化偏置以稳定训练初期"""
        # 分类偏置初始化（使初始预测接近背景）
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)
        
        # 目标置信度偏置
        nn.init.constant_(self.obj_pred.bias, bias_value)
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, C, H, W]
        Returns:
            cls_output: 分类输出 [B, num_classes, H, W]
            reg_output: 回归输出 [B, 4*(reg_max+1), H, W]
            obj_output: 目标置信度 [B, 1, H, W]
        """
        # 分类分支
        cls_feat = self.cls_convs(x)
        cls_output = self.cls_pred(cls_feat)
        
        # 定位分支
        reg_feat = self.reg_convs(x)
        reg_output = self.reg_pred(reg_feat)
        
        # 目标置信度
        obj_output = self.obj_pred(reg_feat)
        
        return cls_output, reg_output, obj_output


class LightweightDecoupledHead(nn.Module):
    """
    轻量级解耦检测头
    使用深度可分离卷积减少计算量
    适用于P2层和边缘设备部署
    """
    
    def __init__(self, num_classes, in_channels, num_anchors=1, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.reg_max = reg_max
        
        # 共享特征提取（使用深度可分离卷积）
        self.stem = nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels, 3, 1),
            DepthwiseSeparableConv(in_channels, in_channels, 3, 1)
        )
        
        # 分类头
        self.cls_pred = nn.Conv2d(in_channels, num_classes * num_anchors, 1)
        
        # 回归头
        self.reg_pred = nn.Conv2d(in_channels, 4 * (reg_max + 1) * num_anchors, 1)
        
        # 目标置信度头
        self.obj_pred = nn.Conv2d(in_channels, num_anchors, 1)
        
        self._initialize_biases()
    
    def _initialize_biases(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)
        nn.init.constant_(self.obj_pred.bias, bias_value)
    
    def forward(self, x):
        feat = self.stem(x)
        cls_output = self.cls_pred(feat)
        reg_output = self.reg_pred(feat)
        obj_output = self.obj_pred(feat)
        return cls_output, reg_output, obj_output


class DynamicAnchorHead(nn.Module):
    """
    动态锚框检测头
    论文3.1节: 动态锚框调整机制
    """
    
    def __init__(self, num_classes, in_channels, num_anchors=3, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.reg_max = reg_max
        
        # 特征提取
        self.convs = nn.Sequential(
            Conv(in_channels, in_channels, 3, 1),
            Conv(in_channels, in_channels, 3, 1)
        )
        
        # 预测头
        self.cls_pred = nn.Conv2d(in_channels, num_classes * num_anchors, 1)
        self.reg_pred = nn.Conv2d(in_channels, 4 * num_anchors, 1)
        self.obj_pred = nn.Conv2d(in_channels, num_anchors, 1)
        
        # 动态锚框调整
        self.anchor_adjust = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 2 * num_anchors, 1),  # 宽高调整
            nn.Sigmoid()
        )
        
        self._initialize_biases()
    
    def _initialize_biases(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)
        nn.init.constant_(self.obj_pred.bias, bias_value)
    
    def forward(self, x):
        feat = self.convs(x)
        
        cls_output = self.cls_pred(feat)
        reg_output = self.reg_pred(feat)
        obj_output = self.obj_pred(feat)
        
        # 动态锚框调整因子
        anchor_adjust = self.anchor_adjust(feat)  # [B, 2*num_anchors, H, W]
        
        return cls_output, reg_output, obj_output, anchor_adjust


class MultiScaleDetectionHead(nn.Module):
    """
    多尺度检测头
    为P2, P3, P4, P5四个尺度分别设置检测头
    """
    
    def __init__(self, num_classes, in_channels=(64, 128, 256, 512), 
                 strides=(4, 8, 16, 32), reg_max=16, use_lightweight_p2=True):
        """
        Args:
            num_classes: 类别数
            in_channels: 各尺度输入通道数
            strides: 各尺度的步长
            reg_max: DFL最大值
            use_lightweight_p2: P2层是否使用轻量级头
        """
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.reg_max = reg_max
        self.num_scales = len(in_channels)
        
        # 为各尺度创建检测头
        self.heads = nn.ModuleList()
        for i, channels in enumerate(in_channels):
            if i == 0 and use_lightweight_p2:
                # P2使用轻量级头
                head = LightweightDecoupledHead(num_classes, channels, reg_max=reg_max)
            else:
                head = DecoupledHead(num_classes, channels, reg_max=reg_max)
            self.heads.append(head)
        
        # DFL层（用于从分布转换为实际值）
        self.dfl = DFLModule(reg_max)
    
    def forward(self, features):
        """
        Args:
            features: 多尺度特征列表 [P2, P3, P4, P5]
        Returns:
            outputs: 各尺度的预测结果
        """
        outputs = []
        for feat, head in zip(features, self.heads):
            cls_out, reg_out, obj_out = head(feat)
            outputs.append({
                'cls': cls_out,
                'reg': reg_out,
                'obj': obj_out
            })
        return outputs
    
    def decode(self, outputs, img_size):
        """
        解码预测结果为边界框
        Args:
            outputs: 多尺度预测结果
            img_size: 输入图像尺寸 (H, W)
        Returns:
            boxes: [B, N, 4] 边界框坐标
            scores: [B, N, num_classes] 类别分数
            objectness: [B, N] 目标置信度
        """
        all_boxes = []
        all_scores = []
        all_objectness = []
        
        for i, (output, stride) in enumerate(zip(outputs, self.strides)):
            cls_out = output['cls']  # [B, C, H, W]
            reg_out = output['reg']  # [B, 4*(reg_max+1), H, W]
            obj_out = output['obj']  # [B, 1, H, W]
            
            b, _, h, w = cls_out.shape
            
            # 生成网格
            yv, xv = torch.meshgrid([
                torch.arange(h, device=cls_out.device),
                torch.arange(w, device=cls_out.device)
            ], indexing='ij')
            grid = torch.stack([xv, yv], dim=-1).float()  # [H, W, 2]
            grid = grid.view(1, h, w, 2).repeat(b, 1, 1, 1)  # [B, H, W, 2]
            
            # 解码回归输出
            reg_out = reg_out.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 4*(reg_max+1)]
            reg_out = reg_out.view(b, h, w, 4, self.reg_max + 1)  # [B, H, W, 4, reg_max+1]
            
            # DFL解码
            reg_out = self.dfl(reg_out)  # [B, H, W, 4]
            
            # 计算边界框坐标
            # lt: left, top; rb: right, bottom
            lt = grid - reg_out[..., :2]
            rb = grid + reg_out[..., 2:]
            boxes = torch.cat([lt, rb], dim=-1) * stride  # [B, H, W, 4]
            
            # 分类分数
            cls_out = cls_out.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
            scores = cls_out.sigmoid()
            
            # 目标置信度
            obj_out = obj_out.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 1]
            objectness = obj_out.sigmoid().squeeze(-1)
            
            # 展平
            boxes = boxes.view(b, -1, 4)
            scores = scores.view(b, -1, self.num_classes)
            objectness = objectness.view(b, -1)
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_objectness.append(objectness)
        
        # 合并所有尺度
        all_boxes = torch.cat(all_boxes, dim=1)
        all_scores = torch.cat(all_scores, dim=1)
        all_objectness = torch.cat(all_objectness, dim=1)
        
        return all_boxes, all_scores, all_objectness


class DFLModule(nn.Module):
    """
    Distribution Focal Loss解码模块
    论文公式(4): L_loc = -((y_{l+1}-y)log(ŷ_l) + (y-y_l)log(ŷ_{l+1}))
    
    将离散分布转换为连续值
    """
    
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        # 创建投影权重 [0, 1, 2, ..., reg_max]
        self.register_buffer('proj', torch.arange(reg_max + 1, dtype=torch.float32))
    
    def forward(self, x):
        """
        Args:
            x: [B, H, W, 4, reg_max+1] 分布预测
        Returns:
            [B, H, W, 4] 解码后的值
        """
        # Softmax得到概率分布
        x = F.softmax(x, dim=-1)
        # 加权求和得到期望值
        x = (x * self.proj.view(1, 1, 1, 1, -1)).sum(dim=-1)
        return x


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss
    结合分类和定位质量的损失函数
    """
    
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, pred, target, quality):
        """
        Args:
            pred: 预测分数 [N, C]
            target: 目标类别 [N]
            quality: IoU质量分数 [N]
        """
        pred_sigmoid = pred.sigmoid()
        
        # 计算focal权重
        pt = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_weight = (1 - pt) ** self.gamma
        
        # 用quality替换二值标签
        scale_factor = quality.unsqueeze(-1) - pred_sigmoid
        
        # BCE损失
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # 应用focal权重
        loss = focal_weight * bce * scale_factor.abs().pow(self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        
        return loss.sum()


if __name__ == "__main__":
    print("Testing Detection Head modules...")
    
    # 模拟多尺度特征输入
    p2 = torch.randn(2, 64, 160, 160)
    p3 = torch.randn(2, 128, 80, 80)
    p4 = torch.randn(2, 256, 40, 40)
    p5 = torch.randn(2, 512, 20, 20)
    features = [p2, p3, p4, p5]
    
    num_classes = 10
    
    # 测试解耦检测头
    print("\nTesting DecoupledHead...")
    head = DecoupledHead(num_classes, 64)
    cls_out, reg_out, obj_out = head(p2)
    print(f"Input: {p2.shape}")
    print(f"Cls output: {cls_out.shape}")
    print(f"Reg output: {reg_out.shape}")
    print(f"Obj output: {obj_out.shape}")
    
    # 测试轻量级检测头
    print("\nTesting LightweightDecoupledHead...")
    light_head = LightweightDecoupledHead(num_classes, 64)
    cls_out, reg_out, obj_out = light_head(p2)
    print(f"Input: {p2.shape}")
    print(f"Cls output: {cls_out.shape}")
    print(f"Reg output: {reg_out.shape}")
    print(f"Obj output: {obj_out.shape}")
    
    # 测试多尺度检测头
    print("\nTesting MultiScaleDetectionHead...")
    ms_head = MultiScaleDetectionHead(
        num_classes=num_classes,
        in_channels=(64, 128, 256, 512),
        strides=(4, 8, 16, 32)
    )
    outputs = ms_head(features)
    print("Multi-scale outputs:")
    for i, out in enumerate(outputs):
        print(f"  Scale {i}: cls={out['cls'].shape}, reg={out['reg'].shape}, obj={out['obj'].shape}")
    
    # 测试解码
    print("\nTesting decode...")
    boxes, scores, objectness = ms_head.decode(outputs, img_size=(640, 640))
    print(f"Decoded boxes: {boxes.shape}")
    print(f"Decoded scores: {scores.shape}")
    print(f"Decoded objectness: {objectness.shape}")
    
    # 测试DFL模块
    print("\nTesting DFLModule...")
    dfl = DFLModule(reg_max=16)
    x = torch.randn(2, 80, 80, 4, 17)
    out = dfl(x)
    print(f"DFL input: {x.shape} -> output: {out.shape}")
    
    # 计算参数量
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter counts:")
    print(f"DecoupledHead: {count_params(head):,}")
    print(f"LightweightDecoupledHead: {count_params(light_head):,}")
    print(f"MultiScaleDetectionHead: {count_params(ms_head):,}")
    
    print("\nAll tests passed!")
