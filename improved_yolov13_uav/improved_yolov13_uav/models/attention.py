"""
注意力机制模块
包含空间注意力(Spatial Attention)和通道注意力(Channel Attention)
基于论文公式(6)和公式(7)

公式(6) - 空间注意力:
M_s(F) = σ(f^{7×7}([AvgPool(F); MaxPool(F)]))

公式(7) - 通道注意力:
M_c(F) = σ(W_1δ(W_0(GAP(F))) + W_1δ(W_0(GMP(F))))

其中:
- F ∈ R^{C×H×W} 是输入特征张量
- σ 是Sigmoid激活函数
- δ 是ReLU激活函数
- W_0 ∈ R^{C/r×C}, W_1 ∈ R^{C×C/r} 是全连接层权重
- r 是缩减比例(实验中设为16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import Conv


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    论文公式(6): M_s(F) = σ(f^{7×7}([AvgPool(F); MaxPool(F)]))
    
    通过分析特征图的空间统计信息识别图像中的关键区域，
    通过并行的平均池化和最大池化操作提取互补的空间显著性特征，
    然后通过卷积层生成空间注意力权重图。
    """
    
    def __init__(self, kernel_size=7):
        """
        Args:
            kernel_size: 卷积核大小，论文中使用7×7
        """
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = kernel_size // 2
        
        # 7×7卷积层，输入为2通道（avg和max），输出为1通道
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 F ∈ R^{C×H×W}
        Returns:
            空间注意力加权后的特征
        """
        # 沿通道维度进行平均池化，生成 1×H×W 特征图
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 沿通道维度进行最大池化，生成 1×H×W 特征图
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接得到 2×H×W 特征图
        concat = torch.cat([avg_out, max_out], dim=1)
        # 通过7×7卷积和Sigmoid生成空间注意力权重 [0,1]
        attention = self.sigmoid(self.conv(concat))
        # 应用空间注意力
        return x * attention


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    论文公式(7): M_c(F) = σ(W_1δ(W_0(GAP(F))) + W_1δ(W_0(GMP(F))))
    
    从全局感受野角度评估每个特征通道的重要性，
    通过squeeze和excitation操作实现通道间的动态重标定。
    
    其中:
    - GAP: 全局平均池化
    - GMP: 全局最大池化
    - W_0 ∈ R^{C/r×C}: 降维全连接层
    - W_1 ∈ R^{C×C/r}: 升维全连接层
    - r: 缩减比例(reduction ratio)，实验中设为16
    - δ: ReLU激活函数
    - σ: Sigmoid激活函数
    """
    
    def __init__(self, channels, reduction_ratio=16):
        """
        Args:
            channels: 输入通道数 C
            reduction_ratio: 缩减比例 r，默认16
        """
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # 中间通道数
        mid_channels = max(channels // reduction_ratio, 8)
        
        # 共享的全连接层 (实现为1x1卷积)
        # W_0: C -> C/r
        self.fc1 = nn.Linear(channels, mid_channels, bias=False)
        # W_1: C/r -> C
        self.fc2 = nn.Linear(mid_channels, channels, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 F ∈ R^{B×C×H×W}
        Returns:
            通道注意力加权后的特征
        """
        b, c, h, w = x.size()
        
        # 全局平均池化: B×C×H×W -> B×C×1×1 -> B×C
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # 全局最大池化: B×C×H×W -> B×C×1×1 -> B×C
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        
        # 通过共享MLP: W_1(δ(W_0(·)))
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        
        # 相加并通过Sigmoid
        attention = self.sigmoid(avg_out + max_out)
        
        # 调整形状并应用通道注意力
        attention = attention.view(b, c, 1, 1)
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    结合通道注意力和空间注意力
    顺序: 输入 -> 通道注意力 -> 空间注意力 -> 输出
    """
    
    def __init__(self, channels, reduction_ratio=16, spatial_kernel=7):
        """
        Args:
            channels: 输入通道数
            reduction_ratio: 通道注意力的缩减比例
            spatial_kernel: 空间注意力的卷积核大小
        """
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel)
    
    def forward(self, x):
        # 先应用通道注意力
        x = self.channel_attention(x)
        # 再应用空间注意力
        x = self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    简化版通道注意力，只使用全局平均池化
    """
    
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        mid_channels = max(channels // reduction_ratio, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class EfficientChannelAttention(nn.Module):
    """
    Efficient Channel Attention (ECA)
    使用1D卷积代替全连接层，避免降维带来的信息损失
    """
    
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # 自适应核大小
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class SpatialChannelAttention(nn.Module):
    """
    空间-通道联合注意力模块
    论文3.3节核心创新
    
    通道注意力权重与空间注意力权重通过元素级乘法融合，
    生成增强的特征表示。
    """
    
    def __init__(self, channels, reduction_ratio=16, spatial_kernel=7):
        """
        Args:
            channels: 输入通道数
            reduction_ratio: 通道注意力缩减比例
            spatial_kernel: 空间注意力卷积核大小
        """
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
    
    def forward(self, x):
        """
        特征增强过程:
        1. 计算通道注意力
        2. 计算空间注意力
        3. 融合两种注意力
        4. 残差连接
        """
        identity = x
        
        # 通道注意力分支
        ca = self.channel_attention(x)
        # 空间注意力分支
        sa = self.spatial_attention(x)
        
        # 融合
        out = self.fusion(ca + sa)
        
        # 残差连接
        return out + identity


class MultiScaleAttention(nn.Module):
    """
    多尺度注意力模块
    在多个尺度上计算注意力，增强对不同大小目标的感知能力
    """
    
    def __init__(self, channels, scales=(1, 2, 4)):
        super().__init__()
        self.scales = scales
        self.attentions = nn.ModuleList([
            SpatialChannelAttention(channels) for _ in scales
        ])
        self.fusion = Conv(channels * len(scales), channels, 1)
    
    def forward(self, x):
        _, _, h, w = x.size()
        features = []
        
        for scale, attn in zip(self.scales, self.attentions):
            if scale == 1:
                feat = attn(x)
            else:
                # 下采样
                down = F.adaptive_avg_pool2d(x, (h // scale, w // scale))
                feat = attn(down)
                # 上采样回原尺寸
                feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            features.append(feat)
        
        # 融合多尺度特征
        return self.fusion(torch.cat(features, dim=1))


class CoordAttention(nn.Module):
    """
    Coordinate Attention
    分解通道注意力为水平和垂直两个方向，保留精确的位置信息
    """
    
    def __init__(self, channels, reduction_ratio=32):
        super().__init__()
        mid_channels = max(8, channels // reduction_ratio)
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU()
        
        self.conv_h = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        identity = x
        b, c, h, w = x.size()
        
        # 水平和垂直池化
        x_h = self.pool_h(x)  # B×C×H×1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # B×C×W×1
        
        # 拼接
        y = torch.cat([x_h, x_w], dim=2)  # B×C×(H+W)×1
        y = self.act(self.bn1(self.conv1(y)))
        
        # 分离
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # 生成注意力权重
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        
        # 应用注意力
        return identity * a_h * a_w


if __name__ == "__main__":
    # 测试注意力模块
    print("Testing attention modules...")
    
    x = torch.randn(2, 64, 80, 80)
    
    # 测试空间注意力
    sa = SpatialAttention(kernel_size=7)
    out = sa(x)
    print(f"SpatialAttention: {x.shape} -> {out.shape}")
    
    # 测试通道注意力
    ca = ChannelAttention(64, reduction_ratio=16)
    out = ca(x)
    print(f"ChannelAttention: {x.shape} -> {out.shape}")
    
    # 测试CBAM
    cbam = CBAM(64, reduction_ratio=16, spatial_kernel=7)
    out = cbam(x)
    print(f"CBAM: {x.shape} -> {out.shape}")
    
    # 测试空间-通道联合注意力
    sca = SpatialChannelAttention(64, reduction_ratio=16, spatial_kernel=7)
    out = sca(x)
    print(f"SpatialChannelAttention: {x.shape} -> {out.shape}")
    
    # 测试坐标注意力
    coord = CoordAttention(64)
    out = coord(x)
    print(f"CoordAttention: {x.shape} -> {out.shape}")
    
    # 计算参数量
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter counts:")
    print(f"SpatialAttention: {count_params(sa):,}")
    print(f"ChannelAttention: {count_params(ca):,}")
    print(f"CBAM: {count_params(cbam):,}")
    print(f"SpatialChannelAttention: {count_params(sca):,}")
    print(f"CoordAttention: {count_params(coord):,}")
    
    print("\nAll tests passed!")
