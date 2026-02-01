"""
基础模块定义
包含深度可分离卷积(DSC)、Ghost模块、DS-C3k2等
基于论文公式(2): C_DSC = D_K^2 * M * D_F^2 + M * N * D_F^2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def autopad(k, p=None, d=1):
    """自动计算padding以保持特征图尺寸"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """标准卷积模块: Conv + BN + SiLU"""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小
            s: 步长
            p: padding
            g: 分组数
            d: 膨胀率
            act: 是否使用激活函数
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        """融合BN后的前向传播"""
        return self.act(self.conv(x))


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积 (DSC)
    论文公式(2): C_DSC = D_K^2 * M * D_F^2 + M * N * D_F^2
    相比标准卷积，计算量减少约 1/D_K^2 + 1/N 倍
    """
    
    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        """
        Args:
            c1: 输入通道数 (M)
            c2: 输出通道数 (N)
            k: 卷积核大小 (D_K)
            s: 步长
            p: padding
            act: 是否使用激活函数
        """
        super().__init__()
        # Depthwise convolution: 每个通道单独卷积
        self.dwconv = nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        # Pointwise convolution: 1x1卷积进行通道混合
        self.pwconv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    
    def forward(self, x):
        x = self.act(self.bn1(self.dwconv(x)))
        x = self.act(self.bn2(self.pwconv(x)))
        return x


class GhostConv(nn.Module):
    """
    Ghost卷积模块
    通过少量卷积生成部分特征图，然后通过廉价操作生成冗余特征图
    减少计算量约40%
    """
    
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小
            s: 步长
            g: 分组数
            act: 是否使用激活函数
        """
        super().__init__()
        c_ = c2 // 2  # 隐藏通道数
        self.cv1 = Conv(c1, c_, k, s, g=g, act=act)  # 主卷积
        self.cv2 = Conv(c_, c_, 5, 1, g=c_, act=act)  # 廉价操作(深度卷积)
    
    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], dim=1)


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck模块"""
    
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),
            DepthwiseSeparableConv(c_, c_, k, s) if s == 2 else nn.Identity(),
            GhostConv(c_, c2, 1, 1, act=False)
        )
        self.shortcut = nn.Sequential(
            DepthwiseSeparableConv(c1, c1, k, s),
            Conv(c1, c2, 1, 1, act=False)
        ) if s == 2 or c1 != c2 else nn.Identity()
    
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """标准Bottleneck模块"""
    
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DSBottleneck(nn.Module):
    """
    深度可分离Bottleneck模块
    使用深度可分离卷积替换标准卷积以减少计算量
    """
    
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DepthwiseSeparableConv(c_, c2, k[1], 1)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k2(nn.Module):
    """
    C3k2模块 - YOLOv13基础模块
    使用较小的卷积核提高效率
    """
    
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            n: Bottleneck数量
            c3k: 是否使用C3k变体
            e: 扩展比例
            g: 分组数
            shortcut: 是否使用残差连接
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(
            Bottleneck(c_, c_, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        ))
    
    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


class DS_C3k2(nn.Module):
    """
    DS-C3k2模块 - 改进YOLOv13核心模块
    论文3.2节: 使用深度可分离卷积替换大核卷积
    保持感受野同时大幅减少参数量和计算复杂度
    """
    
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            n: DSBottleneck数量
            c3k: 是否使用C3k变体
            e: 扩展比例
            g: 分组数
            shortcut: 是否使用残差连接
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        # 使用深度可分离Bottleneck
        self.m = nn.Sequential(*(
            DSBottleneck(c_, c_, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        ))
    
    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


class C2f(nn.Module):
    """C2f模块 - CSP Bottleneck with 2 convolutions"""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_Ghost(nn.Module):
    """C2f-Ghost模块 - 使用Ghost卷积的C2f变体"""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = GhostConv(c1, 2 * self.c, 1, 1)
        self.cv2 = GhostConv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            GhostBottleneck(self.c, self.c) for _ in range(n)
        )
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """快速空间金字塔池化层"""
    
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class AdaptiveSPPF(nn.Module):
    """
    自适应空间金字塔池化层
    论文3.1节: 嵌入自适应空间金字塔池化层
    """
    
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k
        ])
        # 自适应权重
        self.weights = nn.Parameter(torch.ones(len(k) + 1) / (len(k) + 1))
    
    def forward(self, x):
        x = self.cv1(x)
        features = [x] + [m(x) for m in self.m]
        # 应用自适应权重
        weights = F.softmax(self.weights, dim=0)
        weighted_features = [w * f for w, f in zip(weights, features)]
        return self.cv2(torch.cat(weighted_features, 1))


class Focus(nn.Module):
    """Focus模块 - 将空间信息聚焦到通道空间"""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
    
    def forward(self, x):
        return self.conv(torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], 1))


class Concat(nn.Module):
    """沿指定维度拼接张量"""
    
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    
    def forward(self, x):
        return torch.cat(x, self.d)


class Upsample(nn.Module):
    """上采样模块"""
    
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Downsample(nn.Module):
    """下采样模块"""
    
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.conv = Conv(c1, c2, k, s)
    
    def forward(self, x):
        return self.conv(x)


def initialize_weights(model):
    """初始化模型权重"""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif t is nn.BatchNorm2d:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif t is nn.Linear:
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    # 测试各模块
    print("Testing basic modules...")
    
    # 测试深度可分离卷积
    x = torch.randn(1, 64, 80, 80)
    dsc = DepthwiseSeparableConv(64, 128, k=3)
    out = dsc(x)
    print(f"DSC: {x.shape} -> {out.shape}")
    
    # 测试DS-C3k2
    ds_c3k2 = DS_C3k2(64, 128, n=2)
    out = ds_c3k2(x)
    print(f"DS-C3k2: {x.shape} -> {out.shape}")
    
    # 测试Ghost卷积
    ghost = GhostConv(64, 128)
    out = ghost(x)
    print(f"GhostConv: {x.shape} -> {out.shape}")
    
    # 测试自适应SPPF
    asppf = AdaptiveSPPF(64, 128)
    out = asppf(x)
    print(f"AdaptiveSPPF: {x.shape} -> {out.shape}")
    
    # 计算参数量
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter counts:")
    print(f"DSC: {count_params(dsc):,}")
    print(f"DS-C3k2: {count_params(ds_c3k2):,}")
    print(f"GhostConv: {count_params(ghost):,}")
    print(f"AdaptiveSPPF: {count_params(asppf):,}")
    
    print("\nAll tests passed!")
