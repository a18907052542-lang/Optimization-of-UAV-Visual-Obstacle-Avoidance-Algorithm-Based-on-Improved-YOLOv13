"""
Neck模块 - 四尺度特征金字塔网络
基于论文3.2节和公式(1):

公式(1) - 加权双向特征金字塔融合:
P_out^td = Conv(Σ(w_i * P_in^i) / (Σw_j + ε))

其中:
- P_in^i: 第i层输入特征
- w_i: 可学习权重参数
- ε: 防止除零的小常数
- P_out^td: 自顶向下路径的输出特征

论文3.2节关键改进:
1. 将原始YOLOv13的三尺度检测头扩展为四尺度结构
2. 新增P2层专门捕获小于32×32像素的目标
3. P2层特征图分辨率为输入图像的1/4，保留更多细节信息
4. 使用深度可分离卷积(DSC)替换P2层的标准卷积以减少计算负担
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import Conv, DepthwiseSeparableConv, DS_C3k2, C2f, Concat, Upsample
from models.attention import SpatialChannelAttention, CBAM


class BiFPNBlock(nn.Module):
    """
    双向特征金字塔网络块
    实现公式(1)的加权特征融合
    """
    
    def __init__(self, channels, num_inputs=2, epsilon=1e-4):
        """
        Args:
            channels: 通道数
            num_inputs: 融合的输入特征数量
            epsilon: 防止除零的小常数
        """
        super().__init__()
        self.epsilon = epsilon
        
        # 可学习权重参数 w_i
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        
        # 融合后的卷积
        self.conv = Conv(channels, channels, 3, 1)
    
    def forward(self, inputs):
        """
        加权特征融合
        P_out^td = Conv(Σ(w_i * P_in^i) / (Σw_j + ε))
        """
        # 确保权重为正
        weights = F.relu(self.weights)
        
        # 归一化权重
        weights = weights / (weights.sum() + self.epsilon)
        
        # 加权求和
        fused = sum(w * inp for w, inp in zip(weights, inputs))
        
        # 卷积
        return self.conv(fused)


class FourScaleFPN(nn.Module):
    """
    四尺度特征金字塔网络
    论文图3: P2, P3, P4, P5四个检测尺度
    
    P2: 1/4分辨率，用于检测小于32×32像素的目标
    P3: 1/8分辨率
    P4: 1/16分辨率
    P5: 1/32分辨率
    """
    
    def __init__(self, in_channels=(64, 128, 256, 512), out_channels=256):
        """
        Args:
            in_channels: 来自backbone的各层通道数 [C2, C3, C4, C5]
            out_channels: 输出统一通道数
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 横向连接：将backbone特征转换为统一通道数
        self.lateral_convs = nn.ModuleList([
            Conv(c, out_channels, 1, 1) for c in in_channels
        ])
        
        # 自顶向下路径的融合卷积
        self.td_convs = nn.ModuleList([
            Conv(out_channels, out_channels, 3, 1) for _ in range(len(in_channels) - 1)
        ])
        
        # 自底向上路径的融合卷积
        self.bu_convs = nn.ModuleList([
            Conv(out_channels, out_channels, 3, 1) for _ in range(len(in_channels) - 1)
        ])
        
        # 下采样卷积（用于自底向上路径）
        self.down_convs = nn.ModuleList([
            Conv(out_channels, out_channels, 3, 2) for _ in range(len(in_channels) - 1)
        ])
        
        # P2层使用深度可分离卷积减少计算量
        self.p2_dsc = DepthwiseSeparableConv(out_channels, out_channels, 3, 1)
    
    def forward(self, inputs):
        """
        Args:
            inputs: 来自backbone的特征 [C2, C3, C4, C5]
        Returns:
            outputs: 融合后的特征 [P2, P3, P4, P5]
        """
        assert len(inputs) == len(self.in_channels)
        
        # 横向连接
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        
        # 自顶向下路径 (Top-Down)
        # 从P5到P2
        for i in range(len(laterals) - 1, 0, -1):
            # 上采样高层特征
            upsampled = F.interpolate(laterals[i], size=laterals[i-1].shape[2:], 
                                      mode='nearest')
            # 与低层特征融合
            laterals[i-1] = self.td_convs[i-1](laterals[i-1] + upsampled)
        
        # P2层使用DSC
        laterals[0] = self.p2_dsc(laterals[0])
        
        # 自底向上路径 (Bottom-Up)
        # 从P2到P5
        outputs = [laterals[0]]
        for i in range(len(laterals) - 1):
            # 下采样低层特征
            downsampled = self.down_convs[i](outputs[-1])
            # 与高层特征融合
            fused = self.bu_convs[i](laterals[i+1] + downsampled)
            outputs.append(fused)
        
        return outputs


class EnhancedBiFPN(nn.Module):
    """
    增强型双向特征金字塔网络
    结合FullPAD范式实现细粒度信息流
    """
    
    def __init__(self, in_channels=(64, 128, 256, 512), out_channels=256, num_repeats=2):
        """
        Args:
            in_channels: backbone各层通道数
            out_channels: 统一输出通道数
            num_repeats: BiFPN重复次数
        """
        super().__init__()
        
        # 第一层横向连接
        self.lateral_convs = nn.ModuleList([
            Conv(c, out_channels, 1, 1) for c in in_channels
        ])
        
        # BiFPN块
        self.bifpn_blocks = nn.ModuleList([
            BiFPNLayer(out_channels, len(in_channels)) for _ in range(num_repeats)
        ])
        
        # P2层深度可分离卷积
        self.p2_refine = DepthwiseSeparableConv(out_channels, out_channels, 3, 1)
        
        # 注意力增强（用于各层输出）
        self.attention = nn.ModuleList([
            SpatialChannelAttention(out_channels) for _ in in_channels
        ])
    
    def forward(self, inputs):
        """
        Args:
            inputs: backbone特征列表 [C2, C3, C4, C5]
        Returns:
            outputs: 融合后的特征列表 [P2, P3, P4, P5]
        """
        # 横向连接
        features = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        
        # BiFPN融合
        for bifpn in self.bifpn_blocks:
            features = bifpn(features)
        
        # P2层精细化
        features[0] = self.p2_refine(features[0])
        
        # 注意力增强
        outputs = [attn(feat) for attn, feat in zip(self.attention, features)]
        
        return outputs


class BiFPNLayer(nn.Module):
    """单层BiFPN"""
    
    def __init__(self, channels, num_levels, epsilon=1e-4):
        super().__init__()
        self.num_levels = num_levels
        self.epsilon = epsilon
        
        # 自顶向下路径
        self.td_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32)) 
            for _ in range(num_levels - 1)
        ])
        self.td_convs = nn.ModuleList([
            Conv(channels, channels, 3, 1) for _ in range(num_levels - 1)
        ])
        
        # 自底向上路径
        self.bu_weights = nn.ParameterList([
            nn.Parameter(torch.ones(3, dtype=torch.float32)) 
            for _ in range(num_levels - 1)
        ])
        self.bu_convs = nn.ModuleList([
            Conv(channels, channels, 3, 1) for _ in range(num_levels - 1)
        ])
        
        # 下采样
        self.down_samples = nn.ModuleList([
            Conv(channels, channels, 3, 2) for _ in range(num_levels - 1)
        ])
    
    def forward(self, features):
        """
        BiFPN前向传播
        """
        # 自顶向下
        td_features = [features[-1]]  # 从最高层开始
        for i in range(self.num_levels - 2, -1, -1):
            w = F.relu(self.td_weights[i])
            w = w / (w.sum() + self.epsilon)
            
            up = F.interpolate(td_features[-1], size=features[i].shape[2:], mode='nearest')
            fused = w[0] * features[i] + w[1] * up
            td_features.append(self.td_convs[i](fused))
        
        td_features = td_features[::-1]  # 反转为 [P2, P3, P4, P5] 顺序
        
        # 自底向上
        bu_features = [td_features[0]]
        for i in range(self.num_levels - 1):
            w = F.relu(self.bu_weights[i])
            w = w / (w.sum() + self.epsilon)
            
            down = self.down_samples[i](bu_features[-1])
            # 三路融合：原始特征 + TD特征 + 下采样特征
            fused = w[0] * features[i+1] + w[1] * td_features[i+1] + w[2] * down
            bu_features.append(self.bu_convs[i](fused))
        
        return bu_features


class PANet(nn.Module):
    """
    Path Aggregation Network
    论文引用[25]: 通过自底向上路径增强和自适应特征池化改进性能
    """
    
    def __init__(self, in_channels=(64, 128, 256, 512), out_channels=(64, 128, 256, 512)):
        super().__init__()
        
        # 自顶向下路径
        self.up_samples = nn.ModuleList()
        self.td_convs = nn.ModuleList()
        
        for i in range(len(in_channels) - 1, 0, -1):
            self.up_samples.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.td_convs.append(DS_C3k2(in_channels[i] + in_channels[i-1], out_channels[i-1], n=1))
        
        # 自底向上路径
        self.down_samples = nn.ModuleList()
        self.bu_convs = nn.ModuleList()
        
        for i in range(len(out_channels) - 1):
            self.down_samples.append(Conv(out_channels[i], out_channels[i], 3, 2))
            self.bu_convs.append(DS_C3k2(out_channels[i] + out_channels[i+1], out_channels[i+1], n=1))
    
    def forward(self, inputs):
        """
        Args:
            inputs: [C2, C3, C4, C5] 或 [P3, P4, P5]
        Returns:
            outputs: [P2, P3, P4, P5] 或 [P3, P4, P5]
        """
        # 自顶向下
        features = list(inputs)
        for i, (up, conv) in enumerate(zip(self.up_samples, self.td_convs)):
            idx = len(features) - 1 - i
            up_feat = up(features[idx])
            # 确保尺寸匹配
            if up_feat.shape[2:] != features[idx-1].shape[2:]:
                up_feat = F.interpolate(up_feat, size=features[idx-1].shape[2:], mode='nearest')
            features[idx-1] = conv(torch.cat([features[idx-1], up_feat], dim=1))
        
        # 自底向上
        for i, (down, conv) in enumerate(zip(self.down_samples, self.bu_convs)):
            down_feat = down(features[i])
            features[i+1] = conv(torch.cat([features[i+1], down_feat], dim=1))
        
        return features


class FeatureAlignmentModule(nn.Module):
    """
    跨尺度特征对齐模块
    论文3.2节: 设计跨尺度特征对齐策略改善小目标检测性能
    """
    
    def __init__(self, channels, num_scales=4):
        super().__init__()
        
        # 各尺度间的对齐卷积
        self.align_convs = nn.ModuleList()
        for i in range(num_scales - 1):
            self.align_convs.append(
                nn.Sequential(
                    Conv(channels, channels, 3, 1),
                    Conv(channels, channels, 1, 1)
                )
            )
        
        # 自适应权重
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
    
    def forward(self, features):
        """
        对齐并融合多尺度特征
        """
        aligned = [features[-1]]  # 从最高层开始
        
        for i in range(len(features) - 2, -1, -1):
            # 上采样高层特征
            up = F.interpolate(aligned[-1], size=features[i].shape[2:], mode='bilinear', align_corners=False)
            # 对齐
            aligned_feat = self.align_convs[i](features[i] + up)
            aligned.append(aligned_feat)
        
        # 反转为正确顺序
        aligned = aligned[::-1]
        
        # 应用尺度权重
        weights = F.softmax(self.scale_weights, dim=0)
        
        return [w * feat for w, feat in zip(weights, aligned)]


class ImprovedNeck(nn.Module):
    """
    改进的Neck网络
    整合BiFPN、PANet和特征对齐模块
    """
    
    def __init__(self, in_channels=(64, 128, 256, 512), out_channels=256, 
                 use_bifpn=True, use_attention=True):
        super().__init__()
        self.use_bifpn = use_bifpn
        self.use_attention = use_attention
        
        # 通道调整
        self.channel_adjust = nn.ModuleList([
            Conv(c, out_channels, 1, 1) for c in in_channels
        ])
        
        if use_bifpn:
            self.fpn = EnhancedBiFPN([out_channels]*4, out_channels, num_repeats=2)
        else:
            self.fpn = FourScaleFPN([out_channels]*4, out_channels)
        
        # 特征对齐
        self.feature_align = FeatureAlignmentModule(out_channels, num_scales=4)
        
        # 输出通道调整
        self.output_convs = nn.ModuleList([
            Conv(out_channels, c, 1, 1) for c in in_channels
        ])
        
        if use_attention:
            self.output_attention = nn.ModuleList([
                CBAM(c) for c in in_channels
            ])
    
    def forward(self, inputs):
        """
        Args:
            inputs: backbone特征 [C2, C3, C4, C5]
        Returns:
            outputs: neck特征 [P2, P3, P4, P5]
        """
        # 通道调整
        features = [conv(x) for conv, x in zip(self.channel_adjust, inputs)]
        
        # FPN融合
        features = self.fpn(features)
        
        # 特征对齐
        features = self.feature_align(features)
        
        # 输出通道调整
        outputs = [conv(feat) for conv, feat in zip(self.output_convs, features)]
        
        # 注意力增强
        if self.use_attention:
            outputs = [attn(feat) for attn, feat in zip(self.output_attention, outputs)]
        
        return outputs


if __name__ == "__main__":
    print("Testing Neck modules...")
    
    # 模拟backbone输出
    c2 = torch.randn(2, 64, 160, 160)   # 1/4
    c3 = torch.randn(2, 128, 80, 80)    # 1/8
    c4 = torch.randn(2, 256, 40, 40)    # 1/16
    c5 = torch.randn(2, 512, 20, 20)    # 1/32
    inputs = [c2, c3, c4, c5]
    
    print("Input shapes:")
    for i, x in enumerate(inputs):
        print(f"  C{i+2}: {x.shape}")
    
    # 测试四尺度FPN
    print("\nTesting FourScaleFPN...")
    fpn = FourScaleFPN(in_channels=(64, 128, 256, 512), out_channels=256)
    outputs = fpn(inputs)
    print("Output shapes:")
    for i, x in enumerate(outputs):
        print(f"  P{i+2}: {x.shape}")
    
    # 测试增强BiFPN
    print("\nTesting EnhancedBiFPN...")
    bifpn = EnhancedBiFPN(in_channels=(64, 128, 256, 512), out_channels=256)
    outputs = bifpn(inputs)
    print("Output shapes:")
    for i, x in enumerate(outputs):
        print(f"  P{i+2}: {x.shape}")
    
    # 测试改进Neck
    print("\nTesting ImprovedNeck...")
    neck = ImprovedNeck(in_channels=(64, 128, 256, 512), out_channels=256)
    outputs = neck(inputs)
    print("Output shapes:")
    for i, x in enumerate(outputs):
        print(f"  P{i+2}: {x.shape}")
    
    # 计算参数量
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter counts:")
    print(f"FourScaleFPN: {count_params(fpn):,}")
    print(f"EnhancedBiFPN: {count_params(bifpn):,}")
    print(f"ImprovedNeck: {count_params(neck):,}")
    
    print("\nAll tests passed!")
