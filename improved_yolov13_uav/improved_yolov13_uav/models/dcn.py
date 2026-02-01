"""
可变形卷积模块 (Deformable Convolution Network)
基于论文公式(5):
y(p_0) = Σ_{p_n∈R} w(p_n) · x(p_0 + p_n + Δp_n) · m_n

其中:
- p_0: 输出特征图上的位置坐标
- R: 常规卷积采样网格（如3×3为9个位置）
- p_n: 枚举网格中的每个采样点
- w(p_n): 对应位置的卷积权重
- Δp_n: 通过额外卷积层学习的位置偏移
- m_n: 调制标量，用于控制每个采样点的贡献

偏移量的学习采用双线性插值处理非整数采样位置，确保梯度平滑传播。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DeformConv2d(nn.Module):
    """
    可变形卷积v2 (Deformable Convolution v2)
    论文3.3节: 通过学习二维偏移场动态调整卷积核的采样位置
    使网络能够根据目标的实际形状和姿态自适应地提取特征
    
    公式(5): y(p_0) = Σ_{p_n∈R} w(p_n) · x(p_0 + p_n + Δp_n) · m_n
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, dilation=1, groups=1, deformable_groups=1, bias=True):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            dilation: 膨胀率
            groups: 分组数
            deformable_groups: 可变形分组数
            bias: 是否使用偏置
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        
        # 采样点数量
        self.num_points = self.kernel_size[0] * self.kernel_size[1]
        
        # 偏移量预测卷积: 输出 2*num_points 个通道 (x和y方向的偏移)
        # 调制标量预测: 输出 num_points 个通道
        self.offset_conv = nn.Conv2d(
            in_channels,
            deformable_groups * 3 * self.num_points,  # 2 for offset + 1 for modulation
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True
        )
        
        # 初始化偏移量为0
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        # 主卷积权重
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # 初始化权重
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            输出特征图 [B, C_out, H_out, W_out]
        """
        # 预测偏移量和调制标量
        offset_mask = self.offset_conv(x)
        
        # 分离偏移量和调制标量
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = torch.sigmoid(mask)  # 调制标量通过Sigmoid归一化到[0,1]
        
        # 应用可变形卷积
        return self.deform_conv2d(x, offset, mask)
    
    def deform_conv2d(self, x, offset, mask):
        """
        实现可变形卷积的核心计算
        使用双线性插值处理非整数采样位置
        """
        b, c, h, w = x.shape
        kh, kw = self.kernel_size
        
        # 生成常规采样网格
        # 对于3x3卷积，网格为 [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(-(kh // 2), kh // 2 + 1, device=x.device, dtype=x.dtype),
            torch.arange(-(kw // 2), kw // 2 + 1, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0)  # [2, kh*kw]
        
        # 计算输出尺寸
        out_h = (h + 2 * self.padding[0] - self.dilation[0] * (kh - 1) - 1) // self.stride[0] + 1
        out_w = (w + 2 * self.padding[1] - self.dilation[1] * (kw - 1) - 1) // self.stride[1] + 1
        
        # 生成输出位置坐标
        out_y, out_x = torch.meshgrid(
            torch.arange(out_h, device=x.device, dtype=x.dtype),
            torch.arange(out_w, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        out_pos = torch.stack([out_x.flatten(), out_y.flatten()], dim=0)  # [2, out_h*out_w]
        
        # 重塑偏移量: [B, 2*num_points, out_h, out_w] -> [B, 2, num_points, out_h*out_w]
        offset = offset.view(b, 2, self.num_points, out_h * out_w)
        
        # 重塑调制标量: [B, num_points, out_h, out_w] -> [B, num_points, out_h*out_w]
        mask = mask.view(b, self.num_points, out_h * out_w)
        
        # 计算采样位置 = 输出位置 * stride - padding + 网格位置 * dilation + 偏移量
        # p_0 + p_n + Δp_n
        sample_pos = (
            out_pos.view(1, 2, 1, -1) * torch.tensor(self.stride, device=x.device, dtype=x.dtype).view(1, 2, 1, 1)
            - torch.tensor(self.padding, device=x.device, dtype=x.dtype).view(1, 2, 1, 1)
            + grid.view(1, 2, -1, 1) * torch.tensor(self.dilation, device=x.device, dtype=x.dtype).view(1, 2, 1, 1)
            + offset
        )
        
        # 归一化到[-1, 1]用于grid_sample
        sample_pos[:, 0, :, :] = 2 * sample_pos[:, 0, :, :] / (w - 1) - 1
        sample_pos[:, 1, :, :] = 2 * sample_pos[:, 1, :, :] / (h - 1) - 1
        
        # 重塑为grid_sample所需格式: [B, out_h*out_w*num_points, 1, 2]
        sample_pos = sample_pos.permute(0, 2, 3, 1).contiguous()
        sample_pos = sample_pos.view(b, -1, 1, 2)
        
        # 双线性插值采样
        sampled = F.grid_sample(
            x, sample_pos, mode='bilinear', padding_mode='zeros', align_corners=True
        )  # [B, C, num_points*out_h*out_w, 1]
        
        # 重塑采样结果
        sampled = sampled.view(b, c, self.num_points, out_h * out_w)
        
        # 应用调制标量 m_n
        mask = mask.unsqueeze(1)  # [B, 1, num_points, out_h*out_w]
        sampled = sampled * mask
        
        # 重塑权重
        weight = self.weight.view(self.out_channels, -1)  # [C_out, C_in/g * kh * kw]
        
        # 重塑采样特征用于矩阵乘法
        sampled = sampled.view(b, c * self.num_points, out_h * out_w)
        
        # 应用卷积权重: [C_out, C_in*num_points] x [B, C_in*num_points, out_h*out_w]
        out = torch.einsum('oi,bio->bo', weight, sampled.view(b, -1, out_h * out_w))
        out = out.view(b, self.out_channels, out_h, out_w)
        
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        
        return out


class DeformConv2dSimple(nn.Module):
    """
    简化版可变形卷积
    使用PyTorch原生操作实现，兼容性更好
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, dilation=1, groups=1, deformable_groups=1, bias=True):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        
        # 偏移量预测
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size * deformable_groups,
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=True
        )
        
        # 调制标量预测
        self.modulator_conv = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size * deformable_groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        
        # 主卷积
        self.regular_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        
        # 初始化
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
    
    def forward(self, x):
        # 预测偏移量
        offset = self.offset_conv(x)
        # 预测调制标量
        modulator = torch.sigmoid(self.modulator_conv(x))
        
        # 这里简化处理：将偏移量和调制标量的影响近似为特征变换
        # 实际应用中建议使用torchvision.ops.deform_conv2d
        x_offset = self._apply_offset(x, offset, modulator)
        
        return self.regular_conv(x_offset)
    
    def _apply_offset(self, x, offset, modulator):
        """应用偏移量的简化实现"""
        b, c, h, w = x.shape
        
        # 生成采样网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
        
        # 添加偏移量（归一化）
        offset_x = offset[:, 0::2, :, :].mean(dim=1, keepdim=True)
        offset_y = offset[:, 1::2, :, :].mean(dim=1, keepdim=True)
        
        # 调整偏移量尺寸
        offset_x = F.interpolate(offset_x, size=(h, w), mode='bilinear', align_corners=True)
        offset_y = F.interpolate(offset_y, size=(h, w), mode='bilinear', align_corners=True)
        
        # 归一化偏移量
        offset_x = offset_x.squeeze(1) * 2 / w
        offset_y = offset_y.squeeze(1) * 2 / h
        
        # 应用偏移
        grid[..., 0] = grid[..., 0] + offset_x
        grid[..., 1] = grid[..., 1] + offset_y
        
        # 采样
        x_offset = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # 应用调制
        modulator = F.interpolate(modulator.mean(dim=1, keepdim=True), size=(h, w), 
                                  mode='bilinear', align_corners=True)
        x_offset = x_offset * modulator
        
        return x_offset


class DCNv2(nn.Module):
    """
    Deformable Convolutional Networks v2
    使用torchvision实现（如果可用）或回退到简化版本
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, deformable_groups=1, bias=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        
        # 尝试使用torchvision的可变形卷积
        try:
            from torchvision.ops import DeformConv2d as TorchDeformConv2d
            self.use_torch_dcn = True
            
            # 偏移量和调制标量预测
            self.offset_conv = nn.Conv2d(
                in_channels,
                deformable_groups * 3 * kernel_size * kernel_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True
            )
            nn.init.constant_(self.offset_conv.weight, 0.)
            nn.init.constant_(self.offset_conv.bias, 0.)
            
            self.deform_conv = TorchDeformConv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=bias
            )
        except ImportError:
            self.use_torch_dcn = False
            self.deform_conv = DeformConv2dSimple(
                in_channels, out_channels, kernel_size, stride,
                padding, dilation, groups, deformable_groups, bias
            )
    
    def forward(self, x):
        if self.use_torch_dcn:
            out = self.offset_conv(x)
            o1, o2, mask = torch.chunk(out, 3, dim=1)
            offset = torch.cat([o1, o2], dim=1)
            mask = torch.sigmoid(mask)
            return self.deform_conv(x, offset, mask)
        else:
            return self.deform_conv(x)


class DeformableConvBlock(nn.Module):
    """
    可变形卷积块
    包含可变形卷积 + BN + 激活函数
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, deformable_groups=1, act=True):
        super().__init__()
        self.dcn = DCNv2(in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, deformable_groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.dcn(x)))


class DeformableBottleneck(nn.Module):
    """
    使用可变形卷积的Bottleneck模块
    """
    
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )
        self.cv2 = DeformableConvBlock(c_, c2, 3, 1, 1)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


if __name__ == "__main__":
    print("Testing Deformable Convolution modules...")
    
    x = torch.randn(2, 64, 80, 80)
    
    # 测试简化版DCN
    dcn_simple = DeformConv2dSimple(64, 128, 3, 1, 1)
    out = dcn_simple(x)
    print(f"DeformConv2dSimple: {x.shape} -> {out.shape}")
    
    # 测试DCNv2包装器
    dcnv2 = DCNv2(64, 128, 3, 1, 1)
    out = dcnv2(x)
    print(f"DCNv2: {x.shape} -> {out.shape}")
    
    # 测试可变形卷积块
    dcn_block = DeformableConvBlock(64, 128, 3, 1, 1)
    out = dcn_block(x)
    print(f"DeformableConvBlock: {x.shape} -> {out.shape}")
    
    # 测试可变形Bottleneck
    dcn_bottleneck = DeformableBottleneck(64, 64)
    out = dcn_bottleneck(x)
    print(f"DeformableBottleneck: {x.shape} -> {out.shape}")
    
    # 计算参数量
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter counts:")
    print(f"DeformConv2dSimple: {count_params(dcn_simple):,}")
    print(f"DCNv2: {count_params(dcnv2):,}")
    print(f"DeformableConvBlock: {count_params(dcn_block):,}")
    print(f"DeformableBottleneck: {count_params(dcn_bottleneck):,}")
    
    print("\nAll tests passed!")
