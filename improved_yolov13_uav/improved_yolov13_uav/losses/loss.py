"""
损失函数模块
基于论文公式(3)和公式(4)

公式(3) - Focal Loss (分类损失):
L_cls = -α_t(1-p_t)^γ log(p_t)

其中:
- p_t: 预测概率
- α_t: 平衡因子
- γ: 聚焦参数，通过调整γ值可以控制对难样本的关注程度

公式(4) - Distribution Focal Loss (定位损失):
L_loc = -((y_{l+1}-y)log(ŷ_l) + (y-y_l)log(ŷ_{l+1}))

其中:
- y: 目标值
- y_l, y_{l+1}: 离散化边界值
- ŷ_l, ŷ_{l+1}: 对应的预测概率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection
    论文公式(3): L_cls = -α_t(1-p_t)^γ log(p_t)
    
    用于处理目标检测中正负样本严重不平衡的问题
    通过调整γ值控制对难分类样本的关注程度
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 平衡因子α，控制正负样本权重，默认0.25
            gamma: 聚焦参数γ，控制难易样本权重，默认2.0
            reduction: 损失聚合方式 ('none', 'mean', 'sum')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        计算Focal Loss
        
        Args:
            pred: 预测logits [N, C] 或 [N, C, H, W]
            target: 目标标签 [N] 或 [N, H, W]
        
        Returns:
            loss: Focal Loss值
        """
        # 获取预测概率
        pred_sigmoid = pred.sigmoid()
        
        # 将target转换为one-hot编码
        if pred.dim() == 2:
            # [N, C]
            num_classes = pred.shape[1]
            target_onehot = F.one_hot(target, num_classes).float()
        else:
            # [N, C, H, W]
            num_classes = pred.shape[1]
            target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        # 计算p_t
        # p_t = p if y=1 else (1-p)
        pt = pred_sigmoid * target_onehot + (1 - pred_sigmoid) * (1 - target_onehot)
        
        # 计算focal权重 (1-p_t)^γ
        focal_weight = (1 - pt) ** self.gamma
        
        # 计算α_t
        # α_t = α if y=1 else (1-α)
        alpha_t = self.alpha * target_onehot + (1 - self.alpha) * (1 - target_onehot)
        
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(pred, target_onehot, reduction='none')
        
        # 应用focal权重和alpha
        focal_loss = alpha_t * focal_weight * bce_loss
        
        # 聚合
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss (DFL)
    论文公式(4): L_loc = -((y_{l+1}-y)log(ŷ_l) + (y-y_l)log(ŷ_{l+1}))
    
    用于边界框回归，将连续回归问题转换为离散分布学习问题
    通过学习目标值两侧的离散概率分布来预测连续值
    """
    
    def __init__(self, reg_max=16, reduction='mean'):
        """
        Args:
            reg_max: 分布的最大值（离散化范围为[0, reg_max]）
            reduction: 损失聚合方式
        """
        super().__init__()
        self.reg_max = reg_max
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        计算DFL损失
        
        Args:
            pred: 预测分布 [N, reg_max+1] 或 [N, 4, reg_max+1]
            target: 目标值 [N] 或 [N, 4]，范围在[0, reg_max]
        
        Returns:
            loss: DFL损失值
        """
        # 确保target在有效范围内
        target = target.clamp(0, self.reg_max - 0.01)
        
        # 获取左右边界索引
        # y_l = floor(y), y_{l+1} = ceil(y)
        target_left = target.long()  # 下边界索引
        target_right = target_left + 1  # 上边界索引
        
        # 计算权重
        # weight_left = y_{l+1} - y
        # weight_right = y - y_l
        weight_left = target_right.float() - target
        weight_right = target - target_left.float()
        
        # 获取预测概率的log
        pred_softmax = F.log_softmax(pred, dim=-1)
        
        # 提取对应位置的概率
        # 使用gather提取索引对应的值
        if pred.dim() == 2:
            # [N, reg_max+1]
            loss_left = F.nll_loss(pred_softmax, target_left, reduction='none')
            loss_right = F.nll_loss(pred_softmax, target_right.clamp(max=self.reg_max), reduction='none')
        else:
            # [N, 4, reg_max+1] -> 需要reshape
            n, num_coords, num_bins = pred.shape
            pred_flat = pred.view(-1, num_bins)
            target_left_flat = target_left.view(-1)
            target_right_flat = target_right.view(-1).clamp(max=self.reg_max)
            
            pred_softmax_flat = F.log_softmax(pred_flat, dim=-1)
            
            loss_left = F.nll_loss(pred_softmax_flat, target_left_flat, reduction='none')
            loss_right = F.nll_loss(pred_softmax_flat, target_right_flat, reduction='none')
            
            loss_left = loss_left.view(n, num_coords)
            loss_right = loss_right.view(n, num_coords)
            weight_left = weight_left.view(n, num_coords)
            weight_right = weight_right.view(n, num_coords)
        
        # 加权求和
        # L = weight_left * loss_left + weight_right * loss_right
        loss = weight_left * loss_left + weight_right * loss_right
        
        # 聚合
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class IoULoss(nn.Module):
    """
    IoU系列损失函数
    支持IoU, GIoU, DIoU, CIoU, SIoU等变体
    """
    
    def __init__(self, loss_type='ciou', reduction='mean', eps=1e-7):
        """
        Args:
            loss_type: 损失类型 ('iou', 'giou', 'diou', 'ciou', 'siou')
            reduction: 聚合方式
            eps: 防止除零的小常数
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, pred, target):
        """
        计算IoU损失
        
        Args:
            pred: 预测框 [N, 4] (x1, y1, x2, y2)
            target: 目标框 [N, 4] (x1, y1, x2, y2)
        
        Returns:
            loss: IoU损失
        """
        # 确保坐标有效
        pred_x1, pred_y1, pred_x2, pred_y2 = pred.unbind(dim=-1)
        target_x1, target_y1, target_x2, target_y2 = target.unbind(dim=-1)
        
        # 计算面积
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        
        # 计算交集
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # 计算并集
        union_area = pred_area + target_area - inter_area + self.eps
        
        # 计算IoU
        iou = inter_area / union_area
        
        if self.loss_type == 'iou':
            loss = 1 - iou
        
        elif self.loss_type == 'giou':
            # 计算最小外接矩形
            enclose_x1 = torch.min(pred_x1, target_x1)
            enclose_y1 = torch.min(pred_y1, target_y1)
            enclose_x2 = torch.max(pred_x2, target_x2)
            enclose_y2 = torch.max(pred_y2, target_y2)
            
            enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + self.eps
            
            giou = iou - (enclose_area - union_area) / enclose_area
            loss = 1 - giou
        
        elif self.loss_type == 'diou':
            # 计算中心点距离
            pred_cx = (pred_x1 + pred_x2) / 2
            pred_cy = (pred_y1 + pred_y2) / 2
            target_cx = (target_x1 + target_x2) / 2
            target_cy = (target_y1 + target_y2) / 2
            
            center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
            
            # 计算对角线距离
            enclose_x1 = torch.min(pred_x1, target_x1)
            enclose_y1 = torch.min(pred_y1, target_y1)
            enclose_x2 = torch.max(pred_x2, target_x2)
            enclose_y2 = torch.max(pred_y2, target_y2)
            
            enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + self.eps
            
            diou = iou - center_dist / enclose_diag
            loss = 1 - diou
        
        elif self.loss_type == 'ciou':
            # DIoU基础
            pred_cx = (pred_x1 + pred_x2) / 2
            pred_cy = (pred_y1 + pred_y2) / 2
            target_cx = (target_x1 + target_x2) / 2
            target_cy = (target_y1 + target_y2) / 2
            
            center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
            
            enclose_x1 = torch.min(pred_x1, target_x1)
            enclose_y1 = torch.min(pred_y1, target_y1)
            enclose_x2 = torch.max(pred_x2, target_x2)
            enclose_y2 = torch.max(pred_y2, target_y2)
            
            enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + self.eps
            
            # 宽高比惩罚
            pred_w = pred_x2 - pred_x1
            pred_h = pred_y2 - pred_y1
            target_w = target_x2 - target_x1
            target_h = target_y2 - target_y1
            
            v = (4 / (math.pi ** 2)) * torch.pow(
                torch.atan(target_w / (target_h + self.eps)) - torch.atan(pred_w / (pred_h + self.eps)), 2
            )
            
            with torch.no_grad():
                alpha = v / (1 - iou + v + self.eps)
            
            ciou = iou - center_dist / enclose_diag - alpha * v
            loss = 1 - ciou
        
        else:  # siou
            # SIoU - 考虑角度和距离
            # 简化实现
            loss = 1 - iou
        
        # 聚合
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss
    将分类分数与定位质量（IoU）结合的损失函数
    """
    
    def __init__(self, beta=2.0, reduction='mean'):
        """
        Args:
            beta: 质量调节因子
            reduction: 聚合方式
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, pred, target, quality):
        """
        Args:
            pred: 预测分数 [N, C]
            target: 目标类别 [N]
            quality: IoU质量分数 [N]
        """
        pred_sigmoid = pred.sigmoid()
        
        # 将quality作为软标签
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target, num_classes).float()
        
        # 用quality替换1
        soft_target = target_onehot * quality.unsqueeze(-1)
        
        # 计算focal权重
        scale_factor = (soft_target - pred_sigmoid).abs().pow(self.beta)
        
        # BCE损失
        bce = F.binary_cross_entropy_with_logits(pred, soft_target, reduction='none')
        
        # 应用focal权重
        loss = scale_factor * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class VarifocalLoss(nn.Module):
    """
    Varifocal Loss
    用于密集目标检测的变焦损失
    """
    
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target, quality):
        """
        Args:
            pred: 预测分数 [N, C]
            target: 目标类别 [N]
            quality: IoU分数 [N]
        """
        pred_sigmoid = pred.sigmoid()
        num_classes = pred.shape[1]
        
        # 创建软标签
        target_onehot = F.one_hot(target, num_classes).float()
        soft_target = target_onehot * quality.unsqueeze(-1)
        
        # 计算权重
        focal_weight = soft_target * (soft_target - pred_sigmoid).abs().pow(self.gamma)
        focal_weight = focal_weight + self.alpha * (1 - soft_target) * pred_sigmoid.pow(self.gamma)
        
        # BCE损失
        bce = F.binary_cross_entropy_with_logits(pred, soft_target, reduction='none')
        
        loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DetectionLoss(nn.Module):
    """
    综合检测损失
    结合分类损失、定位损失和目标置信度损失
    """
    
    def __init__(self, num_classes, reg_max=16, 
                 cls_weight=1.0, loc_weight=1.0, obj_weight=1.0,
                 focal_alpha=0.25, focal_gamma=2.0,
                 iou_type='ciou'):
        """
        Args:
            num_classes: 类别数
            reg_max: DFL最大值
            cls_weight: 分类损失权重
            loc_weight: 定位损失权重
            obj_weight: 目标置信度损失权重
            focal_alpha: Focal Loss的alpha参数
            focal_gamma: Focal Loss的gamma参数
            iou_type: IoU损失类型
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
        self.obj_weight = obj_weight
        
        # 分类损失
        self.cls_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        # 定位损失
        self.dfl_loss = DistributionFocalLoss(reg_max=reg_max)
        self.iou_loss = IoULoss(loss_type=iou_type)
        
        # 目标置信度损失
        self.obj_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        """
        计算综合损失
        
        Args:
            predictions: 模型预测，包含多尺度输出
            targets: 标注信息
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        device = predictions[0]['cls'].device
        
        cls_loss = torch.tensor(0., device=device)
        loc_loss = torch.tensor(0., device=device)
        obj_loss = torch.tensor(0., device=device)
        
        num_pos = 0
        
        for pred, target in zip(predictions, targets):
            cls_pred = pred['cls']  # [B, C, H, W]
            reg_pred = pred['reg']  # [B, 4*(reg_max+1), H, W]
            obj_pred = pred['obj']  # [B, 1, H, W]
            
            # 这里需要与targets进行匹配
            # 简化示例：假设targets已经对齐
            if 'cls_target' in target:
                cls_target = target['cls_target']
                reg_target = target['reg_target']
                obj_target = target['obj_target']
                
                # 分类损失
                cls_loss = cls_loss + self.cls_loss(
                    cls_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes),
                    cls_target.reshape(-1)
                )
                
                # 定位损失（只对正样本计算）
                pos_mask = obj_target > 0
                if pos_mask.sum() > 0:
                    reg_pred_pos = reg_pred.permute(0, 2, 3, 1).reshape(-1, 4, self.reg_max + 1)[pos_mask.reshape(-1)]
                    reg_target_pos = reg_target.reshape(-1, 4)[pos_mask.reshape(-1)]
                    
                    loc_loss = loc_loss + self.dfl_loss(reg_pred_pos, reg_target_pos)
                
                # 目标置信度损失
                obj_loss = obj_loss + self.obj_loss(
                    obj_pred.reshape(-1),
                    obj_target.reshape(-1).float()
                )
                
                num_pos += pos_mask.sum()
        
        # 归一化
        num_pos = max(num_pos, 1)
        
        cls_loss = cls_loss / num_pos
        loc_loss = loc_loss / num_pos
        obj_loss = obj_loss / len(predictions)
        
        # 加权求和
        total_loss = (
            self.cls_weight * cls_loss +
            self.loc_weight * loc_loss +
            self.obj_weight * obj_loss
        )
        
        loss_dict = {
            'total': total_loss,
            'cls': cls_loss,
            'loc': loc_loss,
            'obj': obj_loss
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    print("Testing Loss Functions...")
    
    # 测试Focal Loss
    print("\n1. Testing FocalLoss...")
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    pred = torch.randn(100, 10)
    target = torch.randint(0, 10, (100,))
    loss = focal(pred, target)
    print(f"   FocalLoss: {loss.item():.4f}")
    
    # 测试DFL
    print("\n2. Testing DistributionFocalLoss...")
    dfl = DistributionFocalLoss(reg_max=16)
    pred = torch.randn(100, 4, 17)
    target = torch.rand(100, 4) * 15
    loss = dfl(pred, target)
    print(f"   DFL: {loss.item():.4f}")
    
    # 测试IoU Loss
    print("\n3. Testing IoU Losses...")
    for iou_type in ['iou', 'giou', 'diou', 'ciou']:
        iou_loss = IoULoss(loss_type=iou_type)
        pred_boxes = torch.rand(100, 4) * 100
        pred_boxes[:, 2:] = pred_boxes[:, :2] + torch.rand(100, 2) * 50
        target_boxes = torch.rand(100, 4) * 100
        target_boxes[:, 2:] = target_boxes[:, :2] + torch.rand(100, 2) * 50
        loss = iou_loss(pred_boxes, target_boxes)
        print(f"   {iou_type.upper()} Loss: {loss.item():.4f}")
    
    # 测试Quality Focal Loss
    print("\n4. Testing QualityFocalLoss...")
    qfl = QualityFocalLoss()
    pred = torch.randn(100, 10)
    target = torch.randint(0, 10, (100,))
    quality = torch.rand(100)
    loss = qfl(pred, target, quality)
    print(f"   QFL: {loss.item():.4f}")
    
    # 测试Varifocal Loss
    print("\n5. Testing VarifocalLoss...")
    vfl = VarifocalLoss()
    loss = vfl(pred, target, quality)
    print(f"   VFL: {loss.item():.4f}")
    
    print("\nAll loss function tests passed!")
