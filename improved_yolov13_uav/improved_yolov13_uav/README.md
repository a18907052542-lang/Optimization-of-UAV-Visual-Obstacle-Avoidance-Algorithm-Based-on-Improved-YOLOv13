# Improved YOLOv13 for UAV Visual Obstacle Avoidance

## 项目简介
本项目实现了论文《Optimization of UAV Visual Obstacle Avoidance Algorithm Based on Improved YOLOv13 in Complex Scenarios》中的改进YOLOv13算法。

## 主要创新点
1. **P2高分辨率检测层**: 增强微小目标感知能力
2. **可变形卷积单元**: 自适应目标几何形变
3. **空间-通道联合注意力机制**: 增强关键特征表达
4. **INT8量化感知训练**: 模型轻量化部署

## 项目结构
```
improved_yolov13_uav/
├── models/
│   ├── __init__.py
│   ├── common.py          # 基础模块（DSC, Ghost等）
│   ├── attention.py       # 注意力机制模块
│   ├── dcn.py            # 可变形卷积模块
│   ├── neck.py           # 特征金字塔网络
│   ├── head.py           # 检测头
│   └── yolov13_improved.py # 完整网络
├── losses/
│   ├── __init__.py
│   └── loss.py           # 损失函数
├── data/
│   ├── __init__.py
│   ├── dataset.py        # 数据集加载
│   └── augmentation.py   # 数据增强
├── obstacle_avoidance/
│   ├── __init__.py
│   └── decision.py       # 避障决策模块
├── quantization/
│   ├── __init__.py
│   └── qat.py            # 量化感知训练
├── utils/
│   ├── __init__.py
│   └── metrics.py        # 评估指标
├── configs/
│   └── config.yaml       # 配置文件
├── train.py              # 训练脚本
├── evaluate.py           # 评估脚本
└── requirements.txt      # 依赖包
```

## 环境配置
```bash
pip install -r requirements.txt
```

## 数据集
- VisDrone2019: https://github.com/VisDrone/VisDrone-Dataset
- UAVDT: https://sites.google.com/view/grli-uavdt

## 训练
```bash
python train.py --config configs/config.yaml
```

## 评估
```bash
python evaluate.py --weights checkpoints/best.pt --data data/visdrone.yaml
```

## 实验结果
| 指标 | YOLOv13-S | Ours | 提升 |
|------|-----------|------|------|
| mAP@0.5 | 41.6% | 44.8% | +3.2% |
| APS | 17.4% | 21.9% | +25.9% |
| FPS(TX2) | 32 | 48 | +50% |

## 引用
如果本项目对您的研究有帮助，请引用原论文。
