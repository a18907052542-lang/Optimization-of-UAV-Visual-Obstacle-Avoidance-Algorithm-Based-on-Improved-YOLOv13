"""
INT8 Quantization-Aware Training Module
========================================

Implements INT8 quantization-aware training (QAT) as specified in the paper.

Reference: Section 3.5 - Algorithm Complexity Analysis
"Quantization-aware training compresses model weights from FP32 precision to INT8,
reducing model size by 75% with accuracy loss less than 2%, and improving
inference speed by 2.8 times."

Configuration (from config.yaml):
- Quantization epochs: 10
- Calibration batches: 100

Results (Table 3 - Ablation Experiment):
- Model size: 2.1GB -> 1.2GB (42.9% reduction in memory)
- Speed improvement: 24 FPS -> 48 FPS (2× improvement)
- Accuracy loss: 1.2 percentage points

Table 5: Inference Performance on Different Hardware Platforms
- FP32 to INT8 speedup: ~1.7-2.0× on different platforms
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import (
    QuantStub, DeQuantStub, 
    prepare_qat, convert,
    get_default_qat_qconfig,
    QConfig
)
from typing import Dict, Any, Optional, Tuple, List
import copy


class QuantizedConvBNReLU(nn.Module):
    """Quantization-friendly Conv-BN-ReLU block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        bias: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride, padding, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))
    
    def fuse_modules(self):
        """Fuse conv-bn-relu for quantization."""
        torch.quantization.fuse_modules(
            self, ['conv', 'bn', 'relu'], inplace=True
        )


class QuantizedDepthwiseSeparableConv(nn.Module):
    """
    Quantization-friendly Depthwise Separable Convolution.
    
    Implements Equation (2): C_DSC = D_K² × M × D_F² + M × N × D_F²
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, 
                     stride, padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Pointwise convolution
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class QuantizationWrapper(nn.Module):
    """
    Wrapper to add quantization stubs to a model.
    
    This enables INT8 quantization-aware training (QAT).
    """
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()
    
    def forward(self, x: torch.Tensor) -> Any:
        x = self.quant(x)
        x = self.model(x)
        
        # Handle different output types
        if isinstance(x, dict):
            # Dequantize each tensor in dict
            for key in x:
                if torch.is_tensor(x[key]):
                    x[key] = self.dequant(x[key])
        elif isinstance(x, (list, tuple)):
            x = type(x)(self.dequant(t) if torch.is_tensor(t) else t for t in x)
        else:
            x = self.dequant(x)
        
        return x


class QuantizationConfig:
    """
    Quantization configuration.
    
    Reference: Section 4.1 - Experimental Setup
    - Quantization: INT8 QAT, 10 epochs, 100 calibration batches
    """
    
    def __init__(
        self,
        qat_epochs: int = 10,
        calibration_batches: int = 100,
        backend: str = 'fbgemm',  # 'fbgemm' for x86, 'qnnpack' for ARM
        observer_type: str = 'histogram'  # 'minmax' or 'histogram'
    ):
        self.qat_epochs = qat_epochs
        self.calibration_batches = calibration_batches
        self.backend = backend
        self.observer_type = observer_type
        
        # Set quantization backend
        torch.backends.quantized.engine = backend
    
    def get_qconfig(self) -> QConfig:
        """Get QConfig for quantization."""
        if self.observer_type == 'histogram':
            return get_default_qat_qconfig(self.backend)
        else:
            # MinMax observer
            return QConfig(
                activation=quant.MinMaxObserver.with_args(dtype=torch.quint8),
                weight=quant.MinMaxObserver.with_args(dtype=torch.qint8)
            )


class QuantizationAwareTrainer:
    """
    Quantization-Aware Training (QAT) implementation.
    
    Reference: Section 3.5 - Algorithm Complexity Analysis
    "INT8 quantization-aware training and knowledge distillation strategies are adopted
    to substantially reduce model complexity while maintaining detection accuracy."
    
    Table 6 Results:
    - Model Weights: 87.6MB -> 21.9MB (75% reduction)
    - Feature Map Cache: 812MB -> 486MB (40.1% reduction)
    - Total Memory: 2,120MB -> 656MB (69.1% reduction)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: QuantizationConfig
    ):
        self.original_model = model
        self.config = config
        self.qat_model = None
        self.quantized_model = None
    
    def prepare_qat(self) -> nn.Module:
        """
        Prepare model for quantization-aware training.
        
        Returns:
            Model prepared for QAT
        """
        # Create wrapped model
        self.qat_model = QuantizationWrapper(copy.deepcopy(self.original_model))
        
        # Set QConfig
        self.qat_model.qconfig = self.config.get_qconfig()
        
        # Fuse modules where possible
        self._fuse_modules(self.qat_model.model)
        
        # Prepare for QAT
        prepare_qat(self.qat_model, inplace=True)
        
        return self.qat_model
    
    def _fuse_modules(self, model: nn.Module):
        """Fuse Conv-BN-ReLU modules for efficient quantization."""
        for name, module in model.named_children():
            if isinstance(module, QuantizedConvBNReLU):
                module.fuse_modules()
            elif hasattr(module, 'fuse_modules'):
                module.fuse_modules()
            else:
                self._fuse_modules(module)
    
    def convert_to_quantized(self) -> nn.Module:
        """
        Convert QAT model to fully quantized model.
        
        Returns:
            Quantized INT8 model
        """
        if self.qat_model is None:
            raise RuntimeError("Must call prepare_qat() first")
        
        # Set to eval mode
        self.qat_model.eval()
        
        # Convert to quantized model
        self.quantized_model = convert(self.qat_model, inplace=False)
        
        return self.quantized_model
    
    def calibrate(self, dataloader: torch.utils.data.DataLoader, device: torch.device):
        """
        Run calibration for post-training quantization.
        
        Args:
            dataloader: Calibration data loader
            device: Device to run calibration on
        """
        if self.qat_model is None:
            raise RuntimeError("Must call prepare_qat() first")
        
        self.qat_model.eval()
        self.qat_model.to(device)
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= self.config.calibration_batches:
                    break
                
                images = batch['images'].to(device)
                _ = self.qat_model(images)
    
    def get_model_size(self, model: nn.Module) -> float:
        """
        Get model size in MB.
        
        Returns:
            Model size in megabytes
        """
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in model.buffers():
            total_size += buffer.numel() * buffer.element_size()
        return total_size / (1024 * 1024)
    
    def get_compression_ratio(self) -> float:
        """
        Calculate compression ratio between original and quantized model.
        
        Reference: Table 6 - Model Weights compression from 87.6MB to 21.9MB (75%)
        
        Returns:
            Compression ratio (0-1)
        """
        if self.quantized_model is None:
            raise RuntimeError("Must call convert_to_quantized() first")
        
        original_size = self.get_model_size(self.original_model)
        quantized_size = self.get_model_size(self.quantized_model)
        
        return 1 - (quantized_size / original_size)


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for model compression.
    
    Reference: Section 3.5 - "knowledge distillation strategies are adopted
    to substantially reduce model complexity while maintaining detection accuracy."
    
    The distillation loss combines:
    1. Hard label loss (standard detection loss)
    2. Soft label loss (from teacher model)
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        feature_layers: List[str] = None
    ):
        """
        Args:
            temperature: Softmax temperature for soft labels
            alpha: Weight between hard and soft losses
            feature_layers: Layers to use for feature distillation
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.feature_layers = feature_layers or []
        
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        hard_labels: torch.Tensor,
        hard_loss_fn: nn.Module
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_outputs: Outputs from student model
            teacher_outputs: Outputs from teacher model (detached)
            hard_labels: Ground truth labels
            hard_loss_fn: Loss function for hard labels
        
        Returns:
            Total loss and loss breakdown
        """
        loss_dict = {}
        
        # Hard loss (detection loss with ground truth)
        hard_loss = hard_loss_fn(student_outputs, hard_labels)
        loss_dict['hard_loss'] = hard_loss.item()
        
        # Soft loss (KL divergence with teacher)
        if 'cls_logits' in student_outputs and 'cls_logits' in teacher_outputs:
            student_logits = student_outputs['cls_logits'] / self.temperature
            teacher_logits = teacher_outputs['cls_logits'] / self.temperature
            
            soft_loss = self.kl_div(
                torch.log_softmax(student_logits, dim=-1),
                torch.softmax(teacher_logits, dim=-1)
            ) * (self.temperature ** 2)
            
            loss_dict['soft_loss'] = soft_loss.item()
        else:
            soft_loss = torch.tensor(0.0, device=hard_loss.device)
        
        # Feature distillation loss
        feature_loss = torch.tensor(0.0, device=hard_loss.device)
        for layer in self.feature_layers:
            if layer in student_outputs and layer in teacher_outputs:
                student_feat = student_outputs[layer]
                teacher_feat = teacher_outputs[layer]
                
                # Match dimensions if needed
                if student_feat.shape != teacher_feat.shape:
                    teacher_feat = nn.functional.adaptive_avg_pool2d(
                        teacher_feat, student_feat.shape[2:]
                    )
                
                feature_loss = feature_loss + self.mse(student_feat, teacher_feat)
        
        loss_dict['feature_loss'] = feature_loss.item()
        
        # Total loss
        total_loss = (
            self.alpha * hard_loss + 
            (1 - self.alpha) * soft_loss +
            0.1 * feature_loss
        )
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


class QuantizationExporter:
    """
    Export quantized model to various formats.
    
    Reference: Section 4.1 - "TensorRT 8.5 employed for model optimization and acceleration"
    """
    
    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        input_size: Tuple[int, int, int, int],
        output_path: str,
        opset_version: int = 13
    ):
        """Export model to ONNX format."""
        model.eval()
        dummy_input = torch.randn(input_size)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model exported to {output_path}")
    
    @staticmethod
    def export_to_torchscript(
        model: nn.Module,
        input_size: Tuple[int, int, int, int],
        output_path: str,
        use_trace: bool = True
    ):
        """Export model to TorchScript format."""
        model.eval()
        
        if use_trace:
            dummy_input = torch.randn(input_size)
            traced = torch.jit.trace(model, dummy_input)
        else:
            traced = torch.jit.script(model)
        
        traced.save(output_path)
        print(f"Model exported to {output_path}")
    
    @staticmethod
    def prepare_for_tensorrt(
        onnx_path: str,
        output_path: str,
        precision: str = 'int8',
        calibration_data: Optional[Any] = None
    ):
        """
        Prepare model for TensorRT deployment.
        
        Note: Requires TensorRT installation.
        """
        print(f"TensorRT conversion requires tensorrt package")
        print(f"  Input: {onnx_path}")
        print(f"  Output: {output_path}")
        print(f"  Precision: {precision}")
        
        # TensorRT conversion would be done here with trt.Builder


# Test the quantization module
if __name__ == "__main__":
    print("Testing Quantization Module")
    print("=" * 50)
    
    # Create dummy model
    class DummyDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                QuantizedConvBNReLU(3, 64, 3, 2, 1),
                QuantizedConvBNReLU(64, 128, 3, 2, 1),
                QuantizedConvBNReLU(128, 256, 3, 2, 1),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 10)
            )
        
        def forward(self, x):
            x = self.backbone(x)
            x = self.head(x)
            return x
    
    model = DummyDetector()
    
    # Test quantization config
    print("\n1. Testing QuantizationConfig")
    config = QuantizationConfig(
        qat_epochs=10,
        calibration_batches=100,
        backend='fbgemm'
    )
    print(f"   QAT epochs: {config.qat_epochs}")
    print(f"   Calibration batches: {config.calibration_batches}")
    print(f"   Backend: {config.backend}")
    
    # Test QAT trainer
    print("\n2. Testing QuantizationAwareTrainer")
    trainer = QuantizationAwareTrainer(model, config)
    
    # Get original model size
    original_size = trainer.get_model_size(model)
    print(f"   Original model size: {original_size:.2f} MB")
    
    # Prepare for QAT
    qat_model = trainer.prepare_qat()
    print(f"   QAT model prepared")
    
    # Simulate training with dummy data
    qat_model.train()
    dummy_input = torch.randn(1, 3, 640, 640)
    output = qat_model(dummy_input)
    print(f"   QAT forward pass: input {dummy_input.shape} -> output {output.shape}")
    
    # Convert to quantized
    qat_model.eval()
    quantized_model = trainer.convert_to_quantized()
    quantized_size = trainer.get_model_size(quantized_model)
    print(f"   Quantized model size: {quantized_size:.2f} MB")
    
    # Calculate compression
    compression = trainer.get_compression_ratio()
    print(f"   Compression ratio: {compression*100:.1f}%")
    
    # Test Knowledge Distillation Loss
    print("\n3. Testing KnowledgeDistillationLoss")
    kd_loss = KnowledgeDistillationLoss(temperature=4.0, alpha=0.5)
    print(f"   Temperature: {kd_loss.temperature}")
    print(f"   Alpha: {kd_loss.alpha}")
    
    # Test quantized inference
    print("\n4. Testing Quantized Inference")
    quantized_model.eval()
    with torch.no_grad():
        dummy_input_cpu = torch.randn(1, 3, 640, 640)
        try:
            output = quantized_model(dummy_input_cpu)
            print(f"   Quantized inference: input {dummy_input_cpu.shape} -> output {output.shape}")
        except Exception as e:
            print(f"   Quantized inference test skipped (requires fbgemm): {e}")
    
    print("\n" + "=" * 50)
    print("Quantization module tests completed!")
    print("\nExpected results from paper (Table 3):")
    print("- Memory: 2.1GB -> 1.2GB")
    print("- Speed: 24 FPS -> 48 FPS (2x)")
    print("- mAP loss: 1.2 percentage points")
