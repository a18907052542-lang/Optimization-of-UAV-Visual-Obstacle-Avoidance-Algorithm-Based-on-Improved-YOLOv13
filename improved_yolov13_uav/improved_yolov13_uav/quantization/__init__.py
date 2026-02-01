"""
Quantization module for Improved YOLOv13 UAV Obstacle Avoidance
Contains INT8 quantization-aware training and model compression utilities
"""

from .qat import (
    QuantizationConfig,
    QuantizableConv2d,
    QuantizableDepthwiseSeparableConv,
    prepare_model_for_qat,
    convert_to_quantized,
    calibrate_model,
    QuantizationAwareTrainer,
    export_quantized_model,
    benchmark_quantized_model
)

__all__ = [
    'QuantizationConfig',
    'QuantizableConv2d',
    'QuantizableDepthwiseSeparableConv',
    'prepare_model_for_qat',
    'convert_to_quantized',
    'calibrate_model',
    'QuantizationAwareTrainer',
    'export_quantized_model',
    'benchmark_quantized_model'
]
