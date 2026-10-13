"""LoFi: calibration-free KV cache compression via low-frequency RoPE channels."""

from .config import PRESETS, LoFiConfig
from .core import CompressionResult, CompressionStats, compress_kv_cache, mass_logit_bias
from .frequency import high_freq_indices, low_freq_indices

__all__ = [
    "PRESETS",
    "CompressionResult",
    "CompressionStats",
    "LoFiConfig",
    "compress_kv_cache",
    "high_freq_indices",
    "low_freq_indices",
    "mass_logit_bias",
]
