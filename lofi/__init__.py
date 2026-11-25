"""LoFi: Low-Frequency Key Similarity for Query-Agnostic KV Cache Eviction."""

from .compress import (
    CompressionStats,
    Layout,
    LoFiConfig,
    Prefill,
    compress,
    disable_sliding_window,
    generate,
    greedy_decode,
    layout,
    prefill,
    rope_base,
)
from .frequency import (
    effective_ranges,
    low_frequency_pairs,
    low_high_channels,
    recent_window,
    rope_thetas,
)
from .selection import SelectionOutput, select

__version__ = "1.1.0"

__all__ = [
    "CompressionStats",
    "Layout",
    "LoFiConfig",
    "Prefill",
    "SelectionOutput",
    "compress",
    "disable_sliding_window",
    "effective_ranges",
    "generate",
    "greedy_decode",
    "layout",
    "low_frequency_pairs",
    "low_high_channels",
    "prefill",
    "recent_window",
    "rope_base",
    "rope_thetas",
    "select",
]
