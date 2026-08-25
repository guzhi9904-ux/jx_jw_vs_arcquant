"""Stage 1.5 dual-source functional-spectrum experiment."""

from .exact_dual_source import DualSourceGrams, build_dual_source_grams
from .smooth_scaling import smoothquant_scale

__all__ = ["DualSourceGrams", "build_dual_source_grams", "smoothquant_scale"]
