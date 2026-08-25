"""Stage 1.6 factor-rank and spectral-outlier mechanism diagnostics."""

from .factor_spectrum import SpectrumAnalysis, analyze_factor, analyze_psd
from .functional_factors import FunctionalFactorGrams, build_functional_factor_grams

__all__ = [
    "FunctionalFactorGrams",
    "SpectrumAnalysis",
    "analyze_factor",
    "analyze_psd",
    "build_functional_factor_grams",
]
