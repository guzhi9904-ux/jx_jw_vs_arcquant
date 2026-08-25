"""Stage-1 functional-Gram low-rank validation."""

from .build_gram import (
    FunctionalGrams,
    build_functional_gram,
    build_functional_grams,
    sanity_check_grams,
)
from .collect_stats import collect_decoder_operands
from .eig_analysis import EigenAnalysis, analyze_gram
from .split_stability import projector_overlap_curve

__all__ = [
    "EigenAnalysis",
    "FunctionalGrams",
    "analyze_gram",
    "build_functional_gram",
    "build_functional_grams",
    "collect_decoder_operands",
    "projector_overlap_curve",
    "sanity_check_grams",
]
