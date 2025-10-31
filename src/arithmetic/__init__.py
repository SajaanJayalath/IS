"""
Arithmetic expression recognition pipeline.

This package bundles preprocessing, segmentation, modeling, and pipeline
utilities dedicated to recognising handwritten arithmetic expressions
containing digits and core operators.
"""

from .pipeline import ArithmeticPipeline

__all__ = ["ArithmeticPipeline"]
