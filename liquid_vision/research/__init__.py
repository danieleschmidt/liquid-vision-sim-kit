"""
Research module for Liquid Vision Sim-Kit.
Publication-ready benchmarks and experimental frameworks.
"""

from .research_framework import (
    ResearchBenchmark,
    ExperimentConfig,
    BenchmarkResult,
    StatisticalAnalysis,
    PerformanceBenchmark,
    create_research_experiment
)

__all__ = [
    'ResearchBenchmark',
    'ExperimentConfig', 
    'BenchmarkResult',
    'StatisticalAnalysis',
    'PerformanceBenchmark',
    'create_research_experiment'
]