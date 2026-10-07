"""
Models package for Truncated XLSR front-end and Fake-Mamba back-end classifier.
"""
from .truncated_xlsr import TruncatedXLSR, load_xlsr_300m
from .fake_mamba import FakeMambaClassifier, FakeMambaModel

__all__ = ["TruncatedXLSR", "load_xlsr_300m", "FakeMambaClassifier", "FakeMambaModel"]
