"""rspandas: pandas-like library built on Rust.

A drop-in pandas-like API where the heavy lifting is done in Rust.
"""

from .series import Series
from .dataframe import DataFrame

__version__ = "0.1.0"
__all__ = ["Series", "DataFrame", "__version__"]
