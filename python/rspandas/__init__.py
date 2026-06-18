"""rspandas: pandas-like library built on Rust.

A drop-in pandas-like API where the heavy lifting is done in Rust.
"""

from .series import Series
from .dataframe import DataFrame
from .rspandas import _Series, _DataFrame  # 重新导出 Rust 类型，供内部使用

__version__ = "0.1.0"
__all__ = ["Series", "DataFrame", "_Series", "_DataFrame", "__version__"]
