"""Python 类型提示桩文件 (PEP 484)。

用于 IDE 自动补全和类型检查。
"""

from typing import Any, Dict, List, Optional, Tuple

class _Series:
    name: Optional[str]
    dtype: str
    shape: Tuple[int]
    size: int
    empty: bool
    nbytes: int
    
    def __init__(self, data: list, name: Optional[str] = ...) -> None:
        ...
    
    def head(self, n: int = ...) -> _Series:
        ...

    def tail(self, n: int = ...) -> _Series:
        ...

    def filter(self, mask: list) -> _Series:
        ...

    def eq_scalar(self, value: Any) -> list:
        ...
    
    def gt_scalar(self, value: Any) -> list:
        ...
    
    def lt_scalar(self, value: Any) -> list:
        ...
    
    def ge_scalar(self, value: Any) -> list:
        ...
    
    def le_scalar(self, value: Any) -> list:
        ...
    
    def sum(self) -> Any:
        ...
    
    def mean(self) -> Any:
        ...
    
    def min(self) -> Any:
        ...
    
    def max(self) -> Any:
        ...
    
    def count(self) -> int:
        ...
    
    def std(self) -> Any:
        ...
    
    def var(self) -> Any:
        ...
    
    def median(self) -> Any:
        ...
    
    def any(self) -> Any:
        ...
    
    def all(self) -> Any:
        ...
    
    def isnull(self) -> list:
        ...
    
    def notnull(self) -> list:
        ...
    
    def dropna(self) -> _Series:
        ...
    
    def fillna(self, value: Any) -> _Series:
        ...
    
    def unique(self) -> _Series:
        ...
    
    def nunique(self) -> int:
        ...
    
    def value_counts(self) -> tuple:
        ...
    
    def to_string_vec(self) -> List[str]:
        ...
    
    def values(self) -> list:
        ...

class _DataFrame:
    shape: Tuple[int, int]
    size: int
    empty: bool

    def __init__(self, columns: List[str], series: List[_Series]) -> None:
        ...

    def get_column(self, name: str) -> _Series:
        ...

    def get_column_at(self, idx: int) -> _Series:
        ...

    def column_index(self, name: str) -> Optional[int]:
        ...
        
    def head(self, n: int = ...) -> _DataFrame:
        ...

    def tail(self, n: int = ...) -> _DataFrame:
        ...
        
    def filter_rows(self, mask: list) -> _DataFrame:
        ...

    def dropna(self) -> _DataFrame:
        ...

    def fillna(self, fill_dict: dict) -> _DataFrame:
        ...

    def to_rows(self) -> list:
        ...

    def columns_to_string(self) -> Dict[str, List[str]]:
        ...

    def dtypes(self) -> Dict[str, str]:
        ...

    def columns(self) -> List[str]:
        ...
