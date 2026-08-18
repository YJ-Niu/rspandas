"""Excel 读写：ExcelWriter / read_excel / to_excel

由 rspandas/io.py 拆分而来，向后兼容通过 :mod:`rspandas.io` 包入口保证。
"""

from __future__ import annotations

from ..dataframe import DataFrame
from ..series import Series  # noqa: F401  # 部分函数需要
from typing import List, Tuple, Union


class ExcelWriter:
    """Excel 写入器，支持将多个 DataFrame 写入同一个文件的不同 sheet。

    用法:
        with ExcelWriter('output.xlsx') as writer:
            df1.to_excel(writer, sheet_name='Sheet1')
            df2.to_excel(writer, sheet_name='Sheet2')
    """

    def __init__(self, path: str):
        self._path = path
        self._sheets: List[Tuple[str, DataFrame, bool, bool]] = []

    def write(
        self,
        df: DataFrame,
        sheet_name: str = "Sheet1",
        index: bool = True,
        header: bool = True,
    ):
        """将 DataFrame 写入指定 sheet。"""
        self._sheets.append((sheet_name, df, header, index))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()

    def close(self):
        """关闭写入器并保存文件。"""
        from ..rspandas import write_xlsx_multi as _write_xlsx_multi

        # 使用列表推导式替代显式 for 循环构建 sheets_data
        sheets_data = [
            (
                sheet_name,
                list(df.columns),
                [df._inner.get_column(c) for c in df.columns],
                include_header,
                include_index,
            )
            for sheet_name, df, include_header, include_index in self._sheets
        ]

        _write_xlsx_multi(self._path, sheets_data)


# ============================================================================
# JSON
# ============================================================================


def read_excel(
    path: str,
    sheet_name: Union[str, int] = 0,
    header: int = 0,
    **kwargs,
) -> DataFrame:
    """从 Excel 文件读取 DataFrame (使用 Rust calamine 后端)。

    Parameters
    ----------
    path : str
        Excel 文件路径 (.xlsx / .xls / .ods)。
    sheet_name : str or int, default 0
        工作表名称或索引。
    header : int, default 0
        用作列名的行号。
    **kwargs
        忽略 (兼容 pandas 签名)。

    Returns
    -------
    DataFrame
    """
    from ..rspandas import _DataFrame
    from ..rspandas import read_xlsx as _read_xlsx

    if isinstance(sheet_name, int):
        cols, series_list = _read_xlsx(path, None, sheet_name, header)
    else:
        cols, series_list = _read_xlsx(path, sheet_name, None, header)

    return DataFrame._from_inner(_DataFrame(cols, series_list))


def to_excel(
    df: DataFrame,
    path: str,
    sheet_name: str = "Sheet1",
    index: bool = True,
    header: bool = True,
    **kwargs,
) -> None:
    """将 DataFrame 写入 Excel 文件 (使用 Rust rust_xlsxwriter 后端)。

    Parameters
    ----------
    df : DataFrame
        要写入的 DataFrame。
    path : str
        输出文件路径。
    sheet_name : str, default 'Sheet1'
        工作表名称。
    index : bool, default True
        是否写入行索引。
    header : bool, default True
        是否写入列名。
    **kwargs
        忽略 (兼容 pandas 签名)。
    """
    from ..rspandas import write_xlsx as _write_xlsx

    cols = list(df.columns)
    series_list = [df._inner.get_column(c) for c in cols]
    _write_xlsx(path, cols, series_list, sheet_name, header, index)


# ============================================================================
# Parquet
# ============================================================================
