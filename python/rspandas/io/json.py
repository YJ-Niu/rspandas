"""JSON 读写：read_json / to_json

基于 Rust serde_json 实现，释放 GIL 进行解析和序列化，无需 Python json 模块。
"""

from __future__ import annotations

from ..dataframe import DataFrame
from typing import Optional


def read_json(
    path: str,
    orient: str = "records",
    lines: bool = False,
    encoding: str = "utf-8",
) -> DataFrame:
    """从 JSON 文件读取 DataFrame。

    Parameters
    ----------
    path : str
        JSON 文件路径。
    orient : str, default 'records'
        JSON 格式方向：
        - 'records': list[dict] (每行一个 dict)
        - 'columns': dict[str, list] (每列一个 list)
        - 'index': dict[str, dict] (行索引 → 列值)
        - 'split': {'columns': [...], 'data': [[...], ...]}
        - 'values': list[list] (纯二维数组)
    lines : bool, default False
        是否按行读取 JSON (每行一个 JSON 对象)。
    encoding : str, default 'utf-8'
        文件编码。

    Returns
    -------
    DataFrame
    """
    from ..rspandas import _DataFrame
    from ..rspandas import read_json as _read_json_rust

    headers, series_list = _read_json_rust(path, orient, lines, encoding)
    return DataFrame._from_inner(_DataFrame(headers, series_list))


def to_json(
    df: DataFrame,
    path: Optional[str] = None,
    orient: Optional[str] = None,
    date_format: str = "iso",
    double_precision: int = 10,
    force_ascii: bool = True,
    date_unit: str = "ms",
    default_handler=None,
    lines: bool = False,
    compression: str = "infer",
    index: bool = True,
    indent: Optional[int] = None,
) -> Optional[str]:
    """将 DataFrame 写入 JSON 文件或返回 JSON 字符串。

    Parameters
    ----------
    df : DataFrame
        要写入的 DataFrame。
    path : str, optional
        输出文件路径。None 则返回字符串。
    orient : str, default 'records'
        JSON 格式方向。
    date_format : str, default 'iso'
        日期格式（当前仅支持 'iso'，与 pandas 兼容）。
    double_precision : int, default 10
        浮点数精度（兼容签名，当前忽略）。
    force_ascii : bool, default True
        是否强制 ASCII 编码。
    date_unit : str, default 'ms'
        日期单位（兼容签名，当前忽略）。
    default_handler : callable, optional
        无法序列化对象的处理函数（兼容签名，当前忽略）。
    lines : bool, default False
        是否按行输出 JSON。
    compression : str, default 'infer'
        压缩格式（兼容签名，当前忽略）。
    index : bool, default True
        是否包含索引（兼容签名，当前忽略）。
    indent : int, optional
        缩进空格数。

    Returns
    -------
    str or None
    """
    from ..rspandas import write_json as _write_json_rust

    # 默认 orient
    if orient is None:
        orient = "records"

    cols = list(df.columns)
    series_list = [df._inner.get_column(c) for c in cols]
    return _write_json_rust(cols, series_list, path, orient, lines, force_ascii, indent)
