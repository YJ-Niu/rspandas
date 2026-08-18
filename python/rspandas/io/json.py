"""JSON 读写：read_json / to_json

由 rspandas/io.py 拆分而来，向后兼容通过 :mod:`rspandas.io` 包入口保证。
"""

from __future__ import annotations

from ..dataframe import DataFrame
from ..series import Series  # noqa: F401  # 部分函数需要
from typing import Any, Dict, List, Optional, Tuple, Union

import json as _json
import pickle as _pickle


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
    with open(path, "r", encoding=encoding) as f:
        if lines:
            records = [_json.loads(line) for line in f if line.strip()]
            return DataFrame(records)
        raw = _json.load(f)

    if orient == "records":
        return DataFrame(raw)
    elif orient == "columns":
        return DataFrame(raw)
    elif orient == "index":
        # 使用列表推导式替代显式 for 循环
        records = [{"index": idx, **row_dict} for idx, row_dict in raw.items()]
        return DataFrame(records)
    elif orient == "split":
        cols = raw.get("columns", [])
        data = raw.get("data", [])
        return DataFrame(data, columns=cols)
    elif orient == "values":
        return DataFrame(raw)
    else:
        raise ValueError(f"Unknown orient: {orient}")


def to_json(
    df: DataFrame,
    path: Optional[str] = None,
    orient: str = "records",
    lines: bool = False,
    force_ascii: bool = False,
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
    lines : bool, default False
        是否按行输出 JSON。
    force_ascii : bool, default False
        是否强制 ASCII 编码。
    indent : int, optional
        缩进空格数。

    Returns
    -------
    str or None
    """
    # df.values 返回 list[dict]
    records = df.values

    if orient == "records":
        data = records
    elif orient == "columns":
        data = {col: [row.get(col) for row in records] for col in df.columns}
    elif orient == "index":
        # 使用字典推导式替代显式 for 循环
        data = {str(i): row for i, row in enumerate(records)}
    elif orient == "split":
        data = {
            "columns": list(df.columns),
            "data": [[row.get(c) for c in df.columns] for row in records],
        }
    elif orient == "values":
        data = [[row.get(c) for c in df.columns] for row in records]
    else:
        raise ValueError(f"Unknown orient: {orient}")

    json_kwargs: Dict[str, Any] = {"ensure_ascii": force_ascii}
    if indent is not None:
        json_kwargs["indent"] = indent

    if lines:
        if orient != "records":
            raise ValueError("lines=True requires orient='records'")
        output = "\n".join(_json.dumps(r, **json_kwargs) for r in data)
    else:
        output = _json.dumps(data, **json_kwargs)

    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(output)
        if not output.endswith("\n"):
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n")
        return None
    return output


# ============================================================================
# Excel (使用 Rust 后端 calamine + rust_xlsxwriter，无需 openpyxl)
# ============================================================================
