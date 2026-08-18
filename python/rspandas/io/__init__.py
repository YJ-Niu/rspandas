"""IO 子包：所有读写函数的统一入口。

按功能拆分到子模块：
- :mod:`._common`：内部辅助（``_TextFileReader`` / ``StreamDataFrame`` 等）
- :mod:`.csv`：CSV 读写
- :mod:`.excel`：Excel 读写
- :mod:`.json`：JSON 读写
- :mod:`.arrow`：Parquet / Feather (Arrow IPC)
- :mod:`.pickle`：Pickle
- :mod:`.sql`：SQL 读写
- :mod:`.web`：HTML / Clipboard / XML
- :mod:`.other`：ORC / Stata / HDF / SPSS / GBQ

向后兼容：``from rspandas.io import read_csv`` 仍可用。
"""

from __future__ import annotations

from ._common import (
    StreamDataFrame,
    _TextFileReader,  # noqa: F401  # 内部 API，供高级用户使用
    _NoDefault,  # noqa: F401
    _apply_dtype,  # noqa: F401
    _infer_column_type,  # noqa: F401
    _parse_cols_items,  # noqa: F401
    _parse_date_series,  # noqa: F401
    _read_content,  # noqa: F401
    _rows_to_dict,  # noqa: F401
)
from .arrow import read_feather, read_parquet, to_feather, to_parquet
from .csv import read_csv, read_csv_chunked, to_csv
from .excel import ExcelWriter, read_excel, to_excel
from .json import read_json, to_json
from .other import (
    read_gbq,
    read_hdf,
    read_orc,
    read_spss,
    read_stata,
    to_gbq,
    to_hdf,
    to_orc,
    to_stata,
)
from .pickle import read_pickle, to_pickle
from .sql import read_sql, read_sql_query, read_sql_table, to_sql, to_sql_batch
from .web import read_clipboard, read_html, read_xml, to_clipboard, to_html, to_xml

__all__ = [
    "ExcelWriter",
    "StreamDataFrame",
    "read_csv",
    "read_csv_chunked",
    "read_clipboard",
    "read_excel",
    "read_feather",
    "read_gbq",
    "read_hdf",
    "read_html",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_spss",
    "read_sql",
    "read_sql_query",
    "read_sql_table",
    "read_stata",
    "read_xml",
    "to_clipboard",
    "to_excel",
    "to_feather",
    "to_gbq",
    "to_hdf",
    "to_html",
    "to_json",
    "to_orc",
    "to_parquet",
    "to_pickle",
    "to_sql",
    "to_sql_batch",
    "to_stata",
    "to_xml",
    "to_csv",
]
