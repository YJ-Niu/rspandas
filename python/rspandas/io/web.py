"""HTML / Clipboard / XML 读写

由 rspandas/io.py 拆分而来，向后兼容通过 :mod:`rspandas.io` 包入口保证。
"""

from __future__ import annotations

from ..dataframe import DataFrame
from ..series import Series  # noqa: F401  # 部分函数需要
from typing import Optional


def read_html(
    io,
    match=0,
    flavor=None,
    header=0,
    index_col=None,
    skiprows=None,
    attrs=None,
    encoding=None,
    **kwargs,
) -> DataFrame:
    """从 HTML 表格读取 DataFrame。

    需安装 BeautifulSoup4 和 lxml：pip install beautifulsoup4 lxml
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "read_html requires beautifulsoup4 to be installed. "
            "Install with: pip install beautifulsoup4 lxml"
        )

    if isinstance(io, str):
        with open(io, "r", encoding=encoding or "utf-8") as f:
            content = f.read()
    else:
        content = io.read() if hasattr(io, "read") else str(io)

    soup = BeautifulSoup(content, "lxml")
    tables = soup.find_all("table", attrs=attrs or {})

    if not tables:
        return DataFrame()

    if isinstance(match, int):
        table = tables[match] if match < len(tables) else tables[0]
    elif hasattr(match, "__call__"):
        table = next((t for t in tables if match(t)), tables[0])
    else:
        table = tables[0]

    # 解析表格
    rows_data = []
    for tr in table.find_all("tr")[header or 0 :]:  # noqa
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows_data.append(cells)

    if header is not None and rows_data:
        col_names = rows_data[0]
        data_rows = rows_data[1:]
    else:
        col_names = [str(i) for i in range(len(rows_data[0]))] if rows_data else []
        data_rows = rows_data

    data = {
        col_names[i] if i < len(col_names) else str(i): [
            r[i] if i < len(r) else None for r in data_rows
        ]
        for i in range(max(len(r) for r in data_rows) if data_rows else 0)
    }
    return DataFrame(data)


def to_html(df: DataFrame, path=None, index: bool = True, **kwargs) -> Optional[str]:
    """将 DataFrame 写入 HTML 文件或返回 HTML 字符串。

    基于 Rust 层构建 HTML 字符串，释放 GIL 避免 Python 层循环开销。
    """
    from ..rspandas import to_html as _to_html_rust

    cols = list(df.columns)
    series_list = [df._inner.get_column(c) for c in cols]
    html_content = _to_html_rust(cols, series_list, index)

    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return None
    return html_content


def read_clipboard(**kwargs) -> DataFrame:
    """从系统剪贴板读取 DataFrame。
    需安装 pyperclip：pip install pyperclip
    """
    try:
        import pyperclip

        text = pyperclip.paste()
        # 尝试用 read_csv 解析（以制表符分隔为默认）
        import io as _io

        from . import read_csv as _read_csv

        return _read_csv(_io.StringIO(text), sep="\t")
    except ImportError:
        raise ImportError(
            "read_clipboard requires pyperclip to be installed. "
            "Install with: pip install pyperclip"
        )


def to_clipboard(df: DataFrame, excel: bool = True, **kwargs) -> None:
    """将 DataFrame 写入系统剪贴板。

    需安装 pyperclip：pip install pyperclip
    """
    try:
        import pyperclip

        from . import to_csv

        content = to_csv(df)
        pyperclip.copy(content)
    except ImportError:
        raise ImportError(
            "to_clipboard requires pyperclip to be installed. "
            "Install with: pip install pyperclip"
        )


def read_xml(
    path_or_buffer,
    xpath_regex: str = ".//row",
    row_name: str = "row",
    **kwargs,
) -> DataFrame:
    """从 XML 文件读取 DataFrame。

    需安装 lxml：pip install lxml
    """
    try:
        from lxml import etree
    except ImportError:
        raise ImportError(
            "read_xml requires lxml to be installed. " "Install with: pip install lxml"
        )

    if isinstance(path_or_buffer, str) and not path_or_buffer.strip().startswith("<"):
        tree = etree.parse(path_or_buffer)
        root = tree.getroot()
    else:
        if hasattr(path_or_buffer, "read"):
            content = path_or_buffer.read()
        else:
            content = path_or_buffer
        root = etree.fromstring(
            content.encode() if isinstance(content, str) else content
        )

    rows = root.findall(xpath_regex)
    if not rows:
        return DataFrame()

    # 收集所有列名
    all_cols = set()
    row_data_list = []
    for row in rows:
        row_dict = {}
        for child in row:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            row_dict[tag] = child.text
            all_cols.add(tag)
        # 也检查属性
        for attr_name, attr_value in row.attrib.items():
            row_dict[attr_name] = attr_value
            all_cols.add(attr_name)
        row_data_list.append(row_dict)

    # 构建 DataFrame
    data = {col: [row.get(col) for row in row_data_list] for col in all_cols}
    return DataFrame(data)


def to_xml(
    df: DataFrame,
    path_or_buffer=None,
    index: bool = True,
    root_name: str = "data",
    row_name: str = "row",
    **kwargs,
) -> Optional[str]:
    """将 DataFrame 写入 XML 文件或返回 XML 字符串。

    基于 Rust 层构建 XML 字符串，释放 GIL 避免 Python 层循环开销。
    """
    from ..rspandas import to_xml as _to_xml_rust

    cols = list(df.columns)
    series_list = [df._inner.get_column(c) for c in cols]
    xml_output = _to_xml_rust(cols, series_list, index, root_name, row_name)

    if path_or_buffer:
        if hasattr(path_or_buffer, "write"):
            path_or_buffer.write(xml_output)
        else:
            with open(path_or_buffer, "w", encoding="utf-8") as f:
                f.write(xml_output)
        return None
    return xml_output
