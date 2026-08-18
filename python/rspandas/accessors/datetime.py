"""DatetimeAccessor 时间访问器

由 rspandas/series.py 拆分而来。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from ..dataframe import DataFrame
    from ..series import Series
from typing import Any, Optional, Union


class DatetimeAccessor:
    """Series 的 dt 访问器，提供日期时间操作。"""

    def __init__(self, series: Series):
        self._s = series

    def _get_dt_values(self) -> list:
        """获取 datetime 对象列表，优先使用 _dt_values 缓存。

        若无缓存，尝试将 ISO 字符串解析为 datetime。
        """
        # 优先使用缓存的 datetime 对象
        dt_vals = getattr(self._s, "_dt_values", None)
        if dt_vals is not None and len(dt_vals) > 0:
            return dt_vals
        # 回退：尝试解析 ISO 字符串为 datetime
        from datetime import datetime
        from .._datetime import _parse_iso

        out = []
        for v in self._s.values:
            if isinstance(v, datetime):
                out.append(v)
            elif isinstance(v, str):
                try:
                    out.append(_parse_iso(v))
                except (ValueError, TypeError):
                    out.append(None)
            else:
                out.append(None)
        return out

    def _apply_dt(self, fn) -> Series:
        """对 datetime 值应用函数，返回新 Series。"""
        dt_vals = self._get_dt_values()
        results = [fn(v) if v is not None else None for v in dt_vals]
        return Series(
            results,
            index=self._s._index,
            name=self._s.name,
        )

    @property
    def year(self) -> Series:
        """返回年份。"""
        return self._apply_dt(lambda x: x.year)

    @property
    def month(self) -> Series:
        """返回月份。"""
        return self._apply_dt(lambda x: x.month)

    @property
    def day(self) -> Series:
        """返回日期。"""
        return self._apply_dt(lambda x: x.day)

    @property
    def hour(self) -> Series:
        """返回小时。"""
        return self._apply_dt(lambda x: x.hour)

    @property
    def minute(self) -> Series:
        """返回分钟。"""
        return self._apply_dt(lambda x: x.minute)

    @property
    def second(self) -> Series:
        """返回秒。"""
        return self._apply_dt(lambda x: x.second)

    @property
    def date(self) -> Series:
        """返回日期部分。"""
        return self._apply_dt(lambda x: x.date())

    @property
    def time(self) -> Series:
        """返回时间部分。"""
        return self._apply_dt(lambda x: x.time())

    @property
    def day_name(self) -> Series:
        """返回星期名称。"""
        return self._apply_dt(lambda x: x.strftime("%A"))

    @property
    def month_name(self) -> Series:
        """返回月份名称。"""
        return self._apply_dt(lambda x: x.strftime("%B"))

    def strftime(self, fmt: str) -> Series:
        """使用指定格式字符串格式化 datetime。

        Parameters
        ----------
        fmt : str
            strftime 格式字符串，例如 "%Y/%m/%d"。

        Returns
        -------
        Series
            格式化后的字符串 Series。
        """
        return self._apply_dt(lambda x: x.strftime(fmt) if x is not None else None)

    # ---------- timedelta 相关 ----------

    def _get_td_values(self) -> list:
        """获取 timedelta 对象列表。"""
        td_vals = getattr(self._s, "_td_values", None)
        if td_vals is not None and len(td_vals) > 0:
            return td_vals
        return []

    @property
    def days(self) -> Series:
        """返回 timedelta 的天数部分。"""
        td_vals = self._get_td_values()
        if td_vals:
            results = [td.days if td is not None else None for td in td_vals]
            return Series(results, index=self._s._index, name=self._s.name)
        # 回退到 datetime 的 day
        return self._apply_dt(lambda x: x.day)

    @property
    def seconds(self) -> Series:
        """返回 timedelta 的秒数部分（不含天数）。"""
        td_vals = self._get_td_values()
        if td_vals:
            results = [td.seconds if td is not None else None for td in td_vals]
            return Series(results, index=self._s._index, name=self._s.name)
        # 回退到 datetime 的 second
        return self._apply_dt(lambda x: x.second)

    @property
    def components(self):
        """返回 timedelta 的各组成部分 DataFrame。

        包含列: days, hours, minutes, seconds, milliseconds, microseconds, nanoseconds
        """
        from ..dataframe import DataFrame

        td_vals = self._get_td_values()
        if not td_vals:
            raise AttributeError("Can only use .dt.components with timedelta values")
        data = {
            "days": [],
            "hours": [],
            "minutes": [],
            "seconds": [],
            "milliseconds": [],
            "microseconds": [],
            "nanoseconds": [],
        }
        for td in td_vals:
            if td is None:
                for k in data:
                    data[k].append(None)
                continue
            total_sec = td.total_seconds()
            days = int(total_sec // 86400)
            remainder = int(total_sec % 86400)
            hours = remainder // 3600
            remainder %= 3600
            minutes = remainder // 60
            seconds = remainder % 60
            microseconds = td.microseconds
            milliseconds = microseconds // 1000
            microseconds %= 1000
            data["days"].append(days)
            data["hours"].append(hours)
            data["minutes"].append(minutes)
            data["seconds"].append(seconds)
            data["milliseconds"].append(milliseconds)
            data["microseconds"].append(microseconds)
            data["nanoseconds"].append(0)
        return DataFrame(data, index=self._s._index)

    # ---------- 时区相关 ----------

    @property
    def tz(self):
        """返回时区信息。"""
        dt_vals = self._get_dt_values()
        for v in dt_vals:
            if v is not None and v.tzinfo is not None:
                return v.tzinfo
        return None

    def tz_localize(self, tz) -> Series:
        """将 naive datetime 本地化为指定时区。

        Parameters
        ----------
        tz : str or tzinfo
            时区名称或时区对象。

        Returns
        -------
        Series
            带时区的 datetime Series。
        """
        from .._datetime import _normalize_tz, _localize

        tzobj = _normalize_tz(tz)
        dt_vals = self._get_dt_values()
        localized = [_localize(v, tzobj) if v is not None else None for v in dt_vals]
        # 使用新的 datetime 值创建 Series
        new_series = Series(
            localized,
            index=self._s._index,
            name=self._s.name,
        )
        # 同步 dt 缓存
        new_series._dt_values = localized
        new_series._dt_tz = tzobj
        return new_series

    def tz_convert(self, tz) -> Series:
        """将带时区的 datetime 转换到另一个时区。

        Parameters
        ----------
        tz : str or tzinfo
            目标时区。

        Returns
        -------
        Series
            转换时区后的 datetime Series。
        """
        from .._datetime import _normalize_tz

        tzobj = _normalize_tz(tz)
        dt_vals = self._get_dt_values()
        converted = []
        for v in dt_vals:
            if v is None:
                converted.append(None)
            elif v.tzinfo is None:
                # naive datetime: 先假设是 UTC，再转换
                from datetime import timezone

                converted.append(v.replace(tzinfo=timezone.utc).astimezone(tzobj))
            else:
                converted.append(v.astimezone(tzobj))
        # 使用新的 datetime 值创建 Series
        new_series = Series(
            converted,
            index=self._s._index,
            name=self._s.name,
        )
        # 同步 dt 缓存
        new_series._dt_values = converted
        new_series._dt_tz = tzobj
        return new_series


# ==============================================================================
# SeriesGroupBy  - Series 分组操作
# ==============================================================================


def __getattr__(name: str):
    """模块级延迟导入，避免与 dataframe.py / series.py 的循环导入。"""
    if name == "DataFrame":
        from ..dataframe import DataFrame as _DF

        return _DF
    if name == "Series":
        from ..series import Series as _S

        return _S
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# 懒加载代理：避免循环导入
# ---------------------------------------------------------------------------
import importlib as _importlib


class _LazyProxy:
    """延迟导入代理，支持 __call__（构造）和 __getattr__（属性访问）。"""

    def __init__(self, mod_path: str, attr_name: str):
        self._mod_path = mod_path
        self._attr_name = attr_name
        self._obj = None

    def _load(self):
        if self._obj is None:
            mod = _importlib.import_module(self._mod_path)
            self._obj = getattr(mod, self._attr_name)
        return self._obj

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._load(), name)

    def __instancecheck__(self, instance):
        return isinstance(instance, self._load())


# 安装懒加载代理
if "Series" not in globals() or not callable(globals().get("Series", None)):
    Series = _LazyProxy("rspandas.series", "Series")
if "DataFrame" not in globals() or not callable(globals().get("DataFrame", None)):
    DataFrame = _LazyProxy("rspandas.dataframe", "DataFrame")
