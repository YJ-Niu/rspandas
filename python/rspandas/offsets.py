"""日期偏移量 (pandas offsets 兼容)。

提供常见的日期偏移类，用于日期范围生成和位移：
- Day: 日历日偏移
- BusinessDay: 工作日偏移
- MonthEnd: 月末偏移
- MonthStart: 月初偏移
- YearEnd: 年末偏移
- YearStart: 年初偏移
"""

from datetime import datetime, timedelta


class Day:
    """日历日偏移 (n 天)。"""

    def __init__(self, n: int = 1):
        self.n = int(n)

    def __repr__(self) -> str:
        return f"<Day: n={self.n}>"

    def __add__(self, other):
        if isinstance(other, datetime):
            return other + timedelta(days=self.n)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)


class BusinessDay:
    """工作日偏移 (跳过周末)。"""

    def __init__(self, n: int = 1):
        self.n = int(n)

    def __repr__(self) -> str:
        return f"<BusinessDay: n={self.n}>"

    def __add__(self, other):
        if isinstance(other, datetime):
            result = other
            remaining = self.n
            step = 1 if remaining > 0 else -1
            while remaining != 0:
                result = result + timedelta(days=step)
                if result.weekday() < 5:  # 周一到周五
                    remaining -= step
            return result
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)


class MonthEnd:
    """月末偏移 (到自然月的最后一天)。"""

    def __init__(self, n: int = 1):
        self.n = int(n)

    def __repr__(self) -> str:
        return f"<MonthEnd: n={self.n}>"

    def __add__(self, other):
        if isinstance(other, datetime):
            import calendar
            year, month = other.year, other.month
            # Check if the date is already at month end
            last_day_current = calendar.monthrange(year, month)[1]
            if other.day == last_day_current:
                # Already at month end, roll forward n months
                month += self.n
            else:
                # Not at month end, roll to current month end, then forward n-1 months
                month += self.n - 1
            while month > 12:
                year += 1
                month -= 12
            while month < 1:
                year -= 1
                month += 12
            last_day = calendar.monthrange(year, month)[1]
            return datetime(year, month, last_day, other.hour, other.minute, other.second, other.microsecond)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)


class MonthStart:
    """月初偏移 (到自然月的第一天)。"""

    def __init__(self, n: int = 1):
        self.n = int(n)

    def __repr__(self) -> str:
        return f"<MonthStart: n={self.n}>"

    def __add__(self, other):
        if isinstance(other, datetime):
            year, month = other.year, other.month
            month += self.n
            while month > 12:
                year += 1
                month -= 12
            while month < 1:
                year -= 1
                month += 12
            return datetime(year, month, 1)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)


class YearEnd:
    """年末偏移 (到自然年的最后一天)。"""

    def __init__(self, n: int = 1, month: int = 12):
        self.n = int(n)
        self.month = month

    def __repr__(self) -> str:
        return f"<YearEnd: n={self.n}, month={self.month}>"

    def __add__(self, other):
        if isinstance(other, datetime):
            import calendar
            year = other.year + self.n
            last_day = calendar.monthrange(year, self.month)[1]
            return datetime(year, self.month, last_day)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)


class YearStart:
    """年初偏移 (到自然年的第一天)。"""

    def __init__(self, n: int = 1, month: int = 1):
        self.n = int(n)
        self.month = month

    def __repr__(self) -> str:
        return f"<YearStart: n={self.n}, month={self.month}>"

    def __add__(self, other):
        if isinstance(other, datetime):
            year = other.year + self.n
            return datetime(year, self.month, 1)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)