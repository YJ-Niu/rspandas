import time
start_time = time.time()
from functools import partial  # noqa: E402

import rsnumpy as np  # noqa: E402
from collections import namedtuple  # noqa: E402
from dataclasses import make_dataclass  # noqa: E402
import rspandas as pd  # noqa: E402


def print_series(num, s):
    print("++++++++++++++++++++", num)
    print(s)
    print()


s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
print_series(1, s)
print_series(2, s.index)
print_series(3, pd.Series(np.random.randn(5)))
d = {"b": 1, "a": 0, "c": 2}
print_series(4, pd.Series(d))
d = {"a": 0.0, "b": 1.0, "c": 2.0}
print_series(5, pd.Series(d))
print_series(6, pd.Series(d, index=["b", "c", "d", "a"]))
print_series(7, pd.Series(5.0, index=["a", "b", "c", "d", "e"]))
print_series(8, s.iloc[0])
print_series(9, s.iloc[:3])
print_series(10, s[s > s.median()])
print_series(11, s.iloc[[4, 3, 1]])
print_series(12, np.exp(s))
print_series(13, s.dtype)
print_series(14, s.array)
print_series(15, s.to_numpy())
print_series(16, s["a"])
s["e"] = 12.0
print_series(17, s)
print_series(18, "e" in s)
print_series(19, "f" in s)
print_series(20, s.get("f"))
print_series(21, s.get("f", np.nan))
print_series(22, s + s)
print_series(23, np.exp(s))
print_series(24, s.iloc[1:] + s.iloc[:-1])
s = pd.Series(np.random.randn(5), name="something")
print_series(25, s)
print_series(26, s.name)
s2 = s.rename("different")
print_series(27, s2.name)
d = {
    "one": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
    "two": pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"]),
}
df = pd.DataFrame(d)
print_series(28, df)
print_series(29, pd.DataFrame(d, index=["d", "b", "a"]))
print_series(30, pd.DataFrame(d, index=["d", "b", "a"], columns=["two", "three"]))
print_series(31, df.index)
print_series(32, df.columns)
d = {"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]}
print_series(33, pd.DataFrame(d))
print_series(34, pd.DataFrame(d, index=["a", "b", "c", "d"]))
data = np.zeros((2,), dtype=[("A", "i4"), ("B", "f4"), ("C", "S10")])
data[:] = [(1, 2.0, "Hello"), (2, 3.0, "World")]
print_series(35, pd.DataFrame(data))
print_series(36, pd.DataFrame(data, index=["first", "second"]))
print_series(37, pd.DataFrame(data, columns=["C", "A", "B"]))
data2 = [{"a": 1, "b": 2}, {"a": 5, "b": 10, "c": 20}]
print_series(38, pd.DataFrame(data2))
print_series(39, pd.DataFrame(data2, index=["first", "second"]))
print_series(40, pd.DataFrame(data2, columns=["a", "b"]))
print_series(41, pd.DataFrame(
    {
        ("a", "b"): {("A", "B"): 1, ("A", "C"): 2},
        ("a", "a"): {("A", "C"): 3, ("A", "B"): 4},
        ("a", "c"): {("A", "B"): 5, ("A", "C"): 6},
        ("b", "a"): {("A", "C"): 7, ("A", "B"): 8},
        ("b", "b"): {("A", "D"): 9, ("A", "B"): 10},
    }
))
ser = pd.Series(range(3), index=list("abc"), name="ser")
print_series(42, pd.DataFrame(ser))
Point = namedtuple("Point", "x y")
print_series(43, pd.DataFrame([Point(0, 0), Point(0, 3), (2, 3)]))
Point3D = namedtuple("Point3D", "x y z")
print_series(44, pd.DataFrame([Point3D(0, 0, 0), Point3D(0, 3, 5), Point(2, 3)]))
Point = make_dataclass("Point", [("x", int), ("y", int)])
print_series(45, pd.DataFrame([Point(0, 0), Point(0, 3), Point(2, 3)]))
print_series(46, pd.DataFrame.from_dict(dict([("A", [1, 2, 3]), ("B", [4, 5, 6])])))
print_series(47, pd.DataFrame.from_dict(
    dict([("A", [1, 2, 3]), ("B", [4, 5, 6])]),
    orient="index",
    columns=["one", "two", "three"],
))
print_series(48, data)
print_series(49, pd.DataFrame.from_records(data, index="C"))
print_series(50, df["one"])
df["three"] = df["one"] * df["two"]
print_series(51, df)
del df["two"]
three = df.pop("three")
print_series(52, df)
df["foo"] = "bar"
print_series(53, df)
df["one_trunc"] = df["one"][:2]
print_series(54, df)
df.insert(1, "bar", df["one"])
print_series(55, df)
iris = pd.read_csv("./testing/data/iris.data")
print_series(56, iris.head())
print_series(57, iris.assign(sepal_ratio=iris["SepalWidth"] / iris["SepalLength"]).head())
print_series(58, iris.assign(sepal_ratio=lambda x: (x["SepalWidth"] / x["SepalLength"])).head())
print_series(59, iris.assign(sepal_ratio=pd.col("SepalWidth") / pd.col("SepalLength")).head())
(
    iris.query("SepalLength > 5")
    .assign(
        SepalRatio=lambda x: x.SepalWidth / x.SepalLength,
        PetalRatio=lambda x: x.PetalWidth / x.PetalLength,
    )
    .plot(kind="scatter", x="SepalRatio", y="PetalRatio")
)
dfa = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
print_series(60, dfa.assign(C=lambda x: x["A"] + x["B"], D=lambda x: x["A"] + x["C"]))
print_series(61, df.loc["b"])
print_series(62, df.iloc[2])
df = pd.DataFrame(np.random.randn(10, 4), columns=["A", "B", "C", "D"])
df2 = pd.DataFrame(np.random.randn(7, 3), columns=["A", "B", "C"])
print_series(63, df + df2)
print_series(64, df - df.iloc[0])
print_series(65, df * 5 + 2)
print_series(66, 1 / df)
print_series(67, df ** 4)
df1 = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]}, dtype=bool)
df2 = pd.DataFrame({"a": [0, 1, 1], "b": [1, 1, 0]}, dtype=bool)
print_series(68, df1 & df2)
print_series(69, df1 | df2)
print_series(70, df1 ^ df2)
print_series(71, -df1)
print_series(72, df[:5].T)
print_series(73, np.exp(df))
print_series(74, np.asarray(df))
ser = pd.Series([1, 2, 3, 4])
print_series(75, np.exp(ser))
ser1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
ser2 = pd.Series([1, 3, 5], index=["b", "a", "c"])
print_series(76, ser1)
print_series(77, ser2)
print_series(78, np.remainder(ser1, ser2))
ser3 = pd.Series([2, 4, 6], index=["b", "c", "d"])
print_series(79, ser3)
print_series(80, np.remainder(ser1, ser3))
ser = pd.Series([1, 2, 3])
print_series(81, ser)
idx = pd.Index([4, 5, 6])
print_series(82, np.maximum(ser, idx))
baseball = pd.read_csv("./testing/data/baseball.csv")
print_series(83, baseball)
print_series(84, baseball.info())
print_series(85, baseball.iloc[-20:, :12].to_string())
print_series(86, pd.DataFrame(np.random.randn(3, 12)))
pd.set_option("display.width", 40)
print_series(87, pd.DataFrame(np.random.randn(3, 12)))
datafile = {
    "filename": ["filename_01", "filename_02"],
    "path": [
        "./testing/rspandas/filename_01",
        "./testing/rspandas/filename_02",
    ],
}
pd.set_option("display.max_colwidth", 30)
print_series(88, pd.DataFrame(datafile))
pd.set_option("display.max_colwidth", 100)
print_series(89, pd.DataFrame(datafile))
df = pd.DataFrame({"foo1": np.random.randn(5), "foo2": np.random.randn(5)})
print_series(90, df)
print_series(91, df.foo1)
index = pd.date_range("1/1/2000", periods=8)
s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=["A", "B", "C"])
long_series = pd.Series(np.random.randn(1000))
print_series(92, long_series.head())
print_series(93, long_series.tail(3))
print_series(94, df[:2])
df.columns = [x.lower() for x in df.columns]
print_series(95, df)
print_series(96, s.array)
print_series(97, s.index.array)
print_series(98, s.to_numpy())
print_series(99, np.asarray(s))
ser = pd.Series(pd.date_range("2000", periods=2, tz="CET"))
print_series(100, ser.to_numpy(dtype=object))
print_series(101, ser.to_numpy(dtype="datetime64[ns]"))
print_series(102, df.to_numpy())
pd.set_option("compute.use_bottleneck", False)
pd.set_option("compute.use_numexpr", False)
df = pd.DataFrame(
    {
        "one": pd.Series(np.random.randn(3), index=["a", "b", "c"]),
        "two": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
        "three": pd.Series(np.random.randn(3), index=["b", "c", "d"]),
    }
)

print_series(103, df)
row = df.iloc[1]
column = df["two"]
print_series(104, df.sub(row, axis="columns"))
print_series(105, df.sub(row, axis=1))
print_series(106, df.sub(column, axis="index"))
print_series(107, df.sub(column, axis=0))
dfmi = df.copy()
dfmi.index = pd.MultiIndex.from_tuples(
    [(1, "a"), (1, "b"), (1, "c"), (2, "a")], names=["first", "second"]
)
print_series(108, dfmi.sub(column, axis=0, level="second"))
s = pd.Series(np.arange(10))
print_series(109, s)
div, rem = divmod(s, 3)
print_series(110, div)
print_series(111, rem)
idx = pd.Index(np.arange(10))
print_series(112, idx)
div, rem = divmod(idx, 3)
print_series(113, div)
print_series(114, rem)
div, rem = divmod(s, [2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
print_series(115, div)
print_series(116, rem)
df2 = df.copy()
df2.loc["a", "three"] = 1.0
print_series(117, df)
print_series(118, df2)
print_series(119, df + df2)
print_series(120, df.add(df2, fill_value=0))
print_series(121, df.gt(df2))
print_series(122, df.ne(df2))
print_series(123, (df > 0).all())
print_series(124, (df > 0).any())
print_series(125, (df > 0).any().any())
print_series(126, df.empty)
print_series(127, pd.DataFrame(columns=list("ABC")).empty)
print_series(128, df + df == df * 2)
print_series(129, (df + df == df * 2).all())
print_series(130, np.nan == np.nan)
print_series(131, (df + df).equals(df * 2))
df1 = pd.DataFrame({"col": ["foo", 0, np.nan]})
df2 = pd.DataFrame({"col": [np.nan, 0, "foo"]}, index=[2, 1, 0])
print_series(132, df1.equals(df2))
print_series(133, df1.equals(df2.sort_index()))
print_series(134, pd.Series(["foo", "bar", "baz"]) == "foo")
print_series(135, pd.Series(["foo", "bar", "baz"]) == pd.Index(["foo", "bar", "qux"]))
print_series(136, pd.Series(["foo", "bar", "baz"]) == np.array(["foo", "bar", "qux"]))
df1 = pd.DataFrame(
    {"A": [1.0, np.nan, 3.0, 5.0, np.nan], "B": [np.nan, 2.0, 3.0, np.nan, 6.0]}
)


df2 = pd.DataFrame(
    {
        "A": [5.0, 2.0, 4.0, np.nan, 3.0, 7.0],
        "B": [np.nan, np.nan, 3.0, 4.0, 6.0, 8.0],
    }
)
print_series(137, df1)
print_series(138, df2)
print_series(139, df1.combine_first(df2))

def combiner(x, y):
    return np.where(pd.isna(x), y, x)


print_series(140, df1.combine(df2, combiner))
print_series(141, df)
print_series(142, df.mean(axis=0))
print_series(143, df.mean(axis=1))
print_series(144, df.sum(axis=0, skipna=False))
print_series(145, df.sum(axis=1, skipna=True))
ts_stand = (df - df.mean()) / df.std()
print_series(146, ts_stand.std())
xs_stand = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
print_series(147, xs_stand.std(axis=1))
print_series(148, df.cumsum())
print_series(149, np.mean(df["one"]))
print_series(150, np.mean(df["one"].to_numpy()))
series = pd.Series(np.random.randn(500))
series[20:500] = np.nan
series[10:20] = 5
print_series(151, series.nunique())
series = pd.Series(np.random.randn(1000))
series[::2] = np.nan
print_series(152, series.describe())
frame = pd.DataFrame(np.random.randn(1000, 5), columns=["a", "b", "c", "d", "e"])
frame.iloc[::2] = np.nan
print_series(153, frame.describe())
print_series(154, series.describe(percentiles=[0.05, 0.25, 0.75, 0.95]))
s = pd.Series(["a", "a", "b", "b", "a", "a", np.nan, "c", "d", "a"])
print_series(155, s.describe())
frame = pd.DataFrame({"a": ["Yes", "Yes", "No", "No"], "b": range(4)})
print_series(156, frame.describe())
print_series(157, frame.describe(include=["str"]))
print_series(158, frame.describe(include=["number"]))
print_series(159, frame.describe(include="all"))
s1 = pd.Series(np.random.randn(5))
print_series(160, s1)
print_series(161, (s1.idxmin(), s1.idxmax()))
df1 = pd.DataFrame(np.random.randn(5, 3), columns=["A", "B", "C"])
print_series(162, df1)
print_series(163, df1.idxmin(axis=0))
print_series(164, df1.idxmax(axis=1))
df3 = pd.DataFrame([2, 1, 1, 3, np.nan], columns=["A"], index=list("edcba"))
print_series(165, df3)
print_series(166, df3["A"].idxmin())
data = np.random.randint(0, 7, size=50)
print_series(167, data)
s = pd.Series(data)
print_series(168, s.value_counts())
data = {"a": [1, 2, 3, 4], "b": ["x", "x", "y", "y"]}
frame = pd.DataFrame(data)
print_series(169, frame.value_counts())
s5 = pd.Series([1, 1, 3, 3, 3, 5, 5, 7, 7, 7])
print_series(170, s5.mode())
df5 = pd.DataFrame(
    {
        "A": np.random.randint(0, 7, size=50),
        "B": np.random.randint(-10, 15, size=50),
    }
)
print_series(171, df5)
print_series(172, df5.mode())
arr = np.random.randn(20)
factor = pd.cut(arr, 4)
print_series(173, factor)
arr = np.random.randn(30)
factor = pd.qcut(arr, [0, 0.25, 0.5, 0.75, 1])
print_series(174, factor)
def extract_city_name(df):
    """
    Chicago, IL -> Chicago for city_name column
    """
    df["city_name"] = df["city_and_code"].str.split(",").str.get(0)
    return df


def add_country_name(df, country_name=None):
    """
    Chicago -> Chicago-US for city_name column
    """
    col = "city_name"
    df["city_and_country"] = df[col] + country_name
    return df


df_p = pd.DataFrame({"city_and_code": ["Chicago, IL"]})
print_series(175, add_country_name(extract_city_name(df_p), country_name="US"))
print_series(176, df_p.pipe(extract_city_name).pipe(add_country_name, country_name="US"))
print_series(177, df.apply(lambda x: np.mean(x)))
print_series(178, df.apply(lambda x: np.mean(x), axis=1))
print_series(179, df.apply(lambda x: x.max() - x.min()))
print_series(180, df.apply(np.cumsum))
print_series(181, df.apply(np.exp))
print_series(182, df.apply("mean"))
print_series(183, df.apply("mean", axis=1))
tsdf = pd.DataFrame(
    np.random.randn(1000, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=1000),
)
print_series(184, tsdf.apply(lambda x: x.idxmax()))

def subtract_and_divide(x, sub, divide=1):
    return (x - sub) / divide


df_udf = pd.DataFrame(np.ones((2, 2)))
print_series(185, df_udf.apply(subtract_and_divide, args=(5,), divide=3))
tsdf = pd.DataFrame(
    np.random.randn(10, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=10),
)
tsdf.iloc[3:7] = np.nan
print_series(186, tsdf)
print_series(187, tsdf.apply(pd.Series.interpolate))
tsdf = pd.DataFrame(
    np.random.randn(10, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=10),
)
tsdf.iloc[3:7] = np.nan
print_series(188, tsdf)
print_series(189, tsdf.agg(lambda x: np.sum(x)))
print_series(190, tsdf.agg("sum"))
print_series(191, tsdf.sum())
print_series(192, tsdf["A"].agg("sum"))
print_series(193, tsdf.agg(["sum"]))
print_series(194, tsdf.agg(["sum", "mean"]))
print_series(195, tsdf["A"].agg(["sum", "mean"]))
print_series(196, tsdf["A"].agg(["sum", lambda x: x.mean()]))
def mymean(x):
    return x.mean()


print_series(197, tsdf["A"].agg(["sum", mymean]))
print_series(198, tsdf.agg({"A": "mean", "B": "sum"}))
print_series(199, tsdf.agg({"A": ["mean", "min"], "B": "sum"}))
q_25 = partial(pd.Series.quantile, q=0.25)
q_25.__name__ = "25%"
q_75 = partial(pd.Series.quantile, q=0.75)
q_75.__name__ = "75%"
print_series(200, tsdf.agg(["count", "mean", "std", "min", q_25, "median", q_75, "max"]))
tsdf = pd.DataFrame(
    np.random.randn(10, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=10),
)
tsdf.iloc[3:7] = np.nan
print_series(201, tsdf)
print_series(202, tsdf.transform(np.abs))
print_series(203, tsdf.transform("abs"))
print_series(204, tsdf.transform(lambda x: x.abs()))
print_series(205, np.abs(tsdf))
print_series(206, tsdf["A"].transform(np.abs))
print_series(207, tsdf.transform([np.abs, lambda x: x + 1]))
print_series(208, tsdf["A"].transform([np.abs, lambda x: x + 1]))
print_series(209, tsdf.transform({"A": np.abs, "B": lambda x: x + 1}))
print_series(210, tsdf.transform({"A": np.abs, "B": [lambda x: x + 1, "sqrt"]}))
df4 = df.copy()
print_series(211, df4)
def f(x):
    return len(str(x))


print_series(212, df4["one"].map(f))
print_series(213, df4.map(f))
end_time = time.time()
print("end_time - start_time:", end_time - start_time)
