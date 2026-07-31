import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

start_time = time.time()
def print_series(num, s):
    print("++++++++++++++++++++", num)
    print(s)
    print()


s = pd.Series([1, 3, 5, np.nan, 6, 8])
print_series(1, s)

dates = pd.date_range("20130101", periods=6)
print_series(2, dates)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
print_series(3, df)

df2 = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20130102"),
        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo",
    }
)
print_series(4, df2)

print_series(5, df2.dtypes)

# print_series(6, df2.<TAB>)  # noqa: E225, E999

print_series(7, df.head())

print_series(8, df.tail(3))
print_series(9, df.index)
print_series(10, df.columns)
print_series(11, df.to_numpy())
print_series(14, df2.dtypes)
print_series(15, df.describe())
print_series(16, df.T)
print_series(17, df.sort_index(axis=1, ascending=False))
print_series(18, df.sort_values(by="B"))
print_series(19, df["A"])
print_series(20, df.A)
print_series(21, df[["B", "A"]])
print_series(22, df[0:3])
print_series(23, df["20130102":"20130104"])
print_series(24, df.loc[dates[0]])
print_series(25, df.loc[:, ["A", "B"]])

print_series(26, df.loc["20130102":"20130104", ["A", "B"]])
print_series(27, df.loc[dates[0], "A"])
print_series(28, df.at[dates[0], "A"])

print_series(29, df.iloc[3])
print_series(30, df.iloc[3:5, 0:2])
print_series(31, df.iloc[[1, 2, 4], [0, 2]])
print_series(32, df.iloc[1:3, :])
print_series(33, df.iloc[:, 1:3])
print_series(34, df.iloc[1, 1])
print_series(35, df.iat[1, 1])

print_series(36, df[df["A"] > 0])
print_series(37, df[df > 0])
df2 = df.copy()
df2["E"] = ["one", "one", "two", "three", "four", "three"]
print_series(38, df2)
print_series(39, df2[df2["E"].isin(["two", "four"])])

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20130102", periods=6))

print_series(40, s1)

df.at[dates[0], "A"] = 0
df.iat[0, 1] = 0
df.loc[:, "D"] = np.array([5] * len(df))
print_series(41, df)

df2 = df.copy()
df2[df2 > 0] = -df2
print_series(42, df2)
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ["E"])
df1.loc[dates[0]:dates[1], "E"] = 1
print_series(43, df1)
print_series(44, df1.dropna(how="any", inplace=True))
print_series(45, df1.fillna(value=5))
print_series(46, pd.isna(df1))
print_series(47, df.mean())
print_series(48, df.mean(axis=1))
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
print_series(49, s)
print_series(50, df.sub(s, axis="index"))
print_series(51, df.agg(lambda x: np.mean(x) * 5.6))
print_series(52, df.transform(lambda x: x * 101.2))
s = pd.Series(np.random.randint(0, 7, size=10))
print_series(53, s)
print_series(54, s.value_counts())
s = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"])

print_series(55, s.str.lower())
df = pd.DataFrame(np.random.randn(10, 4))
print_series(56, df)
pieces = [df[:3], df[3:7], df[7:]]
print_series(57, pd.concat(pieces))
left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})
print_series(58, left)
print_series(59, right)
print_series(60, pd.merge(left, right, on="key"))

df = pd.DataFrame(
    {
        "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
        "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
        "C": np.random.randn(8),
        "D": np.random.randn(8),
    }
)
print_series(61, df)
print_series(62, df.groupby("A")[["C", "D"]].sum())

print_series(63, df.groupby(["A", "B"]).sum())
arrays = [["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
          ["one", "two", "one", "two", "one", "two", "one", "two"],
          ]
index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=["A", "B"])
df2 = df[:4]
print_series(64, df2)
stacked = df2.stack()
print_series(65, stacked)
print_series(66, stacked.unstack())
print_series(67, stacked.unstack(1))
print_series(68, stacked.unstack(0))
df = pd.DataFrame(
    {
        "A": ["one", "one", "two", "three"] * 3,
        "B": ["A", "B", "C"] * 4,
        "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
        "D": np.random.randn(12),
        "E": np.random.randn(12),
    }
)
print_series(69, df)
print_series(70, pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"]))
rng = pd.date_range("1/1/2012", periods=100, freq="s")
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
print_series(71, ts.resample("5Min").sum())
rng = pd.date_range("3/6/2012 00:00", periods=5, freq="D")
ts = pd.Series(np.random.randn(len(rng)), rng)
print_series(72, ts)
ts_utc = ts.tz_localize("UTC")
print_series(73, ts_utc)
print_series(74, ts_utc.tz_convert("US/Eastern"))
print_series(75, rng)
print_series(76, rng + pd.offsets.BusinessDay(5))
df = pd.DataFrame(
    {"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]}
)
df["grade"] = df["raw_grade"].astype("category")
print_series(77, df["grade"])
new_categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.rename_categories(new_categories)
df["grade"] = df["grade"].cat.set_categories(
    ["very bad", "bad", "medium", "good", "very good"]
)
print_series(78, df["grade"])
print_series(79, df.sort_values(by="grade"))
print_series(80, df.groupby("grade", observed=False).size())

fig = plt.figure(figsize=(10, 6))
plt.close("all")
ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))

ts = ts.cumsum()
ts.plot()
plt.savefig("./testing/pandas/test1.png")
df = pd.DataFrame(
    np.random.randn(1000, 4), index=ts.index, columns=["A", "B", "C", "D"]
)
df = df.cumsum()
plt.figure()
df.plot()
plt.legend(loc='best')
plt.savefig("./testing/pandas/test2.png")
df = pd.DataFrame(np.random.randint(0, 5, (10, 5)))
df.to_csv("./testing/pandas/test3.csv")
print_series(82, pd.read_csv("./testing/pandas/test3.csv"))

df.to_parquet("./testing/pandas/test4.parquet")
print_series(83, pd.read_parquet("./testing/pandas/test4.parquet"))
df.to_excel("./testing/pandas/test5.xlsx", sheet_name="Sheet1")
print_series(84, pd.read_excel("./testing/pandas/test5.xlsx", "Sheet1", index_col=None, na_values=["NA"]))

end_time = time.time()
print("Time cost:", end_time - start_time)
