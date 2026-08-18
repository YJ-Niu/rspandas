//! DataFrame 的 PyO3 绑定：将 Rust 端 DataFrame 方法暴露给 Python。
//!
//! 该模块包含单个 `#[pymethods] impl PyDataFrame` 块，涵盖构造、属性、子集、
//! 缺失值、排序、行级缺失检测、全列填充、列并行批量聚合、显示辅助、
//! 合并、分组聚合、透视/逆透视、stack/unstack、简单查询等全部 Python 可见方法。
//!
//! 所有计算密集型方法均通过 `py.detach` 释放 GIL 后再委托给 Rust 端 :struct:`DataFrame`。

use pyo3::IntoPyObject;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

use crate::core::dataframe::{DataFrame, FillValue, PyDataFrame};
use crate::core::dtype::ColumnData;
use crate::core::series::PySeries;

#[pymethods]
impl PyDataFrame {
    /// 构造: 接受 columns (list[str]) 和 series (list[_Series])
    #[new]
    fn new(columns: Vec<String>, series: Vec<PySeries>) -> PyResult<Self> {
        let data: Vec<crate::core::series::Series> = series.into_iter().map(|s| s.inner).collect();
        let inner = DataFrame::from_series(columns, data)?;
        Ok(PyDataFrame { inner })
    }

    // ---------- 属性 ----------

    #[getter]
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }
    #[getter]
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }
    #[getter]
    fn size(&self) -> usize {
        self.inner.nrows() * self.inner.ncols()
    }
    #[getter]
    fn empty(&self) -> bool {
        self.inner.nrows() == 0
    }

    #[getter]
    fn columns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.inner.columns.iter().map(|s| s.as_str()))
    }

    #[getter]
    fn dtypes<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let d = PyDict::new(py);
        for (name, dt) in self.inner.dtypes() {
            d.set_item(name, dt).unwrap();
        }
        d
    }

    // ---------- 子集 ----------

    /// 按列名取列 -> _Series
    fn get_column(&self, name: &str) -> PyResult<PySeries> {
        match self.inner.get_column(name) {
            Some(s) => Ok(PySeries { inner: s.clone() }),
            None => Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "column not found: {}",
                name
            ))),
        }
    }

    /// 按索引取列 -> _Series
    fn get_column_at(&self, idx: usize) -> PyResult<PySeries> {
        match self.inner.get_column_at(idx) {
            Some(s) => Ok(PySeries { inner: s.clone() }),
            None => Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "column index out of range: {}",
                idx
            ))),
        }
    }

    /// 列名 -> 索引
    fn column_index(&self, name: &str) -> Option<usize> {
        self.inner.get_column_index(name)
    }

    fn head(&self, py: Python<'_>, n: usize) -> Self {
        let inner = py.detach(|| self.inner.head(n));
        PyDataFrame { inner }
    }
    fn tail(&self, py: Python<'_>, n: usize) -> Self {
        let inner = py.detach(|| self.inner.tail(n));
        PyDataFrame { inner }
    }
    fn filter_rows(&self, py: Python<'_>, mask: Vec<bool>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.filter_rows(&mask))?;
        Ok(PyDataFrame { inner })
    }

    // ---------- 缺失值 ----------

    fn dropna(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.dropna());
        PyDataFrame { inner }
    }

    /// fillna: 接受 dict {col_name: value}，只填充指定列
    fn fillna(&self, py: Python<'_>, fill_dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut converted = std::collections::HashMap::new();
        for (key, val) in fill_dict.iter() {
            let col: String = key.extract()?;
            // 优先尝试 bool，再 int，再 float，最后 string
            if let Ok(b) = val.extract::<bool>() {
                converted.insert(col, FillValue::Bool(b));
            } else if let Ok(i) = val.extract::<i64>() {
                converted.insert(col, FillValue::Int(i));
            } else if let Ok(f) = val.extract::<f64>() {
                converted.insert(col, FillValue::Float(f));
            } else if let Ok(s) = val.extract::<String>() {
                converted.insert(col, FillValue::String(s));
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "unsupported fill value type for column '{}'",
                    col
                )));
            }
        }
        let inner = py.detach(|| self.inner.fillna_rows(&converted))?;
        Ok(PyDataFrame { inner })
    }

    // ---------- 排序 ----------

    /// 按指定列的值排序 (by 为列索引列表，取第一个列排序)
    fn sort_values(&self, py: Python<'_>, by: Vec<usize>, ascending: bool) -> Self {
        let inner = py.detach(|| self.inner.sort_values(&by, ascending));
        PyDataFrame { inner }
    }

    /// 按索引排序 (ascending=true 保持原顺序，false 反转)
    fn sort_index(&self, py: Python<'_>, ascending: bool) -> Self {
        let inner = py.detach(|| self.inner.sort_index(ascending));
        PyDataFrame { inner }
    }

    // ---------- 行级缺失值检测 ----------

    /// 返回每行是否有缺失值
    fn isnull_rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mask = py.detach(|| self.inner.isnull_rows());
        PyList::new(py, mask.iter().copied())
    }

    /// 返回每行是否全部非缺失
    fn notnull_rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mask = py.detach(|| self.inner.notnull_rows());
        PyList::new(py, mask.iter().copied())
    }

    // ---------- 全列填充 ----------

    /// 填充所有列的缺失值 (f64，仅对 Float64 列生效)
    fn fillna_all_f64(&self, py: Python<'_>, v: f64) -> Self {
        let inner = py.detach(|| self.inner.fillna_all_f64(v));
        PyDataFrame { inner }
    }

    /// 填充所有列的缺失值 (i64，仅对 Int64 列生效)
    fn fillna_all_i64(&self, py: Python<'_>, v: i64) -> Self {
        let inner = py.detach(|| self.inner.fillna_all_i64(v));
        PyDataFrame { inner }
    }

    /// 填充所有列的缺失值 (string，仅对 Object 列生效)
    fn fillna_all_string(&self, py: Python<'_>, v: &str) -> Self {
        let v_owned = v.to_string();
        let inner = py.detach(|| self.inner.fillna_all_string(&v_owned));
        PyDataFrame { inner }
    }

    // ---------- 列并行批量方法（Python for 循环 → Rust rayon 并行） ----------

    /// 所有列并行 ffill（一次调用替代 Python 逐列 for 循环）
    fn par_ffill_all(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.par_ffill_all());
        PyDataFrame { inner }
    }

    /// 所有列并行 bfill
    fn par_bfill_all(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.par_bfill_all());
        PyDataFrame { inner }
    }

    /// 所有列并行 sum → list[float|None]（按列顺序）
    fn par_sum_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let result = py.detach(|| self.inner.par_sum_all());
        let list = PyList::empty(py);
        for r in result {
            match r {
                Some(v) => list.append(v).unwrap(),
                None => list.append(py.None()).unwrap(),
            }
        }
        list
    }

    /// 所有列并行 mean
    fn par_mean_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let result = py.detach(|| self.inner.par_mean_all());
        let list = PyList::empty(py);
        for r in result {
            match r {
                Some(v) => list.append(v).unwrap(),
                None => list.append(py.None()).unwrap(),
            }
        }
        list
    }

    /// 所有列并行 std（ddof 自由度）
    fn par_std_all<'py>(&self, py: Python<'py>, ddof: usize) -> Bound<'py, PyList> {
        let result = py.detach(move || self.inner.par_std_all(ddof));
        let list = PyList::empty(py);
        for r in result {
            match r {
                Some(v) => list.append(v).unwrap(),
                None => list.append(py.None()).unwrap(),
            }
        }
        list
    }

    /// 所有列并行 min
    fn par_min_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let result = py.detach(|| self.inner.par_min_all());
        let list = PyList::empty(py);
        for r in result {
            match r {
                Some(v) => list.append(v).unwrap(),
                None => list.append(py.None()).unwrap(),
            }
        }
        list
    }

    /// 所有列并行 max
    fn par_max_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let result = py.detach(|| self.inner.par_max_all());
        let list = PyList::empty(py);
        for r in result {
            match r {
                Some(v) => list.append(v).unwrap(),
                None => list.append(py.None()).unwrap(),
            }
        }
        list
    }

    /// 所有列并行 count（非空值数）
    fn par_count_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let result = py.detach(|| self.inner.par_count_all());
        PyList::new(py, result).unwrap()
    }

    /// 所有列并行 quantile
    fn par_quantile_all<'py>(&self, py: Python<'py>, q: f64) -> Bound<'py, PyList> {
        let result = py.detach(move || self.inner.par_quantile_all(q));
        let list = PyList::empty(py);
        for r in result {
            match r {
                Some(v) => list.append(v).unwrap(),
                None => list.append(py.None()).unwrap(),
            }
        }
        list
    }

    /// 所有列并行批量聚合（返回每列的聚合结果二维列表，行=列，列=aggs顺序）
    fn par_batch_agg<'py>(&self, py: Python<'py>, aggs: Vec<String>) -> Bound<'py, PyList> {
        let result = py.detach(move || self.inner.par_batch_agg(&aggs));
        let outer = PyList::empty(py);
        for col_r in result {
            let inner_list = PyList::empty(py);
            for r in col_r {
                match r {
                    Some(v) => inner_list.append(v).unwrap(),
                    None => inner_list.append(py.None()).unwrap(),
                }
            }
            outer.append(inner_list).unwrap();
        }
        outer
    }

    /// 所有列并行 any → list[bool|None]（按列顺序）
    fn par_any_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let result = py.detach(|| self.inner.par_any_all());
        let list = PyList::empty(py);
        for r in result {
            match r {
                Some(v) => list.append(v).unwrap(),
                None => list.append(py.None()).unwrap(),
            }
        }
        list
    }

    /// 所有列并行 all → list[bool|None]
    fn par_all_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let result = py.detach(|| self.inner.par_all_all());
        let list = PyList::empty(py);
        for r in result {
            match r {
                Some(v) => list.append(v).unwrap(),
                None => list.append(py.None()).unwrap(),
            }
        }
        list
    }

    /// 所有列并行 nunique → list[int]（按列顺序）
    fn par_nunique_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let result = py.detach(|| self.inner.par_nunique_all());
        PyList::new(py, result).unwrap()
    }

    /// 所有列并行 isnull → 返回 bool DataFrame（每列变为 bool 类型）
    fn par_isnull_all(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.par_isnull_all());
        PyDataFrame { inner }
    }

    /// 所有列并行 notnull → 返回 bool DataFrame
    fn par_notnull_all(&self, py: Python<'_>) -> Self {
        let inner = py.detach(|| self.inner.par_notnull_all());
        PyDataFrame { inner }
    }

    // ---------- 显示辅助 ----------

    /// 逐行构造 dict 列表 (用于 Python 端显示)
    fn to_rows<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let rows = PyList::empty(py);
        let nrows = self.inner.nrows();
        for i in 0..nrows {
            let row = PyDict::new(py);
            for (col_name, series) in self.inner.columns.iter().zip(self.inner.data.iter()) {
                let val: pyo3::Py<pyo3::PyAny> = match &series.data {
                    ColumnData::Int(v) => match v.get(i) {
                        Some(Some(n)) => (*n).into_pyobject(py).unwrap().into_any().unbind(),
                        _ => py.None(),
                    },
                    ColumnData::Float(v) => match v.get(i) {
                        Some(Some(n)) => (*n).into_pyobject(py).unwrap().into_any().unbind(),
                        _ => py.None(),
                    },
                    ColumnData::Bool(v) => match v.get(i) {
                        Some(Some(n)) => (*n).into_pyobject(py).unwrap().as_any().clone().unbind(),
                        _ => py.None(),
                    },
                    ColumnData::String(v) => match v.get(i) {
                        Some(Some(s)) => s.clone().into_pyobject(py).unwrap().into_any().unbind(),
                        _ => py.None(),
                    },
                    ColumnData::Categorical(c) => match c.codes.get(i) {
                        Some(Some(code_idx)) => {
                            let cat_str = c
                                .categories
                                .get(*code_idx as usize)
                                .cloned()
                                .unwrap_or_else(|| "NaN".to_string());
                            cat_str.into_pyobject(py).unwrap().into_any().unbind()
                        }
                        _ => py.None(),
                    },
                };
                row.set_item(col_name, val).unwrap();
            }
            rows.append(row).unwrap();
        }
        rows
    }

    /// 每列的 string 列表 (用于显示)
    fn columns_to_string<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let d = PyDict::new(py);
        // 释放 GIL 进行并行字符串转换
        let pairs: Vec<(String, Vec<String>)> = py.detach(|| {
            self.inner
                .columns
                .par_iter()
                .zip(self.inner.data.par_iter())
                .map(|(col_name, series)| (col_name.clone(), series.to_string_vec()))
                .collect()
        });
        for (col_name, svec) in pairs {
            let pylist: Bound<'_, PyList> =
                PyList::new(py, svec.iter().map(|s| s.as_str())).unwrap();
            d.set_item(col_name, pylist).unwrap();
        }
        d
    }

    /// 各列 dtype 的 dict
    fn dtypes_dict<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        self.dtypes(py)
    }

    // ---------- 合并 ----------

    /// 基于键列的哈希连接
    /// how: "inner"/"left"/"right"/"outer"
    fn merge(
        &self,
        py: Python<'_>,
        right: &PyDataFrame,
        left_on: usize,
        right_on: usize,
        how: &str,
    ) -> Self {
        let inner = py.detach(|| self.inner.merge(&right.inner, left_on, right_on, how));
        PyDataFrame { inner }
    }

    // ---------- 分组聚合 ----------

    /// 按 by 列分组并对每列执行聚合
    /// agg: "sum"/"mean"/"count"/"min"/"max"
    /// 返回 (group_keys, aggregated_df)
    fn groupby_agg(&self, py: Python<'_>, by: usize, agg: &str) -> (Vec<String>, PyDataFrame) {
        let (keys, df) = py.detach(|| self.inner.groupby_agg(by, agg));
        (keys, PyDataFrame { inner: df })
    }

    // ---------- 透视与逆透视 ----------

    /// 透视表：按 index_col 分组，columns_col 的值作为新列名，values_col 聚合
    /// agg_func: "sum"/"mean"/"count"/"min"/"max"
    fn pivot(
        &self,
        py: Python<'_>,
        index_col: usize,
        columns_col: usize,
        values_col: usize,
        agg_func: &str,
    ) -> Self {
        let inner = py.detach(|| {
            self.inner
                .pivot(index_col, columns_col, values_col, agg_func)
        });
        PyDataFrame { inner }
    }

    /// 宽转长：将指定的值列转为 (variable, value) 两列
    fn melt(&self, py: Python<'_>, id_cols: Vec<usize>, value_cols: Vec<usize>) -> Self {
        let inner = py.detach(|| self.inner.melt(&id_cols, &value_cols));
        PyDataFrame { inner }
    }

    // ---------- stack / unstack ----------

    /// 将列堆叠为行
    fn stack(&self, py: Python<'_>, level: i64) -> Self {
        let inner = py.detach(|| self.inner.stack(level));
        PyDataFrame { inner }
    }

    /// unstack：将 variable/value 列透视为宽表
    fn unstack(&self, py: Python<'_>, index_col: usize, var_col: usize, value_col: usize) -> Self {
        let inner = py.detach(|| self.inner.unstack(index_col, var_col, value_col));
        PyDataFrame { inner }
    }

    // ---------- 简单查询（query 简化版） ----------

    /// 按列比较标量过滤行
    fn query_filter(&self, py: Python<'_>, col_idx: usize, op: &str, value: f64) -> Self {
        let op_owned = op.to_string();
        let inner = py.detach(|| self.inner.query_filter(col_idx, &op_owned, value));
        PyDataFrame { inner }
    }
}
