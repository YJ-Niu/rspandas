//! DataFrame: 列存储的多列数据结构 + PyO3 绑定

use pyo3::IntoPyObject;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

use super::dtype::{ColumnData, DType};
use super::series::{PySeries, Series};

#[derive(Debug, Clone)]
pub struct DataFrame {
    pub columns: Vec<String>,
    pub data: Vec<Series>,
}

impl DataFrame {
    pub fn new_empty() -> Self {
        Self {
            columns: Vec::new(),
            data: Vec::new(),
        }
    }

    pub fn from_series(columns: Vec<String>, data: Vec<Series>) -> PyResult<Self> {
        if columns.len() != data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "columns len {} != data len {}",
                columns.len(),
                data.len()
            )));
        }
        // 校验列名去重
        let mut seen = std::collections::HashSet::new();
        for c in &columns {
            if !seen.insert(c.clone()) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "duplicate column name: {}",
                    c
                )));
            }
        }
        // 校验每列长度一致
        if let Some(first) = data.first() {
            let n = first.len();
            for (i, s) in data.iter().enumerate() {
                if s.len() != n {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "column '{}' length {} != row length {}",
                        columns[i],
                        s.len(),
                        n
                    )));
                }
            }
        }
        Ok(Self { columns, data })
    }

    pub fn nrows(&self) -> usize {
        self.data.first().map(|s| s.len()).unwrap_or(0)
    }
    pub fn ncols(&self) -> usize {
        self.columns.len()
    }
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
    pub fn column_names(&self) -> &[String] {
        &self.columns
    }

    pub fn dtypes(&self) -> Vec<(&str, &'static str)> {
        self.data
            .iter()
            .zip(self.columns.iter())
            .map(|(s, c)| (c.as_str(), s.dtype_name()))
            .collect()
    }

    pub fn get_column(&self, name: &str) -> Option<&Series> {
        self.columns
            .iter()
            .position(|c| c == name)
            .and_then(|i| self.data.get(i))
    }

    pub fn get_column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }

    pub fn get_column_at(&self, idx: usize) -> Option<&Series> {
        self.data.get(idx)
    }

    pub fn head(&self, n: usize) -> DataFrame {
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.head(n)).collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }

    pub fn tail(&self, n: usize) -> DataFrame {
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.tail(n)).collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }

    pub fn filter_rows(&self, mask: &[bool]) -> PyResult<DataFrame> {
        if mask.len() != self.nrows() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "mask length {} != nrows {}",
                mask.len(),
                self.nrows()
            )));
        }
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.filter(mask)).collect();
        Ok(DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        })
    }

    /// 删除任意一列为 None 的行 (axis=0)
    pub fn dropna_rows(&self) -> DataFrame {
        if self.nrows() == 0 {
            return self.clone();
        }
        let nrows = self.nrows();
        // 并行计算每列的非空 mask，然后合并 (任意列 None 则整行删除)
        let keep: Vec<bool> = (0..nrows)
            .into_par_iter()
            .map(|i| {
                self.data.iter().all(|s| match &s.data {
                    super::dtype::ColumnData::Int(v) => v[i].is_some(),
                    super::dtype::ColumnData::Float(v) => v[i].is_some(),
                    super::dtype::ColumnData::Bool(v) => v[i].is_some(),
                    super::dtype::ColumnData::String(v) => v[i].is_some(),
                    super::dtype::ColumnData::Categorical(c) => c.codes[i].is_some(),
                })
            })
            .collect();
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.filter(&keep)).collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }

    /// 填充整个 DataFrame 中所有列的 None 值
    pub fn fillna_rows(
        &self,
        fill_dict: &std::collections::HashMap<String, FillValue>,
    ) -> PyResult<DataFrame> {
        let n_data: Vec<Series> = self
            .columns
            .par_iter()
            .zip(self.data.par_iter())
            .map(|(col, series)| {
                if let Some(v) = fill_dict.get(col) {
                    match (v, series.dtype()) {
                        (FillValue::Int(x), DType::Int64) => series.fillna_i64(*x),
                        (FillValue::Float(x), DType::Float64) => series.fillna_f64(*x),
                        (FillValue::Bool(x), DType::Bool) => series.fillna_bool(*x),
                        (FillValue::String(x), DType::Object) => series.fillna_string(x),
                        _ => series.clone(),
                    }
                } else {
                    series.clone()
                }
            })
            .collect();
        Ok(DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        })
    }

    /// 按指定列的值排序
    /// by 是列索引列表，按这些列的值排序行
    /// 使用第一列排序即可，多列排序取第一个
    pub fn sort_values(&self, by: &[usize], ascending: bool) -> DataFrame {
        // 空数据或空 by 直接返回克隆
        if by.is_empty() || self.nrows() == 0 || self.data.is_empty() {
            return self.clone();
        }
        let col_idx = by[0];
        let Some(sort_col) = self.data.get(col_idx) else {
            return self.clone();
        };
        let nrows = self.nrows();

        // 生成排序索引 (permutation): perm[i] 是排序后第 i 位对应的原行号
        let mut perm: Vec<usize> = (0..nrows).collect();

        // 根据列类型进行排序
        // ascending=true: 升序，None 放最后
        // ascending=false: 降序，None 放最前
        match &sort_col.data {
            ColumnData::Int(v) => {
                perm.sort_by(|&a, &b| {
                    cmp_opt_ord(
                        v.get(a).and_then(|x| x.as_ref()),
                        v.get(b).and_then(|x| x.as_ref()),
                        ascending,
                    )
                });
            }
            ColumnData::Float(v) => {
                perm.sort_by(|&a, &b| {
                    cmp_opt_f64(
                        v.get(a).and_then(|x| *x),
                        v.get(b).and_then(|x| *x),
                        ascending,
                    )
                });
            }
            ColumnData::Bool(v) => {
                perm.sort_by(|&a, &b| {
                    cmp_opt_ord(
                        v.get(a).and_then(|x| x.as_ref()),
                        v.get(b).and_then(|x| x.as_ref()),
                        ascending,
                    )
                });
            }
            ColumnData::String(v) => {
                perm.sort_by(|&a, &b| {
                    cmp_opt_str(
                        v.get(a).and_then(|x| x.as_deref()),
                        v.get(b).and_then(|x| x.as_deref()),
                        ascending,
                    )
                });
            }
            ColumnData::Categorical(c) => {
                // 对 categorical 按其字符串值排序
                perm.sort_by(|&a, &b| {
                    let sa = c.codes.get(a).and_then(|code| {
                        code.as_ref()
                            .and_then(|&idx| c.categories.get(idx as usize))
                            .map(|s| s.as_str())
                    });
                    let sb = c.codes.get(b).and_then(|code| {
                        code.as_ref()
                            .and_then(|&idx| c.categories.get(idx as usize))
                            .map(|s| s.as_str())
                    });
                    cmp_opt_str(sa, sb, ascending)
                });
            }
        }

        // 应用 permutation 到所有列 (并行)
        let n_data: Vec<Series> = self
            .data
            .par_iter()
            .map(|s| gather_series(s, &perm))
            .collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }

    /// 按索引排序
    /// ascending=true: 保持原顺序
    /// ascending=false: 反转行顺序
    pub fn sort_index(&self, ascending: bool) -> DataFrame {
        if ascending || self.nrows() == 0 {
            return self.clone();
        }
        // 反转: permutation = [n-1, n-2, ..., 0]
        let nrows = self.nrows();
        let perm: Vec<usize> = (0..nrows).rev().collect();
        let n_data: Vec<Series> = self
            .data
            .par_iter()
            .map(|s| gather_series(s, &perm))
            .collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }

    /// 返回每行是否有缺失值
    pub fn isnull_rows(&self) -> Vec<bool> {
        let nrows = self.nrows();
        (0..nrows)
            .into_par_iter()
            .map(|i| {
                self.data.iter().any(|s| match &s.data {
                    ColumnData::Int(v) => v[i].is_none(),
                    ColumnData::Float(v) => v[i].is_none(),
                    ColumnData::Bool(v) => v[i].is_none(),
                    ColumnData::String(v) => v[i].is_none(),
                    ColumnData::Categorical(c) => c.codes[i].is_none(),
                })
            })
            .collect()
    }

    /// 返回每行是否全部非缺失
    pub fn notnull_rows(&self) -> Vec<bool> {
        let nrows = self.nrows();
        (0..nrows)
            .into_par_iter()
            .map(|i| {
                self.data.iter().all(|s| match &s.data {
                    ColumnData::Int(v) => v[i].is_some(),
                    ColumnData::Float(v) => v[i].is_some(),
                    ColumnData::Bool(v) => v[i].is_some(),
                    ColumnData::String(v) => v[i].is_some(),
                    ColumnData::Categorical(c) => c.codes[i].is_some(),
                })
            })
            .collect()
    }

    /// 删除包含缺失值的行
    pub fn dropna(&self) -> DataFrame {
        self.dropna_rows()
    }

    /// 填充所有列的缺失值 (f64，仅对 Float64 列生效)
    pub fn fillna_all_f64(&self, v: f64) -> DataFrame {
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.fillna_f64(v)).collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }

    /// 填充所有列的缺失值 (i64，仅对 Int64 列生效)
    pub fn fillna_all_i64(&self, v: i64) -> DataFrame {
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.fillna_i64(v)).collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }

    /// 填充所有列的缺失值 (string，仅对 Object 列生效)
    pub fn fillna_all_string(&self, v: &str) -> DataFrame {
        let n_data: Vec<Series> = self.data.par_iter().map(|s| s.fillna_string(v)).collect();
        DataFrame {
            columns: self.columns.clone(),
            data: n_data,
        }
    }
}

/// 通用 Option 比较器 (用于 Ord 类型)
/// ascending=true: 升序，None 放最后
/// ascending=false: 降序，None 放最前
fn cmp_opt_ord<T: Ord + ?Sized>(
    a: Option<&T>,
    b: Option<&T>,
    ascending: bool,
) -> std::cmp::Ordering {
    match (a, b) {
        (Some(x), Some(y)) => {
            if ascending {
                x.cmp(y)
            } else {
                y.cmp(x)
            }
        }
        (Some(_), None) => {
            if ascending {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        }
        (None, Some(_)) => {
            if ascending {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            }
        }
        (None, None) => std::cmp::Ordering::Equal,
    }
}

/// f64 专用比较器 (f64 不实现 Ord，使用 partial_cmp)
fn cmp_opt_f64(a: Option<f64>, b: Option<f64>, ascending: bool) -> std::cmp::Ordering {
    match (a, b) {
        (Some(x), Some(y)) => {
            if ascending {
                x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)
            }
        }
        (Some(_), None) => {
            if ascending {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        }
        (None, Some(_)) => {
            if ascending {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            }
        }
        (None, None) => std::cmp::Ordering::Equal,
    }
}

/// &str 比较器
fn cmp_opt_str(a: Option<&str>, b: Option<&str>, ascending: bool) -> std::cmp::Ordering {
    cmp_opt_ord(a, b, ascending)
}

/// 按索引列表收集 Series 中的元素，返回新的 Series
/// 使用 Vec::with_capacity 预分配内存
fn gather_series(s: &Series, indices: &[usize]) -> Series {
    let new_data = match &s.data {
        ColumnData::Int(v) => {
            let mut out: Vec<Option<i64>> = Vec::with_capacity(indices.len());
            for &i in indices {
                out.push(v.get(i).and_then(|x| *x));
            }
            ColumnData::Int(out)
        }
        ColumnData::Float(v) => {
            let mut out: Vec<Option<f64>> = Vec::with_capacity(indices.len());
            for &i in indices {
                out.push(v.get(i).and_then(|x| *x));
            }
            ColumnData::Float(out)
        }
        ColumnData::Bool(v) => {
            let mut out: Vec<Option<bool>> = Vec::with_capacity(indices.len());
            for &i in indices {
                out.push(v.get(i).and_then(|x| *x));
            }
            ColumnData::Bool(out)
        }
        ColumnData::String(v) => {
            let mut out: Vec<Option<String>> = Vec::with_capacity(indices.len());
            for &i in indices {
                out.push(v.get(i).and_then(|x| x.clone()));
            }
            ColumnData::String(out)
        }
        ColumnData::Categorical(c) => {
            let mut out: Vec<Option<i32>> = Vec::with_capacity(indices.len());
            for &i in indices {
                out.push(c.codes.get(i).and_then(|x| *x));
            }
            ColumnData::Categorical(super::dtype::CategoricalData {
                categories: c.categories.clone(),
                codes: out,
                ordered: c.ordered,
            })
        }
    };
    Series {
        name: s.name.clone(),
        data: new_data,
    }
}

/// DataFrame fillna 用的填充值类型
#[derive(Debug, Clone)]
pub enum FillValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

// =====================================================================
// PyO3 绑定
// =====================================================================

#[pyclass(name = "_DataFrame", module = "rspandas", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyDataFrame {
    pub inner: DataFrame,
}

#[pymethods]
impl PyDataFrame {
    /// 构造: 接受 columns (list[str]) 和 series (list[_Series])
    #[new]
    fn new(columns: Vec<String>, series: Vec<PySeries>) -> PyResult<Self> {
        let data: Vec<Series> = series.into_iter().map(|s| s.inner).collect();
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

    // ---------- 显示辅助 ----------

    /// 逐行构造 dict 列表 (用于 Python 端显示)
    fn to_rows<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let rows = PyList::empty(py);
        let nrows = self.inner.nrows();
        for i in 0..nrows {
            let row = PyDict::new(py);
            for (col_name, series) in self.inner.columns.iter().zip(self.inner.data.iter()) {
                let val: pyo3::Py<pyo3::PyAny> = match &series.data {
                    super::dtype::ColumnData::Int(v) => match v.get(i) {
                        Some(Some(n)) => (*n).into_pyobject(py).unwrap().into_any().unbind(),
                        _ => py.None(),
                    },
                    super::dtype::ColumnData::Float(v) => match v.get(i) {
                        Some(Some(n)) => (*n).into_pyobject(py).unwrap().into_any().unbind(),
                        _ => py.None(),
                    },
                    super::dtype::ColumnData::Bool(v) => match v.get(i) {
                        Some(Some(n)) => (*n).into_pyobject(py).unwrap().as_any().clone().unbind(),
                        _ => py.None(),
                    },
                    super::dtype::ColumnData::String(v) => match v.get(i) {
                        Some(Some(s)) => s.clone().into_pyobject(py).unwrap().into_any().unbind(),
                        _ => py.None(),
                    },
                    super::dtype::ColumnData::Categorical(c) => match c.codes.get(i) {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataframe_basic() {
        let s1 = Series::new_int(Some("a".to_string()), vec![Some(1), Some(2), Some(3)]);
        let s2 = Series::new_string(
            Some("b".to_string()),
            vec![
                Some("x".to_string()),
                Some("y".to_string()),
                Some("z".to_string()),
            ],
        );
        let df =
            DataFrame::from_series(vec!["a".to_string(), "b".to_string()], vec![s1, s2]).unwrap();
        assert_eq!(df.shape(), (3, 2));
        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 2);
    }

    #[test]
    fn test_dataframe_head_tail() {
        let s1 = Series::new_int(None, vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
        let df = DataFrame::from_series(vec!["a".to_string()], vec![s1]).unwrap();
        assert_eq!(df.head(2).nrows(), 2);
        assert_eq!(df.tail(2).nrows(), 2);
    }

    #[test]
    fn test_dataframe_filter() {
        let s1 = Series::new_int(None, vec![Some(1), Some(2), Some(3), Some(4)]);
        let df = DataFrame::from_series(vec!["a".to_string()], vec![s1]).unwrap();
        let filtered = df.filter_rows(&[true, false, true, false]).unwrap();
        assert_eq!(filtered.nrows(), 2);
    }

    #[test]
    fn test_dataframe_duplicate_col() {
        let s1 = Series::new_int(None, vec![Some(1)]);
        let s2 = Series::new_int(None, vec![Some(2)]);
        let r = DataFrame::from_series(vec!["a".to_string(), "a".to_string()], vec![s1, s2]);
        assert!(r.is_err());
    }

    #[test]
    fn test_dataframe_shape_mismatch() {
        let s1 = Series::new_int(None, vec![Some(1), Some(2)]);
        let s2 = Series::new_int(None, vec![Some(3)]);
        let r = DataFrame::from_series(vec!["a".to_string(), "b".to_string()], vec![s1, s2]);
        assert!(r.is_err());
    }

    #[test]
    fn test_dataframe_sort_values() {
        // 整型列排序：升序，None 放最后
        let s1 = Series::new_int(None, vec![Some(3), None, Some(1), Some(2)]);
        let s2 = Series::new_string(
            None,
            vec![
                Some("c".to_string()),
                Some("d".to_string()),
                Some("a".to_string()),
                Some("b".to_string()),
            ],
        );
        let df = DataFrame::from_series(vec!["num".to_string(), "str".to_string()], vec![s1, s2])
            .expect("DataFrame 构建失败");

        // 按第 0 列 (num) 升序排序
        let sorted_asc = df.sort_values(&[0], true);
        assert_eq!(sorted_asc.nrows(), 4);
        // 验证 num 列顺序: 1, 2, 3, None
        if let ColumnData::Int(v) = &sorted_asc.data[0].data {
            assert_eq!(v[0], Some(1));
            assert_eq!(v[1], Some(2));
            assert_eq!(v[2], Some(3));
            assert_eq!(v[3], None);
        } else {
            panic!("dtype 错误");
        }
        // 验证 str 列跟随重排: a, b, c, d
        if let ColumnData::String(v) = &sorted_asc.data[1].data {
            assert_eq!(v[0], Some("a".to_string()));
            assert_eq!(v[1], Some("b".to_string()));
            assert_eq!(v[2], Some("c".to_string()));
            assert_eq!(v[3], Some("d".to_string()));
        } else {
            panic!("dtype 错误");
        }

        // 按第 0 列降序排序: None, 3, 2, 1
        let sorted_desc = df.sort_values(&[0], false);
        if let ColumnData::Int(v) = &sorted_desc.data[0].data {
            assert_eq!(v[0], None);
            assert_eq!(v[1], Some(3));
            assert_eq!(v[2], Some(2));
            assert_eq!(v[3], Some(1));
        } else {
            panic!("dtype 错误");
        }

        // 按字符串列 (索引 1) 升序排序
        let s3 = Series::new_string(
            None,
            vec![
                Some("banana".to_string()),
                Some("apple".to_string()),
                Some("cherry".to_string()),
            ],
        );
        let s4 = Series::new_int(None, vec![Some(10), Some(20), Some(30)]);
        let df2 = DataFrame::from_series(vec!["s".to_string(), "n".to_string()], vec![s3, s4])
            .expect("DataFrame 构建失败");
        let sorted_str = df2.sort_values(&[0], true);
        if let ColumnData::String(v) = &sorted_str.data[0].data {
            assert_eq!(v[0], Some("apple".to_string()));
            assert_eq!(v[1], Some("banana".to_string()));
            assert_eq!(v[2], Some("cherry".to_string()));
        } else {
            panic!("dtype 错误");
        }
        // 验证 n 列跟随重排: 20, 10, 30
        if let ColumnData::Int(v) = &sorted_str.data[1].data {
            assert_eq!(v[0], Some(20));
            assert_eq!(v[1], Some(10));
            assert_eq!(v[2], Some(30));
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_dataframe_sort_index() {
        let s1 = Series::new_int(None, vec![Some(1), Some(2), Some(3)]);
        let s2 = Series::new_float(None, vec![Some(1.5), Some(2.5), Some(3.5)]);
        let df = DataFrame::from_series(vec!["a".to_string(), "b".to_string()], vec![s1, s2])
            .expect("DataFrame 构建失败");

        // ascending=true: 保持原顺序
        let sorted_asc = df.sort_index(true);
        assert_eq!(sorted_asc.nrows(), 3);
        if let ColumnData::Int(v) = &sorted_asc.data[0].data {
            assert_eq!(v[0], Some(1));
            assert_eq!(v[1], Some(2));
            assert_eq!(v[2], Some(3));
        } else {
            panic!("dtype 错误");
        }
        if let ColumnData::Float(v) = &sorted_asc.data[1].data {
            assert_eq!(v[0], Some(1.5));
            assert_eq!(v[1], Some(2.5));
            assert_eq!(v[2], Some(3.5));
        } else {
            panic!("dtype 错误");
        }

        // ascending=false: 反转行顺序
        let sorted_desc = df.sort_index(false);
        if let ColumnData::Int(v) = &sorted_desc.data[0].data {
            assert_eq!(v[0], Some(3));
            assert_eq!(v[1], Some(2));
            assert_eq!(v[2], Some(1));
        } else {
            panic!("dtype 错误");
        }
        if let ColumnData::Float(v) = &sorted_desc.data[1].data {
            assert_eq!(v[0], Some(3.5));
            assert_eq!(v[1], Some(2.5));
            assert_eq!(v[2], Some(1.5));
        } else {
            panic!("dtype 错误");
        }
    }

    #[test]
    fn test_dataframe_dropna() {
        // 列 a: [1, None, 3, None]
        // 列 b: [1.0, 2.0, None, None]
        // 期望删除后只剩第 0 行 (两列都非空)
        let s1 = Series::new_int(None, vec![Some(1), None, Some(3), None]);
        let s2 = Series::new_float(None, vec![Some(1.0), Some(2.0), None, None]);
        let df = DataFrame::from_series(vec!["a".to_string(), "b".to_string()], vec![s1, s2])
            .expect("DataFrame 构建失败");
        assert_eq!(df.nrows(), 4);

        let dropped = df.dropna();
        assert_eq!(dropped.nrows(), 1);
        // 验证剩余的第 0 行数据
        if let ColumnData::Int(v) = &dropped.data[0].data {
            assert_eq!(v[0], Some(1));
        } else {
            panic!("dtype 错误");
        }
        if let ColumnData::Float(v) = &dropped.data[1].data {
            assert_eq!(v[0], Some(1.0));
        } else {
            panic!("dtype 错误");
        }

        // 全非空 DataFrame 删除后行数不变
        let s3 = Series::new_int(None, vec![Some(1), Some(2)]);
        let s4 = Series::new_string(None, vec![Some("x".to_string()), Some("y".to_string())]);
        let df2 = DataFrame::from_series(vec!["a".to_string(), "b".to_string()], vec![s3, s4])
            .expect("DataFrame 构建失败");
        assert_eq!(df2.dropna().nrows(), 2);

        // 空 DataFrame dropna 安全
        let empty = DataFrame::new_empty();
        assert_eq!(empty.dropna().nrows(), 0);
    }

    #[test]
    fn test_dataframe_isnull() {
        // 列 a: [1, None, 3]
        // 列 b: ["x", "y", None]
        // isnull_rows: [false, true, true] (任意列为 None)
        // notnull_rows: [true, false, false] (所有列非 None)
        let s1 = Series::new_int(None, vec![Some(1), None, Some(3)]);
        let s2 = Series::new_string(
            None,
            vec![Some("x".to_string()), Some("y".to_string()), None],
        );
        let df = DataFrame::from_series(vec!["a".to_string(), "b".to_string()], vec![s1, s2])
            .expect("DataFrame 构建失败");

        let isnull = df.isnull_rows();
        assert_eq!(isnull, vec![false, true, true]);

        let notnull = df.notnull_rows();
        assert_eq!(notnull, vec![true, false, false]);

        // 全非空: isnull 全 false, notnull 全 true
        let s3 = Series::new_int(None, vec![Some(1), Some(2)]);
        let s4 = Series::new_float(None, vec![Some(1.5), Some(2.5)]);
        let df2 = DataFrame::from_series(vec!["a".to_string(), "b".to_string()], vec![s3, s4])
            .expect("DataFrame 构建失败");
        assert_eq!(df2.isnull_rows(), vec![false, false]);
        assert_eq!(df2.notnull_rows(), vec![true, true]);

        // 空 DataFrame: isnull/notnull 返回空 Vec
        let empty = DataFrame::new_empty();
        assert!(empty.isnull_rows().is_empty());
        assert!(empty.notnull_rows().is_empty());
    }

    // 防止 DType 未使用警告
    #[test]
    fn test_dtype_compile() {
        let _ = DType::Int64;
    }
}
