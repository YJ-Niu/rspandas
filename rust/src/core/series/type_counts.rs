//! 单次遍历同时完成 apply(type) + type 计数，避免两次 1M 行遍历。
//!
//! 对 object 列：并行遍历每一行，判断元素的 int/float/bool/str/None/其他 类型，
//! 同时生成 PyObject 类型对象列表 + 维护 HashMap 计数。
//!
//! 对纯类型列（Int64/Float64/Bool/String）：类型字符串 100% 确定，
//! 但 out_types 向量仍需逐元素生成对应 Python type 对象。

use pyo3::Py;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyFloat, PyInt, PyList, PyString};
use rayon::prelude::*;
use std::collections::HashMap;

use crate::core::dtype::ColumnData;
use crate::core::series::{PySeries, Series};

impl Series {
    /// 单次遍历同时完成 ``apply(type)`` + 类型名计数，避免两次遍历。
    ///
    /// 返回 4 元组 ``(inner_pyseries, out_list, type_names, counts)``：
    /// - ``inner_pyseries``：``Py<PyAny>``，实为一个 ``PySeries``，存储 dtype=object，
    ///   每个元素是类型 repr 字符串（``"<class 'int'>"`` 等），与 ``apply_type()``
    ///   返回的 inner 完全一致；
    /// - ``out_list``：``Py<PyAny>``，实为一个 ``PyList``，每个位置放对应的
    ///   Python type 对象（None 位置放 None），与 ``apply_type()`` 返回的 out 一致；
    /// - ``type_names`` / ``counts``：按 counts 降序（ties 名字升序），仅非 None 计入。
    #[allow(clippy::type_complexity)] // Python 暴露层需要四元组，不适合再拆分
    pub fn apply_type_with_counts<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Vec<String>, Vec<usize>)> {
        const INT_REPR: &str = "<class 'int'>";
        const FLOAT_REPR: &str = "<class 'float'>";
        const BOOL_REPR: &str = "<class 'bool'>";
        const STR_REPR: &str = "<class 'str'>";

        // 预先取到各 Python type 对象，避免循环中反复构造
        let int_type = py.get_type::<PyInt>().unbind();
        let float_type = py.get_type::<PyFloat>().unbind();
        let bool_type = py.get_type::<PyBool>().unbind();
        let str_type = py.get_type::<PyString>().unbind();

        let len = self.len();
        let mut out_types: Vec<Py<PyAny>> = Vec::with_capacity(len);
        let mut reprs: Vec<Option<String>> = Vec::with_capacity(len);

        // ---------- 纯类型快速路径：类型 100% 确定，直接统计 + 填充 ----------
        match &self.data {
            ColumnData::Int(v) => {
                let mut int_count = 0usize;
                for opt in v {
                    if opt.is_some() {
                        out_types.push(int_type.bind(py).clone().into_any().unbind());
                        reprs.push(Some(INT_REPR.to_owned()));
                        int_count += 1;
                    } else {
                        out_types.push(py.None());
                        reprs.push(None);
                    }
                }
                let (names, counts) = build_sorted_pairs(&[("int", int_count)]);
                return build_4tuple(py, self.name.clone(), reprs, out_types, names, counts);
            }
            ColumnData::Float(v) => {
                let mut float_count = 0usize;
                for opt in v {
                    if opt.is_some() {
                        out_types.push(float_type.bind(py).clone().into_any().unbind());
                        reprs.push(Some(FLOAT_REPR.to_owned()));
                        float_count += 1;
                    } else {
                        out_types.push(py.None());
                        reprs.push(None);
                    }
                }
                let (names, counts) = build_sorted_pairs(&[("float", float_count)]);
                return build_4tuple(py, self.name.clone(), reprs, out_types, names, counts);
            }
            ColumnData::Bool(v) => {
                let mut bool_count = 0usize;
                for opt in v {
                    if opt.is_some() {
                        out_types.push(bool_type.bind(py).clone().into_any().unbind());
                        reprs.push(Some(BOOL_REPR.to_owned()));
                        bool_count += 1;
                    } else {
                        out_types.push(py.None());
                        reprs.push(None);
                    }
                }
                let (names, counts) = build_sorted_pairs(&[("bool", bool_count)]);
                return build_4tuple(py, self.name.clone(), reprs, out_types, names, counts);
            }
            _ => {}
        }

        // ---------- String 列：字符串存储，逐值 classify + 计数 ----------
        if let ColumnData::String(v) = &self.data {
            // 阶段一：脱离 GIL 并行分类，得到 (分类编码, 原始字符串是否 None)
            // 编码：0=缺失/1=int/2=float/3=bool/4=str（与 classify_type_code 一致）
            let codes: Vec<Option<u8>> = py.detach(|| {
                v.par_iter()
                    .map(|opt| opt.as_ref().map(|s| classify_type_code(s)))
                    .collect()
            });

            // 阶段二：持有 GIL 构造 PyObject 列表 + 累计计数
            let mut counts_map: HashMap<&'static str, usize> = HashMap::with_capacity(4);
            for c in codes {
                match c {
                    Some(1) => {
                        out_types.push(int_type.bind(py).clone().into_any().unbind());
                        reprs.push(Some(INT_REPR.to_owned()));
                        *counts_map.entry("int").or_insert(0) += 1;
                    }
                    Some(2) => {
                        out_types.push(float_type.bind(py).clone().into_any().unbind());
                        reprs.push(Some(FLOAT_REPR.to_owned()));
                        *counts_map.entry("float").or_insert(0) += 1;
                    }
                    Some(3) => {
                        out_types.push(bool_type.bind(py).clone().into_any().unbind());
                        reprs.push(Some(BOOL_REPR.to_owned()));
                        *counts_map.entry("bool").or_insert(0) += 1;
                    }
                    Some(4) => {
                        out_types.push(str_type.bind(py).clone().into_any().unbind());
                        reprs.push(Some(STR_REPR.to_owned()));
                        *counts_map.entry("str").or_insert(0) += 1;
                    }
                    _ => {
                        out_types.push(py.None());
                        reprs.push(None);
                    }
                }
            }
            let (names, counts) = hashmap_to_sorted_pairs(counts_map);
            return build_4tuple(py, self.name.clone(), reprs, out_types, names, counts);
        }

        // ---------- Categorical 列：所有类别都视为 str ----------
        if let ColumnData::Categorical(c) = &self.data {
            let mut str_count = 0usize;
            for code in &c.codes {
                if code.is_some() {
                    out_types.push(str_type.bind(py).clone().into_any().unbind());
                    reprs.push(Some(STR_REPR.to_owned()));
                    str_count += 1;
                } else {
                    out_types.push(py.None());
                    reprs.push(None);
                }
            }
            let (names, counts) = build_sorted_pairs(&[("str", str_count)]);
            return build_4tuple(py, self.name.clone(), reprs, out_types, names, counts);
        }

        // ---------- 兜底：理论上不该到这里 ----------
        let (names, counts) = build_sorted_pairs(&[]);
        build_4tuple(py, self.name.clone(), reprs, out_types, names, counts)
    }

    /// 仅统计各元素的 Python type 名称计数（不生成任何 PyObject、不填 Vec），
    /// 用于 apply(type).value_counts 短路。
    ///
    /// 返回按 count 降序 + 名字升序排好的 (type_name_vec, count_vec)。
    /// None 位置不计入（dropna=True 语义）。
    pub fn type_counts_only(&self) -> (Vec<String>, Vec<usize>) {
        // ---------- 纯类型快速路径：0 次循环，直接 count() ----------
        match &self.data {
            ColumnData::Int(_) => {
                return build_sorted_pairs(&[("int", self.count())]);
            }
            ColumnData::Float(_) => {
                return build_sorted_pairs(&[("float", self.count())]);
            }
            ColumnData::Bool(_) => {
                return build_sorted_pairs(&[("bool", self.count())]);
            }
            _ => {}
        }

        // ---------- String/Object 列：并行分类 + HashMap 计数 ----------
        if let ColumnData::String(v) = &self.data {
            // 阶段一：脱离 GIL 并行分类，得到计数映射（使用本地计数数组再归并）
            // 编码：1=int / 2=float / 3=bool / 4=str / 其他=other
            let counts_array: [usize; 5] = {
                // per-thread local counts: [int, float, bool, str, other]
                // 使用 rayon 的 fold + reduce 做无锁并行计数
                v.par_iter()
                    .fold(
                        || [0usize; 5],
                        |mut acc, opt| {
                            if let Some(s) = opt.as_ref() {
                                match classify_type_code(s) {
                                    1 => acc[0] += 1, // int
                                    2 => acc[1] += 1, // float
                                    3 => acc[2] += 1, // bool
                                    4 => acc[3] += 1, // str
                                    _ => acc[4] += 1, // other (理论不会到)
                                }
                            }
                            // None -> 不计入
                            acc
                        },
                    )
                    .reduce(
                        || [0usize; 5],
                        |mut a, b| {
                            for i in 0..5 {
                                a[i] += b[i];
                            }
                            a
                        },
                    )
            };

            let [ic, fc, bc, sc, oc] = counts_array;
            let mut pairs: Vec<(&'static str, usize)> = Vec::with_capacity(5);
            if ic > 0 {
                pairs.push(("int", ic));
            }
            if fc > 0 {
                pairs.push(("float", fc));
            }
            if bc > 0 {
                pairs.push(("bool", bc));
            }
            if sc > 0 {
                pairs.push(("str", sc));
            }
            if oc > 0 {
                pairs.push(("other", oc));
            }
            // 按 count 降序，按名字升序（ties 稳定）
            pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
            let names = pairs.iter().map(|(n, _)| n.to_string()).collect();
            let counts = pairs.iter().map(|(_, c)| *c).collect();
            return (names, counts);
        }

        // ---------- Categorical 列：所有非 None 视为 str ----------
        if let ColumnData::Categorical(c) = &self.data {
            let str_count = c.codes.iter().filter(|c| c.is_some()).count();
            return build_sorted_pairs(&[("str", str_count)]);
        }

        // ---------- 兜底：空 ----------
        build_sorted_pairs(&[])
    }
}

// =====================================================================
// 辅助：把 (reprs, out_types, names, counts) 打包成 4 元组
// =====================================================================

/// 辅助函数：基于 reprs 构造 PySeries（存储为 String/Object），
/// 基于 out_types 构造 PyList，再整体打包成 ``(Py<PyAny>, Py<PyAny>, names, counts)``。
#[allow(clippy::type_complexity)]
fn build_4tuple<'py>(
    py: Python<'py>,
    name: Option<String>,
    reprs: Vec<Option<String>>,
    out_types: Vec<Py<PyAny>>,
    names: Vec<String>,
    counts: Vec<usize>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Vec<String>, Vec<usize>)> {
    // 第 1 元：PySeries（存储类型 repr 字符串，与 apply_type 返回的 inner 相同）
    let series_inner = Series::new_string(name, reprs);
    let pyseries = PySeries {
        inner: series_inner,
    };
    let inner_any: Py<PyAny> = Py::new(py, pyseries)?.into_any();

    // 第 2 元：PyList（每个位置放 Python type 对象或 None，与 apply_type 的 out 一致）
    let out_list = PyList::empty(py);
    for obj in out_types.iter() {
        out_list.append(obj)?;
    }
    let out_any: Py<PyAny> = out_list.into_any().unbind();

    Ok((inner_any, out_any, names, counts))
}

// =====================================================================
// 辅助函数
// =====================================================================

/// 将字符串分类为 Python 类型编码（与 string_ops.rs 的 classify_type_code 保持一致）。
///
/// 返回：
/// - 1 -> int
/// - 2 -> float
/// - 3 -> bool
/// - 4 -> str（其他）
fn classify_type_code(s: &str) -> u8 {
    if s.parse::<i64>().is_ok() {
        1
    } else if s.parse::<f64>().is_ok() {
        2
    } else if matches!(s, "True" | "TRUE" | "true" | "False" | "FALSE" | "false") {
        3
    } else {
        4
    }
}

/// 纯类型列：用 slice of (name, count) 构造排序后的 (names, counts)，跳过计数为 0 的项。
fn build_sorted_pairs(items: &[(&'static str, usize)]) -> (Vec<String>, Vec<usize>) {
    let mut filtered: Vec<(&'static str, usize)> =
        items.iter().filter(|(_, c)| *c > 0).copied().collect();
    // 按 count 降序，按名字升序（ties 稳定）
    filtered.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let names = filtered.iter().map(|(n, _)| n.to_string()).collect();
    let counts = filtered.iter().map(|(_, c)| *c).collect();
    (names, counts)
}

/// object 列：把 HashMap<type_name, count> 转换为排序后的 (names, counts)。
fn hashmap_to_sorted_pairs(map: HashMap<&'static str, usize>) -> (Vec<String>, Vec<usize>) {
    let mut pairs: Vec<(&'static str, usize)> = map.into_iter().collect();
    // 按 count 降序，按名字升序（ties 稳定）
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let names = pairs.iter().map(|(n, _)| n.to_string()).collect();
    let counts = pairs.iter().map(|(_, c)| *c).collect();
    (names, counts)
}
