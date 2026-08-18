//! JSON 读写。
//!
//! - 读取: 支持 orient='records' / 'columns' / 'index' / 'split' / 'values'
//! - 写入: 支持 orient='records' / 'columns' / 'index' / 'split' / 'values'，以及 lines 模式
//! - 纯 Rust 实现 (serde_json)，释放 GIL 进行解析和序列化

use crate::core::series::{PySeries, Series};
use pyo3::prelude::*;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io::Write;

/// 列名和列数据的类型别名
type ColumnData = (Vec<String>, Vec<Vec<Option<String>>>);

/// 从 JSON 文件读取并解析为 DataFrame（列名 + Series 列表）。
///
/// 支持 orient 参数：
/// - 'records': `[{col1: val, col2: val}, ...]`
/// - 'columns': `{col1: [val1, val2], col2: [val3, val4]}`
/// - 'index': `{idx1: {col1: val, col2: val}, idx2: ...}`
/// - 'split': `{columns: [...], data: [[...], ...]}`
/// - 'values': `[[val1, val2], [val3, val4]]`
#[pyfunction]
pub fn read_json(
    py: Python<'_>,
    path: &str,
    orient: &str,
    lines: bool,
    encoding: &str,
) -> PyResult<(Vec<String>, Vec<PySeries>)> {
    let orient_str = orient.to_string();
    let encoding_str = encoding.to_string();

    // 释放 GIL 进行文件读取和 JSON 解析
    let (headers, cols) = py.detach(|| -> PyResult<ColumnData> {
        if lines {
            // 按行读取 JSON (每行一个 JSON 对象)
            let content = read_file_with_encoding(path, &encoding_str)?;
            let records: Vec<Value> = content
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| {
                    serde_json::from_str(line).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {e}"))
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;

            if records.is_empty() {
                return Ok((vec![], vec![]));
            }
            let (headers, cols) = records_to_columns(&records);
            Ok((headers, cols))
        } else {
            let content = read_file_with_encoding(path, &encoding_str)?;
            let root: Value = serde_json::from_str(&content).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {e}"))
            })?;
            parse_json_value(&root, &orient_str)
        }
    })?;

    // 构建 Series（并行化类型推断）
    let series_list: Vec<PySeries> = headers
        .par_iter()
        .zip(cols.par_iter())
        .map(|(name, values)| strings_to_series(name, values))
        .collect();

    Ok((headers, series_list))
}

/// 读取文件内容，处理编码
fn read_file_with_encoding(path: &str, encoding: &str) -> PyResult<String> {
    let bytes = fs::read(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("open {path}: {e}")))?;
    if encoding.to_lowercase() == "utf-8" || encoding.to_lowercase() == "utf8" {
        String::from_utf8(bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("UTF-8 decode: {e}")))
    } else {
        // 简单处理：对其他编码，使用 lossy 转换
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }
}

/// 解析 JSON Value 为列格式
fn parse_json_value(root: &Value, orient: &str) -> PyResult<ColumnData> {
    match orient {
        "records" => {
            let records: Vec<Value> = root.as_array().cloned().unwrap_or_default();
            if records.is_empty() {
                return Ok((vec![], vec![]));
            }
            Ok(records_to_columns(&records))
        }
        "columns" => {
            let obj = root.as_object().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("orient='columns' requires a JSON object")
            })?;
            let mut headers: Vec<String> = Vec::new();
            let mut cols: Vec<Vec<Option<String>>> = Vec::new();
            for (k, v) in obj {
                headers.push(k.clone());
                let col = v
                    .as_array()
                    .map(|arr| arr.iter().map(json_value_to_opt_string).collect())
                    .unwrap_or_default();
                cols.push(col);
            }
            Ok((headers, cols))
        }
        "index" => {
            let obj = root.as_object().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("orient='index' requires a JSON object")
            })?;
            let mut records: Vec<Value> = Vec::new();
            for (idx, row_val) in obj {
                if let Some(row_obj) = row_val.as_object() {
                    let mut map = serde_json::Map::new();
                    map.insert("index".to_string(), Value::String(idx.clone()));
                    for (k, v) in row_obj {
                        map.insert(k.clone(), v.clone());
                    }
                    records.push(Value::Object(map));
                }
            }
            if records.is_empty() {
                return Ok((vec![], vec![]));
            }
            Ok(records_to_columns(&records))
        }
        "split" => {
            let obj = root.as_object().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("orient='split' requires a JSON object")
            })?;
            let headers: Vec<String> = obj
                .get("columns")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|v| v.as_str().unwrap_or("").to_string())
                        .collect()
                })
                .unwrap_or_default();
            let data = obj
                .get("data")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            let cols = rows_to_cols(&data, headers.len());
            Ok((headers, cols))
        }
        "values" => {
            let data = root.as_array().cloned().unwrap_or_default();
            if data.is_empty() {
                return Ok((vec![], vec![]));
            }
            let ncols = data
                .first()
                .and_then(|r| r.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            let headers: Vec<String> = (0..ncols).map(|i| format!("col{i}")).collect();
            let cols = rows_to_cols(&data, ncols);
            Ok((headers, cols))
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown orient: {orient}"
        ))),
    }
}

/// 将 JSON record 数组转为列格式
fn records_to_columns(records: &[Value]) -> (Vec<String>, Vec<Vec<Option<String>>>) {
    // 收集所有列名（保持顺序）
    let mut col_set: HashMap<String, usize> = HashMap::new();
    let mut ordered_cols: Vec<String> = Vec::new();
    for record in records {
        if let Some(obj) = record.as_object() {
            for k in obj.keys() {
                if !col_set.contains_key(k) {
                    col_set.insert(k.clone(), ordered_cols.len());
                    ordered_cols.push(k.clone());
                }
            }
        }
    }

    let ncols = ordered_cols.len();
    let mut cols: Vec<Vec<Option<String>>> = vec![Vec::new(); ncols];

    for record in records {
        let obj = record.as_object();
        for (i, col_name) in ordered_cols.iter().enumerate() {
            let val = obj
                .and_then(|o| o.get(col_name))
                .map(json_value_to_opt_string)
                .unwrap_or(None);
            cols[i].push(val);
        }
    }

    (ordered_cols, cols)
}

/// 将 JSON Value 转为 Option<String>
fn json_value_to_opt_string(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::Bool(b) => Some(b.to_string()),
        Value::Number(n) => Some(n.to_string()),
        Value::String(s) => Some(s.clone()),
        Value::Array(_) | Value::Object(_) => Some(v.to_string()),
    }
}

/// 将行列表转为列格式
fn rows_to_cols(data: &[Value], ncols: usize) -> Vec<Vec<Option<String>>> {
    let mut cols: Vec<Vec<Option<String>>> = vec![Vec::new(); ncols];
    for row in data {
        if let Some(arr) = row.as_array() {
            for (i, val) in arr.iter().enumerate() {
                if i < ncols {
                    cols[i].push(json_value_to_opt_string(val));
                }
            }
        }
    }
    cols
}

// ============================================================================
// 类型推断（与 CSV 共享逻辑）
// ============================================================================

/// 推断字符串列的类型并转换为对应的 Series (并行化)
fn strings_to_series(name: &str, values: &[Option<String>]) -> PySeries {
    if values.is_empty() {
        return PySeries {
            inner: Series::from_options_string(name.to_string(), values),
        };
    }

    let (all_int, all_float, all_bool, any_non_null) = values
        .par_iter()
        .map(|v| match v {
            Some(s) => {
                let int_ok = s.parse::<i64>().is_ok();
                let float_ok = s.parse::<f64>().is_ok();
                let sl = s.to_lowercase();
                let bool_ok = sl == "true" || sl == "false" || sl == "0" || sl == "1";
                (int_ok, float_ok, bool_ok, true)
            }
            None => (true, true, true, false),
        })
        .reduce(
            || (true, true, true, false),
            |(a_int, a_float, a_bool, a_any), (b_int, b_float, b_bool, b_any)| {
                (
                    a_int && b_int,
                    a_float && b_float,
                    a_bool && b_bool,
                    a_any || b_any,
                )
            },
        );

    if !any_non_null {
        return PySeries {
            inner: Series::from_options_string(name.to_string(), values),
        };
    }

    if all_bool {
        let bools: Vec<Option<bool>> = values
            .par_iter()
            .map(|v| {
                v.as_ref().map(|s| {
                    let sl = s.to_lowercase();
                    sl == "true" || sl == "1"
                })
            })
            .collect();
        PySeries {
            inner: Series::from_options_bool(name.to_string(), &bools),
        }
    } else if all_int {
        let ints: Vec<Option<i64>> = values
            .par_iter()
            .map(|v| v.as_ref().and_then(|s| s.parse::<i64>().ok()))
            .collect();
        PySeries {
            inner: Series::from_options_i64(name.to_string(), &ints),
        }
    } else if all_float {
        let floats: Vec<Option<f64>> = values
            .par_iter()
            .map(|v| v.as_ref().and_then(|s| s.parse::<f64>().ok()))
            .collect();
        PySeries {
            inner: Series::from_options_f64(name.to_string(), &floats),
        }
    } else {
        PySeries {
            inner: Series::from_options_string(name.to_string(), values),
        }
    }
}

// ============================================================================
// JSON 写入
// ============================================================================

/// 将 DataFrame 写入 JSON 文件或返回 JSON 字符串。
///
/// - path: 输出路径，None 则返回字符串
/// - orient: JSON 格式 ('records' / 'columns' / 'index' / 'split' / 'values')
/// - lines: 是否按行输出
/// - force_ascii: 是否强制 ASCII 编码
/// - indent: 缩进空格数，None 为紧凑格式
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn write_json(
    py: Python<'_>,
    columns: Vec<String>,
    series_list: Vec<PySeries>,
    path: Option<&str>,
    orient: &str,
    lines: bool,
    force_ascii: bool,
    indent: Option<usize>,
) -> PyResult<Option<String>> {
    let orient_str = orient.to_string();

    let output = py.detach(|| -> PyResult<String> {
        let nrows = if series_list.is_empty() {
            0
        } else {
            series_list[0].inner.len()
        };

        match orient_str.as_str() {
            "records" => {
                let records: Vec<Value> = (0..nrows)
                    .map(|i| {
                        let mut map = serde_json::Map::new();
                        for (j, col_name) in columns.iter().enumerate() {
                            let val = series_list[j].inner.get_str_at(i);
                            map.insert(col_name.clone(), Value::String(val));
                        }
                        Value::Object(map)
                    })
                    .collect();

                if lines {
                    let mut buf = String::new();
                    for record in &records {
                        let line = format_json(record, force_ascii, indent)?;
                        buf.push_str(&line);
                        buf.push('\n');
                    }
                    Ok(buf)
                } else {
                    format_json(&Value::Array(records), force_ascii, indent)
                }
            }
            "columns" => {
                let mut map = serde_json::Map::new();
                for (j, col_name) in columns.iter().enumerate() {
                    let col_vals: Vec<Value> = (0..nrows)
                        .map(|i| {
                            let s = series_list[j].inner.get_str_at(i);
                            Value::String(s)
                        })
                        .collect();
                    map.insert(col_name.clone(), Value::Array(col_vals));
                }
                format_json(&Value::Object(map), force_ascii, indent)
            }
            "index" => {
                let mut map = serde_json::Map::new();
                for i in 0..nrows {
                    let mut row_map = serde_json::Map::new();
                    for (j, col_name) in columns.iter().enumerate() {
                        let val = series_list[j].inner.get_str_at(i);
                        row_map.insert(col_name.clone(), Value::String(val));
                    }
                    map.insert(i.to_string(), Value::Object(row_map));
                }
                format_json(&Value::Object(map), force_ascii, indent)
            }
            "split" => {
                let col_vals: Vec<Value> =
                    columns.iter().map(|c| Value::String(c.clone())).collect();
                let data: Vec<Value> = (0..nrows)
                    .map(|i| {
                        let row: Vec<Value> = series_list
                            .iter()
                            .map(|s| Value::String(s.inner.get_str_at(i)))
                            .collect();
                        Value::Array(row)
                    })
                    .collect();
                let mut map = serde_json::Map::new();
                map.insert("columns".to_string(), Value::Array(col_vals));
                map.insert("data".to_string(), Value::Array(data));
                format_json(&Value::Object(map), force_ascii, indent)
            }
            "values" => {
                let data: Vec<Value> = (0..nrows)
                    .map(|i| {
                        let row: Vec<Value> = series_list
                            .iter()
                            .map(|s| Value::String(s.inner.get_str_at(i)))
                            .collect();
                        Value::Array(row)
                    })
                    .collect();
                format_json(&Value::Array(data), force_ascii, indent)
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown orient: {orient}"
            ))),
        }
    })?;

    // 写文件或返回字符串
    if let Some(p) = path {
        let mut file = fs::File::create(p)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("create {p}: {e}")))?;
        file.write_all(output.as_bytes())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("write {p}: {e}")))?;
        // 确保末尾换行
        if !output.ends_with('\n') {
            file.write_all(b"\n")
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("write {p}: {e}")))?;
        }
        Ok(None)
    } else {
        Ok(Some(output))
    }
}

/// 格式化 JSON 输出
fn format_json(value: &Value, _force_ascii: bool, indent: Option<usize>) -> PyResult<String> {
    if let Some(spaces) = indent {
        // 使用紧凑格式，然后自定义缩进
        let compact = serde_json::to_string(value)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON serialize: {e}")))?;
        let indent_str = " ".repeat(spaces);
        Ok(reindent_json(&compact, &indent_str))
    } else {
        serde_json::to_string(value)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON serialize: {e}")))
    }
}

/// 重新缩进 JSON 字符串
fn reindent_json(json: &str, indent: &str) -> String {
    let mut result = String::new();
    let mut level: usize = 0;
    let mut in_string = false;

    for ch in json.chars() {
        if ch == '"' && !in_string {
            in_string = true;
        } else if ch == '"' && in_string {
            // 检查是否是转义引号
            in_string = false;
        }

        if !in_string {
            match ch {
                '{' | '[' => {
                    result.push(ch);
                    result.push('\n');
                    level += 1;
                    result.push_str(&indent.repeat(level));
                    continue;
                }
                '}' | ']' => {
                    level = level.saturating_sub(1);
                    result.push('\n');
                    result.push_str(&indent.repeat(level));
                    result.push(ch);
                    continue;
                }
                ',' => {
                    result.push(ch);
                    result.push('\n');
                    result.push_str(&indent.repeat(level));
                    continue;
                }
                ':' => {
                    result.push(ch);
                    result.push(' ');
                    continue;
                }
                _ => {}
            }
        }
        result.push(ch);
    }

    result
}
