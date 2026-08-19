//! CSV 读写。
//!
//! - 读取: 自动推断每列类型 (int -> float -> bool -> string)，None 表示空字符串
//! - 写入: 顺序写出列

use crate::core::dtype::DType;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Write};

use crate::core::series::{PySeries, Series};

type CsvParseResult = (Vec<String>, Vec<Vec<Option<String>>>);

/// 从 CSV 字符串构造 DataFrame（支持自定义分隔符、引号等参数）
#[allow(clippy::too_many_arguments)]
fn parse_csv_string(
    content: &str,
    has_header: bool,
    delimiter: u8,
    quote: u8,
    quoting: bool,
    double_quote: bool,
    escape: Option<u8>,
    skip_initial_space: bool,
) -> PyResult<CsvParseResult> {
    let mut builder = csv::ReaderBuilder::new();
    builder
        .has_headers(has_header)
        .flexible(true)
        .delimiter(delimiter)
        .quote(quote)
        .quoting(quoting)
        .double_quote(double_quote);
    if let Some(esc) = escape {
        builder.escape(Some(esc));
    } else {
        builder.escape(None);
    }
    let mut rdr = builder.from_reader(content.as_bytes());

    let mut headers: Vec<String> = Vec::new();
    if has_header {
        for h in rdr
            .headers()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("csv header: {e}")))?
            .iter()
        {
            headers.push(h.to_string());
        }
    }

    let mut cols: Vec<Vec<Option<String>>> = Vec::new();
    let mut ncols_hint: Option<usize> = None;
    for (row_idx, result) in rdr.records().enumerate() {
        let record = result.map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("csv row {row_idx}: {e}"))
        })?;
        if !has_header && headers.is_empty() {
            for i in 0..record.len() {
                headers.push(format!("col{i}"));
            }
        }
        // 初始化列
        if ncols_hint.is_none() {
            ncols_hint = Some(record.len());
            cols = vec![Vec::new(); ncols_hint.unwrap()];
        }
        for (i, val) in record.iter().enumerate() {
            if i >= cols.len() {
                cols.push(Vec::new());
                if !has_header {
                    headers.push(format!("col{i}"));
                }
            }
            if val.is_empty() {
                cols[i].push(None);
            } else if skip_initial_space {
                cols[i].push(Some(val.trim_start().to_string()));
            } else {
                cols[i].push(Some(val.to_string()));
            }
        }
        cols.iter_mut().skip(record.len()).for_each(|col| {
            col.push(None);
        });
    }

    // 只有表头没有数据: 用表头长度初始化空列
    if cols.is_empty() && !headers.is_empty() {
        cols = vec![Vec::new(); headers.len()];
    }

    Ok((headers, cols))
}

/// 推断字符串列的类型 (并行化)
fn infer_column(values: &[Option<String>]) -> (crate::core::dtype::DType, Vec<Option<String>>) {
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
    let dtype = if !any_non_null {
        DType::Object
    } else if all_bool {
        DType::Bool
    } else if all_int {
        DType::Int64
    } else if all_float {
        DType::Float64
    } else {
        DType::Object
    };
    (dtype, values.to_vec())
}

/// 将 string 列转换为目标 dtype 的 string 表示 (并行化)
fn cast_strings(
    values: &[Option<String>],
    target: crate::core::dtype::DType,
) -> Vec<Option<String>> {
    values
        .par_iter()
        .map(|opt| match opt {
            None => None,
            Some(s) => match target {
                DType::Int64 => {
                    if let Ok(i) = s.parse::<i64>() {
                        Some(i.to_string())
                    } else {
                        Some(s.clone())
                    }
                }
                DType::Float64 => {
                    if let Ok(f) = s.parse::<f64>() {
                        Some(f.to_string())
                    } else {
                        Some(s.clone())
                    }
                }
                DType::Bool => {
                    let sl = s.to_lowercase();
                    if sl == "true" || sl == "1" {
                        Some("true".to_string())
                    } else if sl == "false" || sl == "0" {
                        Some("false".to_string())
                    } else {
                        Some(s.clone())
                    }
                }
                DType::Object => Some(s.clone()),
                DType::Categorical => Some(s.clone()),
            },
        })
        .collect()
}

/// 从 CSV 字符串构造 DataFrame（释放 GIL 进行解析）
///
/// 参数:
/// - delimiter: 字段分隔符字节（默认 b','）
/// - quote: 引号字符字节（默认 b'"'）
/// - quoting: 是否启用引号处理（默认 true）
/// - double_quote: 是否双引号转义（默认 true）
/// - escape: 转义字符（None 表示无）
/// - skip_initial_space: 是否跳过字段前导空格（默认 false）
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn read_csv_string<'py>(
    py: Python<'py>,
    content: &str,
    has_header: bool,
    delimiter: Option<u8>,
    quote: Option<u8>,
    quoting: Option<bool>,
    double_quote: Option<bool>,
    escape: Option<u8>,
    skip_initial_space: Option<bool>,
) -> PyResult<(Vec<String>, Vec<PySeries>)> {
    let delimiter = delimiter.unwrap_or(b',');
    let quote = quote.unwrap_or(b'"');
    let quoting = quoting.unwrap_or(true);
    let double_quote = double_quote.unwrap_or(true);
    let skip_initial_space = skip_initial_space.unwrap_or(false);
    // 释放 GIL 进行 CSV 解析和类型推断
    let (headers, cols) = py.detach(|| {
        parse_csv_string(
            content,
            has_header,
            delimiter,
            quote,
            quoting,
            double_quote,
            escape,
            skip_initial_space,
        )
    })?;
    let series_list: Vec<PySeries> = headers
        .par_iter()
        .zip(cols.par_iter())
        .map(|(h, col)| {
            // 空列: 强制为 object dtype (0 长度)
            if col.is_empty() {
                return PySeries {
                    inner: Series::from_options_string(h.clone(), &[]),
                };
            }
            let (dtype, _strings) = infer_column(col);
            let casted = cast_strings(col, dtype);
            let series = match dtype {
                crate::core::dtype::DType::Int64 => {
                    let ints: Vec<Option<i64>> = casted
                        .par_iter()
                        .map(|v| v.as_ref().and_then(|s| s.parse::<i64>().ok()))
                        .collect();
                    Series::from_options_i64(h.clone(), &ints)
                }
                crate::core::dtype::DType::Float64 => {
                    let floats: Vec<Option<f64>> = casted
                        .par_iter()
                        .map(|v| v.as_ref().and_then(|s| s.parse::<f64>().ok()))
                        .collect();
                    Series::from_options_f64(h.clone(), &floats)
                }
                crate::core::dtype::DType::Bool => {
                    let bools: Vec<Option<bool>> = casted
                        .par_iter()
                        .map(|v| {
                            v.as_ref().map(|s| {
                                let sl = s.to_lowercase();
                                sl == "true" || sl == "1"
                            })
                        })
                        .collect();
                    Series::from_options_bool(h.clone(), &bools)
                }
                crate::core::dtype::DType::Object => Series::from_options_string(h.clone(), col),
                crate::core::dtype::DType::Categorical => {
                    Series::from_options_string(h.clone(), col)
                }
            };
            PySeries { inner: series }
        })
        .collect();
    Ok((headers, series_list))
}

/// CSV 字段转义: 如果包含 , " 或换行则用引号包裹，引号转义为 ""
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// 写入 CSV 为字符串（释放 GIL 进行并行序列化）
#[pyfunction]
pub fn write_csv_string<'py>(
    py: Python<'py>,
    columns: Vec<String>,
    series_list: Vec<PySeries>,
    include_header: bool,
    sep: Option<String>,
) -> PyResult<String> {
    let sep_str = sep.unwrap_or_else(|| ",".to_string());
    // 释放 GIL 进行并行 CSV 序列化
    py.detach(|| {
        let mut buf = String::new();
        let sep_char = sep_str.as_str();

        if include_header {
            let escaped: Vec<String> = columns.iter().map(|c| csv_escape(c)).collect();
            buf.push_str(&escaped.join(sep_char));
            buf.push('\n');
        }

        if series_list.is_empty() {
            return Ok(buf);
        }

        let nrows = series_list[0].inner.len();
        // 并行生成每行的 CSV 字符串
        let row_strings: Vec<String> = (0..nrows)
            .into_par_iter()
            .map(|i| {
                let mut record: Vec<String> = Vec::with_capacity(series_list.len());
                for s in &series_list {
                    let v = s.inner.get_str_at(i);
                    record.push(csv_escape(&v));
                }
                record.join(sep_char)
            })
            .collect();
        for row_str in &row_strings {
            buf.push_str(row_str);
            buf.push('\n');
        }
        Ok(buf)
    })
}

/// 解析 CSV 为原始字符串列（不进行类型推断），返回 (headers, 字符串列)。
///
/// 与 read_csv_string 不同，本函数不进行类型推断，适合 Python 侧仍需
/// 做 na_values/converters/parse_dates 等后处理的场景。
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn parse_csv_raw<'py>(
    py: Python<'py>,
    content: &str,
    has_header: bool,
    delimiter: Option<u8>,
    quote: Option<u8>,
    quoting: Option<bool>,
    double_quote: Option<bool>,
    escape: Option<u8>,
    skip_initial_space: Option<bool>,
) -> PyResult<(Vec<String>, Vec<Vec<Option<String>>>)> {
    let delimiter = delimiter.unwrap_or(b',');
    let quote = quote.unwrap_or(b'"');
    let quoting = quoting.unwrap_or(true);
    let double_quote = double_quote.unwrap_or(true);
    let skip_initial_space = skip_initial_space.unwrap_or(false);
    py.detach(|| {
        parse_csv_string(
            content,
            has_header,
            delimiter,
            quote,
            quoting,
            double_quote,
            escape,
            skip_initial_space,
        )
    })
}

/// 从文件路径读取 CSV（释放 GIL 进行文件读取和解析）
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn read_csv_path<'py>(
    py: Python<'py>,
    path: &str,
    has_header: bool,
    delimiter: Option<u8>,
    quote: Option<u8>,
    quoting: Option<bool>,
    double_quote: Option<bool>,
    escape: Option<u8>,
    skip_initial_space: Option<bool>,
) -> PyResult<(Vec<String>, Vec<PySeries>)> {
    let content = py.detach(|| -> PyResult<String> {
        let mut content = String::new();
        let mut file = File::open(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("open {path}: {e}")))?;
        file.read_to_string(&mut content)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("read {path}: {e}")))?;
        Ok(content)
    })?;
    read_csv_string(
        py,
        &content,
        has_header,
        delimiter,
        quote,
        quoting,
        double_quote,
        escape,
        skip_initial_space,
    )
}

/// 写入 CSV 到文件路径（释放 GIL 进行文件写入）
#[pyfunction]
pub fn write_csv_path<'py>(
    py: Python<'py>,
    path: &str,
    columns: Vec<String>,
    series_list: Vec<PySeries>,
    include_header: bool,
    sep: Option<String>,
) -> PyResult<()> {
    let content = write_csv_string(py, columns, series_list, include_header, sep)?;
    py.detach(|| -> PyResult<()> {
        let mut file = File::create(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("create {path}: {e}")))?;
        file.write_all(content.as_bytes())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("write {path}: {e}")))?;
        Ok(())
    })
}

/// 分块读取 CSV：返回每个块的 (headers, PySeries 列表) 列表
/// chunk_size: 每块的行数
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn read_csv_chunks<'py>(
    py: Python<'py>,
    content: &str,
    has_header: bool,
    chunk_size: usize,
    delimiter: Option<u8>,
    quote: Option<u8>,
    quoting: Option<bool>,
    double_quote: Option<bool>,
    escape: Option<u8>,
    skip_initial_space: Option<bool>,
) -> PyResult<Vec<(Vec<String>, Vec<PySeries>)>> {
    if chunk_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "chunk_size must be > 0",
        ));
    }
    let delimiter = delimiter.unwrap_or(b',');
    let quote = quote.unwrap_or(b'"');
    let quoting = quoting.unwrap_or(true);
    let double_quote = double_quote.unwrap_or(true);
    let skip_initial_space = skip_initial_space.unwrap_or(false);
    // 释放 GIL 解析 CSV
    let (headers, cols) = py.detach(|| {
        parse_csv_string(
            content,
            has_header,
            delimiter,
            quote,
            quoting,
            double_quote,
            escape,
            skip_initial_space,
        )
    })?;
    if cols.is_empty() {
        return Ok(Vec::new());
    }
    let n_rows = cols[0].len();
    let mut chunks: Vec<(Vec<String>, Vec<PySeries>)> = Vec::new();
    let mut start = 0;
    while start < n_rows {
        let end = (start + chunk_size).min(n_rows);
        // 切片每列
        let chunk_cols: Vec<Vec<Option<String>>> = cols
            .par_iter()
            .map(|col| col[start..end].to_vec())
            .collect();
        // 类型推断并构建 Series
        let series_list: Vec<PySeries> = headers
            .par_iter()
            .zip(chunk_cols.par_iter())
            .map(|(h, col)| {
                if col.is_empty() {
                    return PySeries {
                        inner: Series::from_options_string(h.clone(), &[]),
                    };
                }
                let (dtype, _strings) = infer_column(col);
                let casted = cast_strings(col, dtype);
                let series = match dtype {
                    DType::Int64 => {
                        let ints: Vec<Option<i64>> = casted
                            .par_iter()
                            .map(|v| v.as_ref().and_then(|s| s.parse::<i64>().ok()))
                            .collect();
                        Series::from_options_i64(h.clone(), &ints)
                    }
                    DType::Float64 => {
                        let floats: Vec<Option<f64>> = casted
                            .par_iter()
                            .map(|v| v.as_ref().and_then(|s| s.parse::<f64>().ok()))
                            .collect();
                        Series::from_options_f64(h.clone(), &floats)
                    }
                    DType::Bool => {
                        let bools: Vec<Option<bool>> = casted
                            .par_iter()
                            .map(|v| {
                                v.as_ref().map(|s| {
                                    let sl = s.to_lowercase();
                                    sl == "true" || sl == "1"
                                })
                            })
                            .collect();
                        Series::from_options_bool(h.clone(), &bools)
                    }
                    DType::Object => Series::from_options_string(h.clone(), col),
                    DType::Categorical => Series::from_options_string(h.clone(), col),
                };
                PySeries { inner: series }
            })
            .collect();
        chunks.push((headers.clone(), series_list));
        start = end;
    }
    Ok(chunks)
}

/// 从文件路径读取 CSV 并返回原始字符串列（不进行类型推断）。
///
/// 在 Rust 层完成文件读取、行级过滤（comment/skip_blank_lines/skiprows/skipfooter）
/// 和 CSV 解析，避免 Python 层 splitlines/join 瓶颈。
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn read_csv_path_raw<'py>(
    py: Python<'py>,
    path: &str,
    has_header: bool,
    delimiter: Option<u8>,
    quote: Option<u8>,
    quoting: Option<bool>,
    double_quote: Option<bool>,
    escape: Option<u8>,
    skip_initial_space: Option<bool>,
    comment: Option<u8>,
    skip_blank_lines: Option<bool>,
    skiprows: Option<Vec<usize>>,
    skipfooter: Option<usize>,
    lineterminator: Option<u8>,
) -> PyResult<(Vec<String>, Vec<Vec<Option<String>>>)> {
    py.detach(|| {
        read_path_to_cols(
            path,
            has_header,
            delimiter,
            quote,
            quoting,
            double_quote,
            escape,
            skip_initial_space,
            comment,
            skip_blank_lines,
            skiprows,
            skipfooter,
            lineterminator,
        )
    })
}

/// 从文件路径读取文件内容为字符串（释放 GIL 进行文件读取）
#[pyfunction]
pub fn read_file_to_string<'py>(py: Python<'py>, path: &str) -> PyResult<String> {
    py.detach(|| -> PyResult<String> {
        let mut content = String::new();
        let mut file = File::open(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("open {path}: {e}")))?;
        file.read_to_string(&mut content)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("read {path}: {e}")))?;
        Ok(content)
    })
}

/// 将字符串写入文件路径（释放 GIL 进行文件写入）
#[pyfunction]
pub fn write_string_to_file<'py>(py: Python<'py>, path: &str, content: &str) -> PyResult<()> {
    py.detach(|| -> PyResult<()> {
        let mut file = File::create(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("create {path}: {e}")))?;
        file.write_all(content.as_bytes())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("write {path}: {e}")))?;
        Ok(())
    })
}

/// 判断字符串是否为 pandas 默认 NaN 值。
fn is_default_na(s: &str) -> bool {
    matches!(
        s,
        "" | "#N/A"
            | "#N/A N/A"
            | "#NA"
            | "-1.#IND"
            | "-1.#QNAN"
            | "-NaN"
            | "-nan"
            | "1.#IND"
            | "1.#QNAN"
            | "<NA>"
            | "N/A"
            | "NA"
            | "NULL"
            | "NaN"
            | "None"
            | "n/a"
            | "nan"
            | "null"
    )
}

/// 将列中的默认 NaN 字符串归一化为 None。
fn normalize_na(values: &mut [Option<String>]) {
    for v in values.iter_mut() {
        if let Some(s) = v
            && is_default_na(s)
        {
            *v = None;
        }
    }
}

/// 判断字符串是否为布尔字面量（与 Python `_infer_column_type` 对齐，不含 "0"/"1"）。
fn is_bool_str(s: &str) -> bool {
    matches!(s, "True" | "TRUE" | "true" | "False" | "FALSE" | "false")
}

/// 读取文件 + 行级过滤 + CSV 解析（纯 Rust，不涉及 GIL）。
///
/// 供 :func:`read_csv_path_raw` 与 :func:`read_csv_path_typed` 复用，
/// 避免重复实现文件读取与 comment/skip_blank_lines/skiprows/skipfooter 过滤逻辑。
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn read_path_to_cols(
    path: &str,
    has_header: bool,
    delimiter: Option<u8>,
    quote: Option<u8>,
    quoting: Option<bool>,
    double_quote: Option<bool>,
    escape: Option<u8>,
    skip_initial_space: Option<bool>,
    comment: Option<u8>,
    skip_blank_lines: Option<bool>,
    skiprows: Option<Vec<usize>>,
    skipfooter: Option<usize>,
    lineterminator: Option<u8>,
) -> PyResult<(Vec<String>, Vec<Vec<Option<String>>>)> {
    // 1. 读取文件内容
    let mut content = String::new();
    let mut file = File::open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("open {path}: {e}")))?;
    file.read_to_string(&mut content)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("read {path}: {e}")))?;

    // 2. 行级过滤
    let delimiter = delimiter.unwrap_or(b',');
    let quote_char = quote.unwrap_or(b'"');
    let quoting_enabled = quoting.unwrap_or(true);
    let double_quote_enabled = double_quote.unwrap_or(true);
    let skip_blank = skip_blank_lines.unwrap_or(true);
    let skip_rows_set: std::collections::HashSet<usize> =
        skiprows.unwrap_or_default().into_iter().collect();
    let skip_footer = skipfooter.unwrap_or(0);

    let mut processed = String::with_capacity(content.len());
    let mut line_start = 0;
    let mut line_idx = 0usize;
    let n = content.len();
    let bytes = content.as_bytes();

    // 计算总行数用于 skipfooter
    let mut total_lines = 0usize;
    for &b in bytes {
        if b == b'\n' {
            total_lines += 1;
        }
    }
    if !content.is_empty() && !content.ends_with('\n') {
        total_lines += 1;
    }

    while line_start < n {
        let mut line_end = line_start;
        while line_end < n && bytes[line_end] != b'\n' {
            if let Some(lt) = lineterminator
                && bytes[line_end] == lt
            {
                break;
            }
            line_end += 1;
        }
        let mut next_start = line_end + 1;
        if next_start < n && bytes[line_end] == b'\r' && bytes[next_start] == b'\n' {
            next_start += 1;
        }

        let keep = if (skip_footer > 0 && line_idx >= total_lines - skip_footer)
            || skip_rows_set.contains(&line_idx)
        {
            false
        } else if skip_blank {
            let is_blank = bytes[line_start..line_end]
                .iter()
                .all(|&b| b.is_ascii_whitespace());
            !is_blank
        } else {
            true
        };

        if keep {
            let end = if let Some(cmt) = comment {
                let mut found = line_end;
                for (i, &b) in bytes[line_start..line_end].iter().enumerate() {
                    if b == cmt {
                        found = line_start + i;
                        break;
                    }
                }
                found
            } else {
                line_end
            };
            let line = &content[line_start..end];
            if let Some(lt) = lineterminator
                && line.ends_with(|c: char| c as u8 == lt)
            {
                processed.push_str(&line[..line.len() - 1]);
            } else {
                processed.push_str(line);
            }
            processed.push('\n');
        }

        line_start = next_start;
        line_idx += 1;
    }

    // 3. 解析 CSV
    parse_csv_string(
        &processed,
        has_header,
        delimiter,
        quote_char,
        quoting_enabled,
        double_quote_enabled,
        escape,
        skip_initial_space.unwrap_or(false),
    )
}

/// 将一列原始字符串做类型推断，返回 (Series, object_type_tags)。
///
/// 推断顺序与 Python `_infer_column_type` 完全一致：int -> float -> bool -> object。
/// 纯类型列 tags 为空；object 列 tags 长度 = 列长，元素编码：
/// 0=缺失值 / 1=int / 2=float / 3=bool / 4=str。
fn infer_typed_col(name: &str, values: &[Option<String>]) -> (Series, Vec<u8>) {
    // 空列 -> object
    if values.is_empty() {
        return (
            Series::from_options_string(name.to_string(), &[]),
            Vec::new(),
        );
    }

    // 全缺失 -> object 字符串列 + 全 0 tags
    if !values.iter().any(|v| v.is_some()) {
        let tags = vec![0u8; values.len()];
        return (Series::from_options_string(name.to_string(), values), tags);
    }

    // 1. 全 int
    let all_int = values.iter().all(|v| match v {
        Some(s) => s.parse::<i64>().is_ok(),
        None => true,
    });
    if all_int {
        let ints: Vec<Option<i64>> = values
            .iter()
            .map(|v| v.as_ref().and_then(|s| s.parse::<i64>().ok()))
            .collect();
        return (
            Series::from_options_i64(name.to_string(), &ints),
            Vec::new(),
        );
    }

    // 2. 全 float
    let all_float = values.iter().all(|v| match v {
        Some(s) => s.parse::<f64>().is_ok(),
        None => true,
    });
    if all_float {
        let floats: Vec<Option<f64>> = values
            .iter()
            .map(|v| v.as_ref().and_then(|s| s.parse::<f64>().ok()))
            .collect();
        return (
            Series::from_options_f64(name.to_string(), &floats),
            Vec::new(),
        );
    }

    // 3. 全 bool
    let all_bool = values.iter().all(|v| match v {
        Some(s) => is_bool_str(s),
        None => true,
    });
    if all_bool {
        let bools: Vec<Option<bool>> = values
            .iter()
            .map(|v| {
                v.as_ref()
                    .map(|s| matches!(s.as_str(), "True" | "TRUE" | "true"))
            })
            .collect();
        return (
            Series::from_options_bool(name.to_string(), &bools),
            Vec::new(),
        );
    }

    // 4. object 列：逐值打类型标签
    let tags: Vec<u8> = values
        .iter()
        .map(|v| match v {
            None => 0,
            Some(s) => {
                if s.parse::<i64>().is_ok() {
                    1
                } else if s.parse::<f64>().is_ok() {
                    2
                } else if is_bool_str(s) {
                    3
                } else {
                    4
                }
            }
        })
        .collect();
    (Series::from_options_string(name.to_string(), values), tags)
}

/// 从文件路径读取 CSV 并在 Rust 层完成类型推断（释放 GIL）。
///
/// 返回 `(headers, series_list, object_tags)`：
/// - `series_list`：纯类型列（int/float/bool）为 typed Series，object 列以字符串存储；
/// - `object_tags`：与 `headers` 对齐，仅 object 列非空，元素为逐值类型标签
///   （0=缺失 / 1=int / 2=float / 3=bool / 4=str），供 Python 层还原混合类型列。
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn read_csv_path_typed<'py>(
    py: Python<'py>,
    path: &str,
    has_header: bool,
    delimiter: Option<u8>,
    quote: Option<u8>,
    quoting: Option<bool>,
    double_quote: Option<bool>,
    escape: Option<u8>,
    skip_initial_space: Option<bool>,
    comment: Option<u8>,
    skip_blank_lines: Option<bool>,
    skiprows: Option<Vec<usize>>,
    skipfooter: Option<usize>,
    lineterminator: Option<u8>,
) -> PyResult<(Vec<String>, Vec<PySeries>, Vec<Vec<u8>>)> {
    let (headers, series_vec, tags_vec) = py.detach(|| -> PyResult<_> {
        let (headers, cols) = read_path_to_cols(
            path,
            has_header,
            delimiter,
            quote,
            quoting,
            double_quote,
            escape,
            skip_initial_space,
            comment,
            skip_blank_lines,
            skiprows,
            skipfooter,
            lineterminator,
        )?;
        let (series_vec, tags_vec): (Vec<Series>, Vec<Vec<u8>>) = headers
            .par_iter()
            .zip(cols.into_par_iter())
            .map(|(h, mut col)| {
                normalize_na(&mut col);
                infer_typed_col(h, &col)
            })
            .unzip();
        Ok((headers, series_vec, tags_vec))
    })?;

    let series_list: Vec<PySeries> = series_vec
        .into_iter()
        .map(|inner| PySeries { inner })
        .collect();
    Ok((headers, series_list, tags_vec))
}
