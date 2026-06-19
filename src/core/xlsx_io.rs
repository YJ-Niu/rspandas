//! Excel (.xlsx) 读写，使用 calamine + rust_xlsxwriter。
//!
//! - 读取: calamine 支持 .xlsx / .xls / .ods 等格式
//! - 写入: rust_xlsxwriter 生成 .xlsx 文件

use std::collections::HashMap;
use pyo3::prelude::*;
use rayon::prelude::*;
use calamine::Reader;
use crate::core::series::{PySeries, Series};

/// 从 xlsx 文件读取数据，返回 (列名, 列数据)。
/// 返回格式: Vec<(col_name, Vec<Option<String>>)>
fn read_xlsx_raw(
    path: &str,
    sheet_name: Option<&str>,
    sheet_index: Option<usize>,
    header_row: usize,
) -> PyResult<(Vec<String>, Vec<Vec<Option<String>>>)> {
    let mut workbook = calamine::open_workbook_auto(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("open xlsx {path}: {e}"))
    })?;

    let target_sheet = if let Some(name) = sheet_name {
        name.to_string()
    } else {
        let idx = sheet_index.unwrap_or(0);
        let names = workbook.sheet_names();
        if idx >= names.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "sheet index {idx} out of range, max {}",
                names.len()
            )));
        }
        names[idx].to_string()
    };

    let range = workbook
        .worksheet_range(&target_sheet)
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "sheet '{target_sheet}' not found: {e}"
            ))
        })?;

    let rows: Vec<Vec<calamine::Data>> = range.rows().map(|r| r.to_vec()).collect();
    if rows.is_empty() {
        return Ok((vec![], vec![]));
    }

    let max_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if max_cols == 0 {
        return Ok((vec![], vec![]));
    }

    // 确定列名
    let col_names: Vec<String> = if header_row < rows.len() {
        let header = &rows[header_row];
        (0..max_cols)
            .map(|i| {
                if i < header.len() {
                    match &header[i] {
                        calamine::Data::String(s) => s.clone(),
                        calamine::Data::Float(f) => f.to_string(),
                        calamine::Data::Int(i) => i.to_string(),
                        calamine::Data::Bool(b) => b.to_string(),
                        calamine::Data::Empty => format!("col{i}"),
                        calamine::Data::DateTime(d) => format!("{d}"),
                        calamine::Data::DateTimeIso(s) => s.clone(),
                        calamine::Data::DurationIso(s) => s.clone(),
                        _ => format!("col{i}"),
                    }
                } else {
                    format!("col{i}")
                }
            })
            .collect()
    } else {
        (0..max_cols).map(|i| format!("col{i}")).collect()
    };

    // 去重列名
    let mut seen: HashMap<String, usize> = HashMap::new();
    let final_cols: Vec<String> = col_names
        .iter()
        .map(|c| {
            let count = seen.entry(c.clone()).or_insert(0);
            if *count == 0 {
                *count += 1;
                c.clone()
            } else {
                *count += 1;
                format!("{c}.{}", *count - 1)
            }
        })
        .collect();

    let data_start = if header_row < rows.len() {
        header_row + 1
    } else {
        0
    };

    // 构建列数据
    let mut cols: Vec<Vec<Option<String>>> = vec![Vec::new(); max_cols];
    for row in rows.iter().skip(data_start) {
        for i in 0..max_cols {
            let val = if i < row.len() {
                match &row[i] {
                    calamine::Data::Empty => None,
                    calamine::Data::String(s) => Some(s.clone()),
                    calamine::Data::Float(f) => Some(f.to_string()),
                    calamine::Data::Int(i) => Some(i.to_string()),
                    calamine::Data::Bool(b) => Some(b.to_string()),
                    calamine::Data::DateTime(d) => Some(format!("{d}")),
                    calamine::Data::DateTimeIso(s) => Some(s.clone()),
                    calamine::Data::DurationIso(s) => Some(s.clone()),
                    _ => None,
                }
            } else {
                None
            };
            cols[i].push(val);
        }
    }

    Ok((final_cols, cols))
}

/// 推断字符串列的类型并转换为对应的 Series (并行化)
fn strings_to_series(name: &str, values: &[Option<String>]) -> PySeries {
    // 并行推断类型
    let (all_int, all_float, all_bool, any_non_null) = values.par_iter().map(|v| {
        match v {
            Some(s) => {
                let int_ok = s.parse::<i64>().is_ok();
                let float_ok = s.parse::<f64>().is_ok();
                let sl = s.to_lowercase();
                let bool_ok = sl == "true" || sl == "false" || sl == "0" || sl == "1";
                (int_ok, float_ok, bool_ok, true)
            }
            None => (true, true, true, false),
        }
    }).reduce(
        || (true, true, true, false),
        |(a_int, a_float, a_bool, a_any), (b_int, b_float, b_bool, b_any)| {
            (a_int && b_int, a_float && b_float, a_bool && b_bool, a_any || b_any)
        }
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

/// 读取 xlsx 文件，返回 DataFrame 的列名和 Series 列表。
#[pyfunction]
pub fn read_xlsx(
    path: &str,
    sheet_name: Option<&str>,
    sheet_index: Option<usize>,
    header_row: usize,
) -> PyResult<(Vec<String>, Vec<PySeries>)> {
    let (col_names, cols) = read_xlsx_raw(path, sheet_name, sheet_index, header_row)?;

    let series_list: Vec<PySeries> = col_names
        .par_iter()
        .zip(cols.par_iter())
        .map(|(name, values)| strings_to_series(name, values))
        .collect();

    Ok((col_names, series_list))
}

/// 获取 xlsx 文件的工作表名称列表。
#[pyfunction]
pub fn xlsx_sheet_names(path: &str) -> PyResult<Vec<String>> {
    let workbook = calamine::open_workbook_auto(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("open xlsx {path}: {e}"))
    })?;
    Ok(workbook.sheet_names().to_vec())
}

/// 将 DataFrame 写入 xlsx 文件。
///
/// columns: 列名列表
/// series_list: 每列对应的 PySeries
/// sheet_name: 工作表名称
/// include_header: 是否写入表头
/// include_index: 是否写入行索引
#[pyfunction]
pub fn write_xlsx(
    path: &str,
    columns: Vec<String>,
    series_list: Vec<PySeries>,
    sheet_name: &str,
    include_header: bool,
    include_index: bool,
) -> PyResult<()> {
    use rust_xlsxwriter::*;

    let mut workbook = Workbook::new();
    let worksheet = workbook.add_worksheet();

    worksheet.set_name(sheet_name).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("set sheet name: {e}"))
    })?;

    let col_offset: usize = if include_index { 1 } else { 0 };

    // 写入表头
    let mut row: u32 = 0;
    if include_header {
        for (j, col_name) in columns.iter().enumerate() {
            worksheet
                .write_string(row, (j + col_offset) as u16, col_name)
                .map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!("write header: {e}"))
                })?;
        }
        row += 1;
    }

    // 写入数据
    if !series_list.is_empty() {
        let nrows = series_list[0].inner.len();
        for i in 0..nrows {
            if include_index {
                worksheet
                    .write(row, 0, i as u64)
                    .map_err(|e| {
                        pyo3::exceptions::PyIOError::new_err(format!("write index: {e}"))
                    })?;
            }
            for (j, s) in series_list.iter().enumerate() {
                let val = s.inner.get_str_at(i);
                let col = (j + col_offset) as u16;
                // 尝试写入数字
                let write_result = if let Ok(i64_val) = val.parse::<i64>() {
                    worksheet.write(row, col, i64_val)
                } else if let Ok(f64_val) = val.parse::<f64>() {
                    worksheet.write(row, col, f64_val)
                } else if val == "NaN" || val.is_empty() {
                    worksheet.write_blank(row, col, &Format::new())
                } else {
                    worksheet.write_string(row, col, &val)
                };
                write_result.map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!("write cell: {e}"))
                })?;
            }
            row += 1;
        }
    }

    workbook.save(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("save xlsx {path}: {e}"))
    })?;

    Ok(())
}