//! HTML / XML 读写。
//!
//! - HTML 序列化: 将 DataFrame 转为 HTML 表格字符串
//! - XML 序列化: 将 DataFrame 转为 XML 字符串
//! - 纯 Rust 实现，释放 GIL 进行字符串构建

use crate::core::series::PySeries;
use pyo3::prelude::*;

/// 将 DataFrame 转为 HTML 表格字符串。
///
/// 释放 GIL 进行字符串构建，避免 Python 层的循环开销。
#[pyfunction]
pub fn to_html(
    py: Python<'_>,
    columns: Vec<String>,
    series_list: Vec<PySeries>,
    index: bool,
) -> PyResult<String> {
    py.detach(|| {
        let mut html = String::new();
        html.push_str("<table border=\"1\">\n");

        // 表头
        html.push_str("<tr>\n");
        if index {
            html.push_str("<th></th>\n");
        }
        for col in &columns {
            html.push_str(&format!("<th>{col}</th>\n"));
        }
        html.push_str("</tr>\n");

        // 数据行
        if !series_list.is_empty() {
            let nrows = series_list[0].inner.len();
            for i in 0..nrows {
                html.push_str("<tr>\n");
                if index {
                    html.push_str(&format!("<td>{i}</td>\n"));
                }
                for s in &series_list {
                    let val = s.inner.get_str_at(i);
                    let display = if val.is_empty() { "" } else { &val };
                    html.push_str(&format!("<td>{display}</td>\n"));
                }
                html.push_str("</tr>\n");
            }
        }

        html.push_str("</table>\n");
        Ok(html)
    })
}

/// 将 DataFrame 转为 XML 字符串。
///
/// 释放 GIL 进行字符串构建，避免 Python 层的循环开销。
#[pyfunction]
pub fn to_xml(
    py: Python<'_>,
    columns: Vec<String>,
    series_list: Vec<PySeries>,
    index: bool,
    root_name: &str,
    row_name: &str,
) -> PyResult<String> {
    let root = root_name.to_string();
    let row = row_name.to_string();

    py.detach(move || {
        let mut xml = String::new();
        xml.push_str("<?xml version='1.0' encoding='utf-8'?>\n");
        xml.push_str(&format!("<{root}>\n"));

        if !series_list.is_empty() {
            let nrows = series_list[0].inner.len();
            for i in 0..nrows {
                xml.push_str(&format!("  <{row}"));
                if index {
                    xml.push_str(&format!(" index=\"{i}\""));
                }
                xml.push_str(">\n");

                for (j, col_name) in columns.iter().enumerate() {
                    let val = series_list[j].inner.get_str_at(i);
                    let display = if val.is_empty() { "" } else { &val };
                    xml.push_str(&format!("    <{col_name}>{display}</{col_name}>\n"));
                }

                xml.push_str(&format!("  </{row}>\n"));
            }
        }

        xml.push_str(&format!("</{root}>\n"));
        Ok(xml)
    })
}
