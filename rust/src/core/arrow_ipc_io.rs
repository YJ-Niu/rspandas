//! Feather (Arrow IPC) 文件读写。
//!
//! Feather 是 Apache Arrow 的 IPC 文件格式，本模块基于 arrow::ipc 实现，
//! 无需 pyarrow 依赖。复用 parquet_io 中的公共转换函数。

use crate::core::parquet_io::{
    arrow_array_to_column_data, arrow_err, empty_column_data, series_to_record_batch,
};
use crate::core::series::{PySeries, Series};
use arrow::array::{Array, ArrayRef};
use arrow::compute::concat;
use arrow::ipc::reader::{FileReader, StreamReader};
use arrow::ipc::writer::{FileWriter, StreamWriter};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fs::File;
use std::io::Cursor;

// ============================================================================
// Feather 读取
// ============================================================================

/// 从 Feather (Arrow IPC) 文件读取，返回 (列名列表, PySeries 列表)
#[pyfunction]
#[pyo3(signature = (path))]
pub fn read_feather(py: Python<'_>, path: &str) -> PyResult<(Vec<String>, Vec<PySeries>)> {
    let (columns, series_list) = py.detach(|| read_feather_impl(path))?;
    let py_series_list = series_list
        .into_iter()
        .map(|s| PySeries { inner: s })
        .collect();
    Ok((columns, py_series_list))
}

fn read_feather_impl(path: &str) -> PyResult<(Vec<String>, Vec<Series>)> {
    let file = File::open(path)
        .map_err(|e| PyValueError::new_err(format!("Failed to open feather file '{path}': {e}")))?;

    let reader = FileReader::try_new(file, None).map_err(arrow_err)?;
    let schema = reader.schema().clone();
    let num_cols = schema.fields().len();

    // 收集所有 batch 的列（Feather 可能有多个 RecordBatch）
    let mut column_chunks: Vec<Vec<ArrayRef>> = vec![Vec::new(); num_cols];

    for batch_result in reader {
        let batch = batch_result.map_err(arrow_err)?;
        for (i, col) in batch.columns().iter().enumerate() {
            if i < num_cols {
                column_chunks[i].push(col.clone());
            }
        }
    }

    // 合并每列的 chunks 并转换为 ColumnData
    let mut series_list: Vec<Series> = Vec::with_capacity(num_cols);
    for (i, field) in schema.fields().iter().enumerate() {
        let chunks = &column_chunks[i];
        let column_data = if chunks.is_empty() {
            empty_column_data(field.data_type())
        } else if chunks.len() == 1 {
            arrow_array_to_column_data(chunks[0].as_ref())?
        } else {
            let arrays: Vec<&dyn Array> = chunks.iter().map(|a| a.as_ref()).collect();
            let merged = concat(&arrays).map_err(arrow_err)?;
            arrow_array_to_column_data(merged.as_ref())?
        };
        series_list.push(Series {
            name: Some(field.name().clone()),
            data: column_data,
        });
    }

    let columns: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    Ok((columns, series_list))
}

// ============================================================================
// Feather 写入
// ============================================================================

/// 将列数据写入 Feather (Arrow IPC) 文件
///
/// 参数:
/// - path: 输出文件路径
/// - columns: 列名列表
/// - series: PySeries 列表
/// - compression: 压缩算法 (lz4/zstd/uncompressed)，当前实现统一用 uncompressed（IPC v1 限制）
#[pyfunction]
#[pyo3(signature = (path, columns, series, compression="uncompressed"))]
pub fn write_feather(
    py: Python<'_>,
    path: &str,
    columns: Vec<String>,
    series: Vec<PySeries>,
    compression: &str,
) -> PyResult<()> {
    let inner_series: Vec<Series> = series.into_iter().map(|s| s.inner).collect();
    py.detach(|| write_feather_impl(path, &columns, &inner_series, compression))
}

fn write_feather_impl(
    path: &str,
    columns: &[String],
    series: &[Series],
    compression: &str,
) -> PyResult<()> {
    let (schema, batch) = series_to_record_batch(columns, series)?;

    // Arrow IPC v1 不支持压缩；若用户指定压缩，记录警告但使用 uncompressed
    // （Arrow IPC v5 支持 lz4/zstd，但 arrow 55 的 FileWriter 默认用 v1）
    if compression != "uncompressed" && !compression.is_empty() {
        // 静默降级：pandas 的 to_feather 也只在某些版本支持 IPC 压缩
        eprintln!(
            "warning: feather IPC compression '{compression}' not supported, using uncompressed"
        );
    }

    let file = File::create(path).map_err(|e| {
        PyValueError::new_err(format!("Failed to create feather file '{path}': {e}"))
    })?;

    let mut writer = FileWriter::try_new(file, schema.as_ref()).map_err(arrow_err)?;
    writer.write(&batch).map_err(arrow_err)?;
    writer.finish().map_err(arrow_err)?;
    Ok(())
}

// ============================================================================
// Arrow IPC bytes 桥接（用于 to_arrow / from_arrow）
//
// 替代 Python 层逐元素 list 中转的慢路径：
// - to_arrow(): Rust 层将列数据序列化为 IPC 流格式 bytes，
//   Python 层再用 pyarrow.ipc.open_stream 反序列化为 pyarrow.Table。
// - from_arrow(): Python 层用 pyarrow.ipc 将 Table 序列化为 bytes，
//   Rust 层反序列化为 Series 列表。
//
// 选择 IPC 流格式（StreamWriter）而非文件格式（FileWriter）：
// - 流格式无需 seek，可写入任意 Write 目标（含 Vec<u8>）
// - pyarrow.ipc.open_stream 原生支持，读取更简单
// ============================================================================

/// 将列数据序列化为 Arrow IPC 流格式 bytes
///
/// 释放 GIL 进行序列化。返回的 bytes 可由 `pyarrow.ipc.open_stream(BytesIO(bytes))`
/// 反序列化为 pyarrow.Table，避免 Python 层逐元素类型推断循环。
#[pyfunction]
#[pyo3(signature = (columns, series))]
pub fn to_arrow_ipc_bytes(
    py: Python<'_>,
    columns: Vec<String>,
    series: Vec<PySeries>,
) -> PyResult<Vec<u8>> {
    let inner_series: Vec<Series> = series.into_iter().map(|s| s.inner).collect();
    py.detach(|| to_arrow_ipc_bytes_impl(&columns, &inner_series))
}

fn to_arrow_ipc_bytes_impl(columns: &[String], series: &[Series]) -> PyResult<Vec<u8>> {
    let (schema, batch) = series_to_record_batch(columns, series)?;

    let mut buffer: Vec<u8> = Vec::new();
    let mut writer = StreamWriter::try_new(&mut buffer, schema.as_ref()).map_err(arrow_err)?;
    writer.write(&batch).map_err(arrow_err)?;
    writer.finish().map_err(arrow_err)?;
    Ok(buffer)
}

/// 从 Arrow IPC 流格式 bytes 反序列化，返回 (列名列表, PySeries 列表)
///
/// 释放 GIL 进行反序列化。bytes 可由 `pyarrow.ipc.new_stream` 序列化
/// pyarrow.Table 得到，避免 Python 层 `to_pylist()` 中转。
#[pyfunction]
#[pyo3(signature = (bytes))]
pub fn from_arrow_ipc_bytes(
    py: Python<'_>,
    bytes: &[u8],
) -> PyResult<(Vec<String>, Vec<PySeries>)> {
    let (columns, series_list) = py.detach(|| from_arrow_ipc_bytes_impl(bytes))?;
    let py_series_list = series_list
        .into_iter()
        .map(|s| PySeries { inner: s })
        .collect();
    Ok((columns, py_series_list))
}

fn from_arrow_ipc_bytes_impl(bytes: &[u8]) -> PyResult<(Vec<String>, Vec<Series>)> {
    let cursor = Cursor::new(bytes);
    let reader = StreamReader::try_new(cursor, None).map_err(arrow_err)?;
    let schema = reader.schema().clone();
    let num_cols = schema.fields().len();

    // 收集所有 batch 的列（IPC 流可能包含多个 RecordBatch）
    let mut column_chunks: Vec<Vec<ArrayRef>> = vec![Vec::new(); num_cols];

    for batch_result in reader {
        let batch = batch_result.map_err(arrow_err)?;
        for (i, col) in batch.columns().iter().enumerate() {
            if i < num_cols {
                column_chunks[i].push(col.clone());
            }
        }
    }

    // 合并每列的 chunks 并转换为 ColumnData
    let mut series_list: Vec<Series> = Vec::with_capacity(num_cols);
    for (i, field) in schema.fields().iter().enumerate() {
        let chunks = &column_chunks[i];
        let column_data = if chunks.is_empty() {
            empty_column_data(field.data_type())
        } else if chunks.len() == 1 {
            arrow_array_to_column_data(chunks[0].as_ref())?
        } else {
            let arrays: Vec<&dyn Array> = chunks.iter().map(|a| a.as_ref()).collect();
            let merged = concat(&arrays).map_err(arrow_err)?;
            arrow_array_to_column_data(merged.as_ref())?
        };
        series_list.push(Series {
            name: Some(field.name().clone()),
            data: column_data,
        });
    }

    let columns: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    Ok((columns, series_list))
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dtype::ColumnData;
    use crate::core::series::Series;
    use std::env::temp_dir;

    #[test]
    fn test_feather_roundtrip_int_float() {
        let path = temp_dir().join("rspandas_test_feather_int_float.arrow");
        let columns = vec!["a".to_string(), "b".to_string()];
        let series = vec![
            Series::new_int(Some("a".to_string()), vec![Some(1), None, Some(3)]),
            Series::new_float(Some("b".to_string()), vec![Some(1.5), Some(2.5), None]),
        ];

        // 写入
        write_feather_impl(path.to_str().unwrap(), &columns, &series, "uncompressed").unwrap();

        // 读取
        let (read_cols, read_series) = read_feather_impl(path.to_str().unwrap()).unwrap();
        assert_eq!(read_cols, columns);
        assert_eq!(read_series.len(), 2);

        // 验证 Int 列
        if let ColumnData::Int(v) = &read_series[0].data {
            assert_eq!(v, &vec![Some(1), None, Some(3)]);
        } else {
            panic!("expected Int column");
        }

        // 验证 Float 列
        if let ColumnData::Float(v) = &read_series[1].data {
            assert_eq!(v, &vec![Some(1.5), Some(2.5), None]);
        } else {
            panic!("expected Float column");
        }

        // 清理
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_feather_roundtrip_string_bool() {
        let path = temp_dir().join("rspandas_test_feather_string_bool.arrow");
        let columns = vec!["s".to_string(), "f".to_string()];
        let series = vec![
            Series::new_string(
                Some("s".to_string()),
                vec![Some("a".to_string()), None, Some("b".to_string())],
            ),
            Series::new_bool(Some("f".to_string()), vec![Some(true), Some(false), None]),
        ];

        write_feather_impl(path.to_str().unwrap(), &columns, &series, "uncompressed").unwrap();

        let (read_cols, read_series) = read_feather_impl(path.to_str().unwrap()).unwrap();
        assert_eq!(read_cols, columns);

        if let ColumnData::String(v) = &read_series[0].data {
            assert_eq!(v, &vec![Some("a".to_string()), None, Some("b".to_string())]);
        } else {
            panic!("expected String column");
        }

        if let ColumnData::Bool(v) = &read_series[1].data {
            assert_eq!(v, &vec![Some(true), Some(false), None]);
        } else {
            panic!("expected Bool column");
        }

        let _ = std::fs::remove_file(&path);
    }

    // ========================================================================
    // Arrow IPC bytes 桥接测试
    // ========================================================================

    #[test]
    fn test_ipc_bytes_roundtrip_int_float() {
        let columns = vec!["a".to_string(), "b".to_string()];
        let series = vec![
            Series::new_int(Some("a".to_string()), vec![Some(1), None, Some(3)]),
            Series::new_float(Some("b".to_string()), vec![Some(1.5), Some(2.5), None]),
        ];

        let bytes = to_arrow_ipc_bytes_impl(&columns, &series).unwrap();
        assert!(!bytes.is_empty(), "IPC bytes 不应为空");

        let (read_cols, read_series) = from_arrow_ipc_bytes_impl(&bytes).unwrap();
        assert_eq!(read_cols, columns);
        assert_eq!(read_series.len(), 2);

        if let ColumnData::Int(v) = &read_series[0].data {
            assert_eq!(v, &vec![Some(1), None, Some(3)]);
        } else {
            panic!("expected Int column");
        }

        if let ColumnData::Float(v) = &read_series[1].data {
            assert_eq!(v, &vec![Some(1.5), Some(2.5), None]);
        } else {
            panic!("expected Float column");
        }
    }

    #[test]
    fn test_ipc_bytes_roundtrip_string_bool() {
        let columns = vec!["s".to_string(), "f".to_string()];
        let series = vec![
            Series::new_string(
                Some("s".to_string()),
                vec![Some("a".to_string()), None, Some("b".to_string())],
            ),
            Series::new_bool(Some("f".to_string()), vec![Some(true), Some(false), None]),
        ];

        let bytes = to_arrow_ipc_bytes_impl(&columns, &series).unwrap();
        let (read_cols, read_series) = from_arrow_ipc_bytes_impl(&bytes).unwrap();
        assert_eq!(read_cols, columns);

        if let ColumnData::String(v) = &read_series[0].data {
            assert_eq!(v, &vec![Some("a".to_string()), None, Some("b".to_string())]);
        } else {
            panic!("expected String column");
        }

        if let ColumnData::Bool(v) = &read_series[1].data {
            assert_eq!(v, &vec![Some(true), Some(false), None]);
        } else {
            panic!("expected Bool column");
        }
    }

    #[test]
    fn test_ipc_bytes_roundtrip_empty() {
        // 空列往返（0 行）
        let columns = vec!["a".to_string()];
        let series = vec![Series::new_int(Some("a".to_string()), vec![])];

        let bytes = to_arrow_ipc_bytes_impl(&columns, &series).unwrap();
        let (read_cols, read_series) = from_arrow_ipc_bytes_impl(&bytes).unwrap();
        assert_eq!(read_cols, columns);
        assert_eq!(read_series.len(), 1);

        if let ColumnData::Int(v) = &read_series[0].data {
            assert!(v.is_empty());
        } else {
            panic!("expected Int column");
        }
    }
}
