//! Parquet 文件读写（基于 Apache Arrow Rust 实现，无需 pyarrow 依赖）。
//!
//! - 读取: Parquet → Arrow RecordBatch → ColumnData → PySeries
//! - 写入: PySeries → ColumnData → Arrow Array → RecordBatch → Parquet
//!
//! 本模块还导出公共转换函数（ColumnData ↔ Arrow Array），供 arrow_ipc_io.rs 复用。

use crate::core::dtype::ColumnData;
use crate::core::series::{PySeries, Series};
use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, LargeStringArray, StringArray,
};
use arrow::compute::{cast, concat};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::{BrotliLevel, Compression, GzipLevel, ZstdLevel};
use parquet::file::properties::WriterProperties;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fs::File;
use std::sync::Arc;

// ============================================================================
// 错误转换辅助
// ============================================================================

/// 将 ArrowError 转为 PyValueError
pub(crate) fn arrow_err(e: arrow::error::ArrowError) -> PyErr {
    PyValueError::new_err(format!("arrow error: {e}"))
}

/// 将 ParquetError 转为 PyValueError
pub(crate) fn parquet_err(e: parquet::errors::ParquetError) -> PyErr {
    PyValueError::new_err(format!("parquet error: {e}"))
}

// ============================================================================
// 公共转换：ColumnData ↔ Arrow Array
// ============================================================================

/// 将 ColumnData 转换为 Arrow Array + 对应的 DataType
///
/// 返回 (Array, DataType) 用于构建 Schema 和 RecordBatch
pub(crate) fn column_data_to_arrow(data: &ColumnData) -> (ArrayRef, DataType) {
    match data {
        ColumnData::Int(v) => {
            // Int64Array 实现了 From<Vec<Option<i64>>>
            let arr = Int64Array::from(v.clone());
            (Arc::new(arr), DataType::Int64)
        }
        ColumnData::Float(v) => {
            let arr = Float64Array::from(v.clone());
            (Arc::new(arr), DataType::Float64)
        }
        ColumnData::Bool(v) => {
            let arr = BooleanArray::from(v.clone());
            (Arc::new(arr), DataType::Boolean)
        }
        ColumnData::String(v) => {
            let arr = StringArray::from(v.clone());
            (Arc::new(arr), DataType::Utf8)
        }
        ColumnData::Categorical(c) => {
            // Categorical 展开为 StringArray（通过 categories[codes] 映射）
            let v: Vec<Option<String>> = c
                .codes
                .iter()
                .map(|code| code.and_then(|idx| c.categories.get(idx as usize).cloned()))
                .collect();
            let arr = StringArray::from(v);
            (Arc::new(arr), DataType::Utf8)
        }
    }
}

/// 将 Arrow Array 转换为 ColumnData
///
/// 支持的类型映射:
/// - Int64 → ColumnData::Int
/// - Float64 → ColumnData::Float
/// - Boolean → ColumnData::Bool
/// - Utf8 / LargeUtf8 → ColumnData::String
/// - 其他整数/浮点类型 → cast 后转换
/// - 其他类型 → cast 为 Utf8 后转换
pub(crate) fn arrow_array_to_column_data(array: &dyn Array) -> PyResult<ColumnData> {
    match array.data_type() {
        DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| PyValueError::new_err("downcast Int64Array failed"))?;
            let v: Vec<Option<i64>> = arr.iter().collect();
            Ok(ColumnData::Int(v))
        }
        DataType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| PyValueError::new_err("downcast Float64Array failed"))?;
            let v: Vec<Option<f64>> = arr.iter().collect();
            Ok(ColumnData::Float(v))
        }
        DataType::Boolean => {
            let arr = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| PyValueError::new_err("downcast BooleanArray failed"))?;
            let v: Vec<Option<bool>> = arr.iter().collect();
            Ok(ColumnData::Bool(v))
        }
        DataType::Utf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| PyValueError::new_err("downcast StringArray failed"))?;
            let v: Vec<Option<String>> = arr.iter().map(|s| s.map(|x| x.to_string())).collect();
            Ok(ColumnData::String(v))
        }
        DataType::LargeUtf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| PyValueError::new_err("downcast LargeStringArray failed"))?;
            let v: Vec<Option<String>> = arr.iter().map(|s| s.map(|x| x.to_string())).collect();
            Ok(ColumnData::String(v))
        }
        // 整数类型 → cast 为 Int64
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32 => {
            let casted = cast(array, &DataType::Int64).map_err(arrow_err)?;
            arrow_array_to_column_data(casted.as_ref())
        }
        // Float32 → cast 为 Float64
        DataType::Float32 => {
            let casted = cast(array, &DataType::Float64).map_err(arrow_err)?;
            arrow_array_to_column_data(casted.as_ref())
        }
        // UInt64 → 可能溢出 i64，先 cast 为 Float64 再处理
        DataType::UInt64 => {
            let casted = cast(array, &DataType::Float64).map_err(arrow_err)?;
            arrow_array_to_column_data(casted.as_ref())
        }
        // 其他类型（Date/Timestamp/Decimal/Dictionary 等）→ cast 为 Utf8
        _ => {
            let casted = cast(array, &DataType::Utf8).map_err(arrow_err)?;
            arrow_array_to_column_data(casted.as_ref())
        }
    }
}

/// 根据 Arrow DataType 创建空的 ColumnData（用于空 Parquet 文件）
pub(crate) fn empty_column_data(dtype: &DataType) -> ColumnData {
    match dtype {
        DataType::Int64
        | DataType::Int32
        | DataType::Int16
        | DataType::Int8
        | DataType::UInt64
        | DataType::UInt32
        | DataType::UInt16
        | DataType::UInt8 => ColumnData::Int(vec![]),
        DataType::Float64 | DataType::Float32 => ColumnData::Float(vec![]),
        DataType::Boolean => ColumnData::Bool(vec![]),
        _ => ColumnData::String(vec![]),
    }
}

/// 将 Series 列表转换为 Arrow Schema + RecordBatch
pub(crate) fn series_to_record_batch(
    columns: &[String],
    series: &[Series],
) -> PyResult<(Arc<Schema>, RecordBatch)> {
    if columns.len() != series.len() {
        return Err(PyValueError::new_err(format!(
            "columns len {} != series len {}",
            columns.len(),
            series.len()
        )));
    }
    let mut fields = Vec::with_capacity(columns.len());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(columns.len());
    for (name, s) in columns.iter().zip(series.iter()) {
        let (array, dtype) = column_data_to_arrow(&s.data);
        fields.push(Field::new(name, dtype, true));
        arrays.push(array);
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays).map_err(arrow_err)?;
    Ok((schema, batch))
}

/// 将压缩算法字符串解析为 parquet::basic::Compression
fn parse_compression(s: &str) -> Compression {
    match s.to_lowercase().as_str() {
        "snappy" => Compression::SNAPPY,
        "gzip" => Compression::GZIP(GzipLevel::default()),
        "brotli" => Compression::BROTLI(BrotliLevel::default()),
        "zstd" => Compression::ZSTD(ZstdLevel::default()),
        "lz4" => Compression::LZ4_RAW,
        "none" | "uncompressed" | "" => Compression::UNCOMPRESSED,
        _ => Compression::SNAPPY, // 默认 snappy（与 pandas 默认一致）
    }
}

// ============================================================================
// Parquet 读取
// ============================================================================

/// 从 Parquet 文件读取，返回 (列名列表, PySeries 列表)
///
/// 释放 GIL 进行文件读取和类型转换。
#[pyfunction]
#[pyo3(signature = (path))]
pub fn read_parquet(py: Python<'_>, path: &str) -> PyResult<(Vec<String>, Vec<PySeries>)> {
    let (columns, series_list) = py.detach(|| read_parquet_impl(path))?;
    let py_series_list = series_list
        .into_iter()
        .map(|s| PySeries { inner: s })
        .collect();
    Ok((columns, py_series_list))
}

fn read_parquet_impl(path: &str) -> PyResult<(Vec<String>, Vec<Series>)> {
    let file = File::open(path)
        .map_err(|e| PyValueError::new_err(format!("Failed to open parquet file '{path}': {e}")))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(parquet_err)?;
    let schema = builder.schema().clone();
    let reader = builder.build().map_err(parquet_err)?;

    // 收集所有 batch 的列（Parquet 可能有多个 row group）
    let num_cols = schema.fields().len();
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
            // 空 Parquet 文件：根据 schema 类型创建空列
            empty_column_data(field.data_type())
        } else if chunks.len() == 1 {
            arrow_array_to_column_data(chunks[0].as_ref())?
        } else {
            // 多个 chunk 合并
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
// Parquet 写入
// ============================================================================

/// 将列数据写入 Parquet 文件
///
/// 参数:
/// - path: 输出文件路径
/// - columns: 列名列表
/// - series: PySeries 列表
/// - compression: 压缩算法 (snappy/gzip/brotli/zstd/lz4/none)
#[pyfunction]
#[pyo3(signature = (path, columns, series, compression="snappy"))]
pub fn write_parquet(
    py: Python<'_>,
    path: &str,
    columns: Vec<String>,
    series: Vec<PySeries>,
    compression: &str,
) -> PyResult<()> {
    let inner_series: Vec<Series> = series.into_iter().map(|s| s.inner).collect();
    py.detach(|| write_parquet_impl(path, &columns, &inner_series, compression))
}

fn write_parquet_impl(
    path: &str,
    columns: &[String],
    series: &[Series],
    compression: &str,
) -> PyResult<()> {
    let (schema, batch) = series_to_record_batch(columns, series)?;

    let comp = parse_compression(compression);
    let props = WriterProperties::builder().set_compression(comp).build();

    let file = File::create(path).map_err(|e| {
        PyValueError::new_err(format!("Failed to create parquet file '{path}': {e}"))
    })?;

    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).map_err(parquet_err)?;
    writer.write(&batch).map_err(parquet_err)?;
    writer.close().map_err(parquet_err)?;
    Ok(())
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dtype::ColumnData;
    use crate::core::series::Series;

    #[test]
    fn test_column_data_to_arrow_int() {
        let data = ColumnData::Int(vec![Some(1), None, Some(3)]);
        let (arr, dtype) = column_data_to_arrow(&data);
        assert_eq!(dtype, DataType::Int64);
        assert_eq!(arr.len(), 3);
        let int_arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(int_arr.value(0), 1);
        assert!(int_arr.is_null(1));
        assert_eq!(int_arr.value(2), 3);
    }

    #[test]
    fn test_column_data_to_arrow_float() {
        let data = ColumnData::Float(vec![Some(1.5), None, Some(3.0)]);
        let (arr, dtype) = column_data_to_arrow(&data);
        assert_eq!(dtype, DataType::Float64);
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn test_column_data_to_arrow_bool() {
        let data = ColumnData::Bool(vec![Some(true), None, Some(false)]);
        let (arr, dtype) = column_data_to_arrow(&data);
        assert_eq!(dtype, DataType::Boolean);
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn test_column_data_to_arrow_string() {
        let data = ColumnData::String(vec![Some("a".to_string()), None, Some("b".to_string())]);
        let (arr, dtype) = column_data_to_arrow(&data);
        assert_eq!(dtype, DataType::Utf8);
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn test_arrow_array_to_column_data_roundtrip() {
        // Int 往返
        let original = ColumnData::Int(vec![Some(1), None, Some(3)]);
        let (arr, _) = column_data_to_arrow(&original);
        let recovered = arrow_array_to_column_data(arr.as_ref()).unwrap();
        if let (ColumnData::Int(orig), ColumnData::Int(rec)) = (&original, &recovered) {
            assert_eq!(orig, rec);
        } else {
            panic!("type mismatch");
        }

        // Float 往返
        let original = ColumnData::Float(vec![Some(1.5), None, Some(3.0)]);
        let (arr, _) = column_data_to_arrow(&original);
        let recovered = arrow_array_to_column_data(arr.as_ref()).unwrap();
        if let (ColumnData::Float(orig), ColumnData::Float(rec)) = (&original, &recovered) {
            assert_eq!(orig, rec);
        } else {
            panic!("type mismatch");
        }

        // String 往返
        let original = ColumnData::String(vec![Some("a".to_string()), None, Some("b".to_string())]);
        let (arr, _) = column_data_to_arrow(&original);
        let recovered = arrow_array_to_column_data(arr.as_ref()).unwrap();
        if let (ColumnData::String(orig), ColumnData::String(rec)) = (&original, &recovered) {
            assert_eq!(orig, rec);
        } else {
            panic!("type mismatch");
        }
    }

    #[test]
    fn test_parse_compression() {
        assert_eq!(parse_compression("snappy"), Compression::SNAPPY);
        assert_eq!(
            parse_compression("gzip"),
            Compression::GZIP(GzipLevel::default())
        );
        assert_eq!(
            parse_compression("zstd"),
            Compression::ZSTD(ZstdLevel::default())
        );
        assert_eq!(parse_compression("none"), Compression::UNCOMPRESSED);
        assert_eq!(parse_compression(""), Compression::UNCOMPRESSED);
        assert_eq!(parse_compression("unknown"), Compression::SNAPPY);
    }

    #[test]
    fn test_series_to_record_batch() {
        let columns = vec!["a".to_string(), "b".to_string()];
        let series = vec![
            Series::new_int(Some("a".to_string()), vec![Some(1), Some(2)]),
            Series::new_float(Some("b".to_string()), vec![Some(1.5), None]),
        ];
        let (schema, batch) = series_to_record_batch(&columns, &series).unwrap();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);
    }
}
