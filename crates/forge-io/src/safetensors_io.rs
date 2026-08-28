use anyhow::Result;
use std::path::Path;
use crate::tensor_store::TensorStore;
use crate::streaming_writer::StreamingWriter;
use forge_core::DType;

/// SafeTensors read/write operations
pub fn load_model(path: &Path) -> Result<TensorStore> {
    TensorStore::open(path)
}

pub fn write_model(
    writer: &mut StreamingWriter,
    store: &TensorStore,
    dtype: DType,
) -> Result<usize> {
    let mut count = 0;
    for name in store.tensor_names() {
        let meta = store.tensor_meta(name)?;
        let bytes = store.tensor_bytes(name)?;

        let dtype_str = match dtype {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            _ => "F16",
        };

        writer.write_tensor(name, bytes, dtype_str, &meta.shape)?;
        count += 1;
    }
    Ok(count)
}
