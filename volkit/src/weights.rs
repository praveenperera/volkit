//! A model's weights from a safetensors export, with a key-for-key schema check
//!
//! The export holds the network's inference state dict, one tensor per storage; names that alias another tensor
//! (`all_modules.*`, `decoder.encoder.*`) are listed in the metadata. The model takes every tensor it needs by
//! name and shape through [`Checkpoint::take`]; [`Checkpoint::finish`] then fails unless every exported tensor was
//! taken exactly once and every alias points at one of them
//!
//! The export comes from inkrs's `export_nnunet.py`, or from an nnU-Net folder read without torch by
//! [`crate::nnunet::export`], which builds the same tensors in memory ([`Checkpoint::from_parts`])

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use color_eyre::eyre::{Result, WrapErr, bail, ensure};
use safetensors::{Dtype, SafeTensors};

use crate::ensure_supported;
use crate::error::Unsupported;

/// Tensors by name: shape and float32 data
pub type Tensors = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

/// The exported tensors, all float32
pub struct Checkpoint {
    tensors: Tensors,
    aliases: BTreeMap<String, String>,
    taken: RefCell<BTreeSet<String>>,
}

impl Checkpoint {
    /// The tensors `name -> (shape, float32 data)` and aliases `alias -> tensor` of an export, such as one built in
    /// memory: every tensor finite and its data its shape's size, every alias pointing at a tensor and not one itself
    pub fn from_parts(tensors: Tensors, aliases: BTreeMap<String, String>) -> Result<Self> {
        for (name, (shape, data)) in &tensors {
            ensure!(shape.iter().product::<usize>() == data.len(), "{name}: {} values for shape {shape:?}", data.len());
            ensure!(data.iter().all(|v| v.is_finite()), "{name}: non-finite weights");
        }
        for (alias, target) in &aliases {
            ensure!(tensors.contains_key(target), "alias {alias} points at missing {target}");
            ensure!(!tensors.contains_key(alias), "alias {alias} is also a tensor");
        }
        Ok(Self { tensors, aliases, taken: RefCell::new(BTreeSet::new()) })
    }

    /// Loads `path` (`.safetensors`) and its `.json` sidecar with no model contract: every tensor
    /// float32 and finite, and the sidecar's `keys` and `aliases` equal to the file's. Returns the
    /// sidecar for the caller's contract checks. The tensors are copied out of a memory map of the file, which
    /// must not be truncated or rewritten in place while this runs (replace it by renaming instead)
    pub fn read(path: &Path) -> Result<(Self, serde_json::Value)> {
        // mapped rather than read: the tensors are decoded straight from the page cache, without a
        // 138 MB copy first (about 65 ms of the load)
        let file = std::fs::File::open(path).wrap_err_with(|| format!("open {}", path.display()))?;
        // SAFETY: the map lives only for this call, and every byte is decoded into owned tensors
        // before it ends; a file truncated meanwhile is the same failure as reading it mid-write
        let bytes = unsafe { memmap2::Mmap::map(&file) }.wrap_err_with(|| format!("map {}", path.display()))?;
        let sidecar_path = path.with_extension("json");
        let sidecar: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&sidecar_path).wrap_err_with(|| format!("read {}", sidecar_path.display()))?,
        )
        .wrap_err_with(|| format!("parse {}", sidecar_path.display()))?;

        let st = SafeTensors::deserialize(&bytes).wrap_err("parse the safetensors header")?;
        let mut tensors = BTreeMap::new();
        for (name, view) in st.tensors() {
            ensure_supported!(view.dtype() == Dtype::F32, "{name}: dtype {:?}, expected F32", view.dtype());
            let data: Vec<f32> = view.data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
            tensors.insert(name, (view.shape().to_vec(), data));
        }

        let (_, header) = SafeTensors::read_metadata(&bytes).wrap_err("read the safetensors metadata")?;
        let aliases: BTreeMap<String, String> = match header.metadata().as_ref().and_then(|m| m.get("aliases")) {
            Some(json) => serde_json::from_str(json).wrap_err("parse the alias metadata")?,
            None => bail!("{}: no alias metadata; not an export of export_nnunet.py", path.display()),
        };

        let ck = Self::with_sidecar(tensors, aliases, &sidecar).wrap_err_with(|| sidecar_path.display().to_string())?;
        Ok((ck, sidecar))
    }

    /// [`Self::from_parts`], checked against the export's sidecar as [`Self::read`] checks a file: the sidecar's
    /// `keys` (names and shapes) and `aliases` must be the tensors'
    pub fn with_sidecar(
        tensors: Tensors,
        aliases: BTreeMap<String, String>,
        sidecar: &serde_json::Value,
    ) -> Result<Self> {
        let side_keys: BTreeMap<String, Vec<usize>> =
            serde_json::from_value(sidecar["keys"].clone()).wrap_err("the sidecar's keys")?;
        let side_aliases: BTreeMap<String, String> =
            serde_json::from_value(sidecar["aliases"].clone()).wrap_err("the sidecar's aliases")?;
        let ck = Self::from_parts(tensors, aliases)?;
        // the sidecar and the tensors must describe the same tensors
        ensure!(ck.keys() == side_keys, "the sidecar's keys and shapes differ from the tensor file's");
        ensure!(ck.aliases == side_aliases, "the sidecar's aliases differ from the tensor file's");
        Ok(ck)
    }

    fn keys(&self) -> BTreeMap<String, Vec<usize>> {
        self.tensors.iter().map(|(k, (s, _))| (k.clone(), s.clone())).collect()
    }

    /// The tensor `key`, which must have `shape` and not have been taken before
    pub fn take(&self, key: &str, shape: &[usize]) -> Result<Vec<f32>> {
        // a missing or reshaped tensor means another architecture than the one this engine builds
        let Some((have, data)) = self.tensors.get(key) else {
            return Err(Unsupported(format!(
                "checkpoint has no tensor {key}; another architecture than the program's"
            ))
            .into());
        };
        ensure_supported!(have == shape, "{key}: shape {have:?}, the model needs {shape:?}");
        ensure!(self.taken.borrow_mut().insert(key.to_owned()), "{key} taken twice");
        Ok(data.clone())
    }

    /// Fails unless every tensor was taken; returns (tensors, aliases, parameters)
    pub fn finish(&self) -> Result<(usize, usize, usize)> {
        let taken = self.taken.borrow();
        let unused: Vec<&String> = self.tensors.keys().filter(|k| !taken.contains(*k)).collect();
        ensure_supported!(unused.is_empty(), "{} checkpoint tensors unused by the model: {unused:?}", unused.len());
        let params = self.tensors.values().map(|(_, d)| d.len()).sum();
        Ok((self.tensors.len(), self.aliases.len(), params))
    }
}
