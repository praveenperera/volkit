//! Placeholder until the torch-free reader lands
use std::collections::BTreeMap;
use std::path::Path;

use color_eyre::eyre::{Result, bail};

use crate::weights::Tensors;

/// An nnU-Net model folder converted to the safetensors export `export_nnunet.py` writes
pub struct Export {
    pub sidecar: serde_json::Value,
    tensors: Tensors,
    aliases: BTreeMap<String, String>,
}

impl Export {
    /// Reads `dir` without torch and maps it as `export_nnunet.py --recipe afv-v2` does
    pub fn afv_v2(dir: &Path) -> Result<Self> {
        bail!("{}: not implemented yet", dir.display())
    }

    /// The tensors, aliases and sidecar
    pub fn into_parts(self) -> (Tensors, BTreeMap<String, String>, serde_json::Value) {
        (self.tensors, self.aliases, self.sidecar)
    }
}
