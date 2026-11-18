//! Runs one AFV v2 column through [`volkit::nnunet::afv2::AfvV2`] from tiles cut elsewhere, so another engine
//! (inkrs's `inkrs.AfvV2`) can be checked against it byte for byte on the same input
//!
//!     afv2_smoke MODEL INPUT_DIR OUT_DIR fp32|bf16
//!
//! `MODEL` is the `.safetensors` export or the nnU-Net model folder. `INPUT_DIR` holds `tiles.u8`, the column's
//! `p^3` uint8 tiles back to back, and `meta.json`, `{"shape": [nz, ny, nx], "y0": 0, "x0": 0, "origins": [[pz, py,
//! px], ...]}`. Writes `semantic.u8`, `tracing.u8` and `report.json` to `OUT_DIR`

use std::path::PathBuf;

use color_eyre::eyre::{Result, WrapErr, bail, ensure};
use serde::Deserialize;
use volkit::dnn::Pick;
use volkit::nnunet::afv2::{AfvPrecision, AfvV2, MAX_BATCH, Options};

/// The column the tiles belong to
#[derive(Deserialize)]
struct Meta {
    shape: [usize; 3],
    y0: usize,
    x0: usize,
    origins: Vec<[usize; 3]>,
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args: Vec<String> = std::env::args().collect();
    let [_, model, input, out, precision] = args.as_slice() else {
        bail!("usage: afv2_smoke MODEL INPUT_DIR OUT_DIR fp32|bf16");
    };
    let (input, out) = (PathBuf::from(input), PathBuf::from(out));
    let meta: Meta = serde_json::from_slice(&std::fs::read(input.join("meta.json"))?).wrap_err("meta.json")?;
    let tiles = std::fs::read(input.join("tiles.u8")).wrap_err("tiles.u8")?;

    // `find`, as inkrs.AfvV2 defaults to
    let opts = Options { device: 0, pick: Pick::Find, precision: AfvPrecision::parse(precision)? };
    let mut net = AfvV2::open(&PathBuf::from(model), &opts)?;
    let vox = net.patch().pow(3);
    ensure!(tiles.len() == meta.origins.len() * vox, "{} tile bytes for {} origins", tiles.len(), meta.origins.len());

    let start = std::time::Instant::now();
    net.begin(meta.shape, meta.y0, meta.x0)?;
    // batches of MAX_BATCH, as the Python layer sends them
    for (batch, origins) in tiles.chunks(MAX_BATCH * vox).zip(meta.origins.chunks(MAX_BATCH)) {
        net.add(batch, origins)?;
    }
    let column = net.finish(true)?;
    let run_s = start.elapsed().as_secs_f64();

    std::fs::create_dir_all(&out)?;
    std::fs::write(out.join("semantic.u8"), column.semantic.unwrap_or_default())?;
    std::fs::write(out.join("tracing.u8"), column.tracing.unwrap_or_default())?;
    let report = serde_json::json!({
        "load": net.load,
        "tiles": column.tiles,
        "uncovered": column.uncovered,
        "run_s": run_s,
    });
    std::fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{} tiles in {run_s:.2} s, {} voxels uncovered", column.tiles, column.uncovered);
    Ok(())
}
