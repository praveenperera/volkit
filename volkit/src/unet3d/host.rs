//! An export's network tensors read from its checkpoint and laid out on the CPU, shared by the CUDA forward
//! (`unet3d::model`) and the Metal one (`metal::model`): conv weights channels-last `[K, taps, C]`, transposed
//! convs as GEMM weights `[Cin, taps * Cout]`, biases rounded to the precision

use color_eyre::eyre::Result;

use super::{ConvSpec, Precision, Program};
use crate::weights::Checkpoint;

/// One conv's weights read from the checkpoint and laid out on the CPU: the weight channels-last
/// `[K, taps, C]` float32 (rounded at upload, after any channel padding), the bias rounded to the
/// precision, the norm's float32 affine
pub struct HostConv {
    /// Input channels
    pub cin: usize,
    /// Output channels
    pub cout: usize,
    pub(crate) w: Vec<f32>,
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) norm: Option<(Vec<f32>, Vec<f32>, f32)>,
}

impl HostConv {
    pub(crate) fn take(ck: &Checkpoint, spec: &ConvSpec, precision: Precision) -> Result<Self> {
        let (k, c) = (spec.cout, spec.cin);
        let taps: usize = spec.kernel.iter().product();
        let shape: Vec<usize> = [k, c].into_iter().chain(spec.kernel.iter().copied()).collect();
        let w = ck.take(&spec.w, &shape)?;
        // pushed in `[K, taps, C]` order, so the buffer is written once with no zero-fill first
        let mut ktc = Vec::with_capacity(k * taps * c);
        for ko in 0..k {
            for t in 0..taps {
                ktc.extend((0..c).map(|ci| w[(ko * c + ci) * taps + t]));
            }
        }
        let bias = match &spec.b {
            Some(key) => Some(ck.take(key, &[k])?.iter().map(|&v| precision.round(v)).collect()),
            None => None,
        };
        let norm = match &spec.norm {
            Some(n) => Some((ck.take(&n.w, &[k])?, ck.take(&n.b, &[k])?, n.eps)),
            None => None,
        };
        Ok(Self { cin: c, cout: k, w: ktc, bias, norm })
    }

    /// The weight as `[K, taps, c_pad]` in the storage type, as bytes, the channels past `cin` zero
    #[cfg(any(test, not(target_os = "macos")))]
    pub(crate) fn stored(&self, precision: Precision, c_pad: usize) -> Vec<u8> {
        assert!(c_pad >= self.cin, "{c_pad} channels pad {} input channels", self.cin);
        let taps = self.w.len() / (self.cout * self.cin);
        let pad = (c_pad - self.cin) * precision.size();
        let mut out = Vec::with_capacity(self.cout * taps * c_pad * precision.size());
        for src in self.w.chunks_exact(self.cin) {
            precision.encode_into(src, &mut out);
            // +0.0 is all zero bits in every storage type
            out.resize(out.len() + pad, 0);
        }
        out
    }
}

/// One BasicBlockD's weights on the CPU
pub struct HostBlock {
    pub(crate) conv1: HostConv,
    pub(crate) conv2: HostConv,
    pub(crate) proj: Option<HostConv>,
}

/// One decoder stage's weights on the CPU: the transposed conv as a GEMM weight `[cin, taps * cout]`
/// (column `tap * cout + co`), its bias rounded, and the convs
pub struct HostDec {
    pub(crate) g: Vec<f32>,
    pub(crate) tb: Vec<f32>,
    pub(crate) tcin: usize,
    pub(crate) convs: Vec<HostConv>,
}

/// Every network tensor read from the checkpoint and laid out on the CPU, each taken once by name and
/// shape, so [`Checkpoint::finish`] after this proves the program reads the whole export
pub struct HostNet {
    pub(crate) precision: Precision,
    pub(crate) stem: Vec<HostConv>,
    pub(crate) blocks: Vec<HostBlock>,
    pub(crate) decs: Vec<HostDec>,
    /// The head's weight `[classes, cin]` and bias, rounded
    pub(crate) head_w: Vec<f32>,
    pub(crate) head_b: Vec<f32>,
    pub(crate) head_cin: usize,
}

impl HostNet {
    /// Takes the program's tensors from `ck`
    pub fn take(ck: &Checkpoint, prog: &Program, precision: Precision) -> Result<Self> {
        let stem = prog.stem.iter().map(|s| HostConv::take(ck, s, precision)).collect::<Result<_>>()?;
        let mut blocks = Vec::new();
        for b in prog.stages.iter().flatten() {
            blocks.push(HostBlock {
                conv1: HostConv::take(ck, &b.conv1, precision)?,
                conv2: HostConv::take(ck, &b.conv2, precision)?,
                proj: b.skip.proj.as_ref().map(|pj| HostConv::take(ck, pj, precision)).transpose()?,
            });
        }
        let mut decs = Vec::new();
        for ds in &prog.decoder {
            let t = &ds.transp;
            let (ci, co) = (t.cin, t.cout);
            let taps: usize = t.kernel.iter().product();
            let shape: Vec<usize> = [ci, co].into_iter().chain(t.kernel.iter().copied()).collect();
            let w = ck.take(&t.w, &shape)?;
            // pushed in `[Cin, taps, Cout]` order, written once with no zero-fill first
            let mut g = Vec::with_capacity(ci * taps * co);
            for i in 0..ci {
                for tap in 0..taps {
                    g.extend((0..co).map(|o| w[(i * co + o) * taps + tap]));
                }
            }
            let key = t.b.as_ref().expect("checked by the contract");
            let tb = ck.take(key, &[co])?.iter().map(|&v| precision.round(v)).collect();
            let convs = ds.convs.iter().map(|c| HostConv::take(ck, c, precision)).collect::<Result<_>>()?;
            decs.push(HostDec { g, tb, tcin: ci, convs });
        }
        let h = &prog.head;
        let head_w = ck.take(&h.w, &[h.cout, h.cin, 1, 1, 1])?.iter().map(|&v| precision.round(v)).collect();
        let key = h.b.as_ref().expect("checked by the contract");
        let head_b = ck.take(key, &[h.cout])?.iter().map(|&v| precision.round(v)).collect();
        Ok(Self { precision, stem, blocks, decs, head_w, head_b, head_cin: h.cin })
    }
}

#[cfg(test)]
mod host_tests {
    use std::collections::BTreeMap;

    use super::HostConv;
    use crate::unet3d::{ConvSpec, Precision};
    use crate::weights::Checkpoint;

    /// A conv's weight goes from PyTorch's `[K, C, taps]` to channels-last `[K, taps, c_pad]` with the padded
    /// channels zero, value for value in float32 and rounded in bf16
    #[test]
    fn conv_weights_store_channels_last_and_padded() {
        // K 2, C 3, a 1x1x2 kernel; weight (k, c, t) holds 10k + 2c + t + 1/3, which bf16 rounds
        let w: Vec<f32> = (0..12).map(|i| (10 * (i / 6) + i % 6) as f32 + 1.0 / 3.0).collect();
        let tensors = BTreeMap::from([("w".to_owned(), (vec![2, 3, 1, 1, 2], w))]);
        let ck = Checkpoint::from_parts(tensors, BTreeMap::new()).unwrap();
        let spec = ConvSpec {
            w: "w".into(),
            b: None,
            cin: 3,
            cout: 2,
            kernel: vec![1, 1, 2],
            stride: vec![1, 1, 1],
            norm: None,
            leaky: None,
        };
        let host = HostConv::take(&ck, &spec, Precision::Tf32).unwrap();
        assert_eq!(ck.finish().unwrap(), (1, 0, 12));

        // rows (k, t), each the three channels at that tap then the padding
        let rows = [[0, 2, 4], [1, 3, 5], [10, 12, 14], [11, 13, 15]];
        let value = |v: i32| v as f32 + 1.0 / 3.0;
        let fp32: Vec<u8> =
            rows.iter().flat_map(|r| r.iter().map(|&v| value(v).to_le_bytes()).chain([[0; 4]; 2])).flatten().collect();
        assert_eq!(host.stored(Precision::Tf32, 5), fp32);
        let bf16: Vec<u8> = rows
            .iter()
            .flat_map(|r| r.iter().map(|&v| half::bf16::from_f32(value(v)).to_le_bytes()).chain([[0; 2]]))
            .flatten()
            .collect();
        assert_eq!(host.stored(Precision::Bf16, 4), bf16);
    }
}
