//! bf16 and TF32 GEMMs with float32 accumulation on [`Blas`], for the 3D U-Net's k2 s2 transposed convs

use std::ffi::{c_int, c_void};

use color_eyre::eyre::{Result, ensure, eyre};
use cudarc::cublas::sys as blas;

use super::Blas;

impl Blas {
    /// Row-major bf16 `y[m, n] = x[m, k] @ w[k, n]`, float32 accumulation, bf16 output
    pub fn gemm_bf16(
        &self,
        (m, n, k): (usize, usize, usize),
        x: (u64, usize),
        w: (u64, usize),
        y: (u64, usize),
    ) -> Result<()> {
        self.gemm_ex(
            (m, n, k),
            (x, w, y),
            (2, blas::cudaDataType::CUDA_R_16BF),
            blas::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            "bf16",
        )
    }

    /// Row-major float32 `y[m, n] = x[m, k] @ w[k, n]` with TF32 tensor-core math, float32 accumulation and
    /// output: PyTorch's float32 default for the cuDNN transposed convs this GEMM replaces, which also run TF32
    pub fn gemm_tf32(
        &self,
        (m, n, k): (usize, usize, usize),
        x: (u64, usize),
        w: (u64, usize),
        y: (u64, usize),
    ) -> Result<()> {
        self.gemm_ex(
            (m, n, k),
            (x, w, y),
            (4, blas::cudaDataType::CUDA_R_32F),
            blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            "tf32",
        )
    }

    /// `cublasGemmEx` of row-major operands of one data type (`size` bytes a value) under `compute`
    fn gemm_ex(
        &self,
        (m, n, k): (usize, usize, usize),
        (x, w, y): ((u64, usize), (u64, usize), (u64, usize)),
        (size, dtype): (usize, blas::cudaDataType),
        compute: blas::cublasComputeType_t,
        what: &str,
    ) -> Result<()> {
        ensure!(
            x.1 >= m * k * size && w.1 >= k * n * size && y.1 >= m * n * size,
            "gemm {m}x{n}x{k}: buffers too small"
        );
        let to = |v: usize| c_int::try_from(v).map_err(|_| eyre!("gemm size {v} overflows int"));
        let (alpha, beta) = (1f32, 0f32);
        // column-major cuBLAS sees the row-major product transposed: y^T = w^T x^T
        // SAFETY: sizes checked; host scalars read before the call returns
        unsafe {
            blas::cublasGemmEx(
                self.0,
                blas::cublasOperation_t::CUBLAS_OP_N,
                blas::cublasOperation_t::CUBLAS_OP_N,
                to(n)?,
                to(m)?,
                to(k)?,
                (&alpha as *const f32).cast(),
                w.0 as *const c_void,
                dtype,
                to(n)?,
                x.0 as *const c_void,
                dtype,
                to(k)?,
                (&beta as *const f32).cast(),
                y.0 as *mut c_void,
                dtype,
                to(n)?,
                compute,
                blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .result()
            .map_err(|e| eyre!("cublasGemmEx {what} {m}x{n}x{k}: {e:?}"))
        }
    }
}
