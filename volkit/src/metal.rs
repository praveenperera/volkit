//! The Metal backend on Apple silicon: the device layer, the MSL kernels compiled at load, and the U-Net forward
//! AFV v2's engine runs on macOS ([`crate::nnunet::afv2`])

pub(crate) mod gpu;
pub(crate) mod model;
pub(crate) mod winograd;

#[cfg(test)]
mod tests;
