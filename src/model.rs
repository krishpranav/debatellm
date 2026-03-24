//! Candle model variant abstraction.
//!
//! We support two architectures (Mistral and Phi-3) behind a single enum.
//! The enum dispatches `forward` and `clear_kv_cache` identically for both,
//! so `inference.rs` never needs to know which architecture it's running.
//!
//! Architecture decision: enum dispatch over trait objects.
//! - Trait objects (`Box<dyn Model>`) would require `dyn` dispatch (vtable
//!   indirection) on every `forward` call — i.e., every token generated.
//! - Enum dispatch resolves to a single `match` arm which the compiler
//!   typically inlines. On a 7B model the attention computation dominates,
//!   but there is no reason to pay vtable overhead.
//!
//! Memory layout:
//! - Model weights are NOT copied into this struct. They live in the mmap'd
//!   region provided by `VarBuilder::from_mmaped_safetensors`. The Candle
//!   model holds `Tensor` handles that reference into that mmap region.
//! - On CUDA, weights are copied to device memory once during load; subsequent
//!   forward passes read from device memory.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{mistral, phi3};
use crate::{
    config::{ModelConfig, ModelType},
    error::{DebateLMError, Result},
};

pub enum ModelVariant {
    Mistral(mistral::Model),
    Phi3(phi3::Model),
}

impl ModelVariant {
    /// Run a forward pass through the transformer.
    ///
    /// # Arguments
    /// - `input_ids`: `Tensor` of shape `[batch=1, seq_len]`, dtype `u32`.
    /// - `seqlen_offset`: number of tokens already in the KV cache.
    ///   - First call (prefill): pass `0`.
    ///   - Decode steps: pass `prompt_len + tokens_generated_so_far`.
    ///
    /// # Returns
    /// `Tensor` of shape `[1, seq_len, vocab_size]`, dtype F32 or F16
    /// depending on the loaded dtype.
    #[inline]
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Mistral(m) => m.forward(input_ids, seqlen_offset),
            Self::Phi3(m)    => m.forward(input_ids, seqlen_offset),
        }
    }

    /// Clear the internal KV cache.
    ///
    /// MUST be called between independent generation requests. If not cleared,
    /// the previous request's tokens are still in the KV cache and act as
    /// invisible context — producing incoherent outputs.
    #[inline]
    pub fn clear_kv_cache(&mut self) {
        match self {
            Self::Mistral(m) => m.clear_kv_cache(),
            Self::Phi3(m)    => m.clear_kv_cache(),
        }
    }
}

pub struct DebateLMModel {
    pub variant: ModelVariant,
    pub device: Device,
    pub dtype: DType,
    pub vocab_size: usize,
    pub max_seq_line: usize,
}

impl DebateLMModel {
    pub fn load(
        weight_paths: &[std::path::PathBuf],
        config: &ModelConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        tracing::info!(
            num_shards  = weight_paths.len(),
            dtype       = ?dtype,
            model_type  = ?config.model_type,
            "loading model weights via mmap"
        );

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(weight_paths, dtype, device)
                .map_err(|e| DebateLMError::WeightsMmap {
                    message: e.to_string(),
                })?
        };

        let variant = match &config.model_type {
            ModelType::Mistral => {

            }

            ModelType::Phi3 => {

            }
        };

        Ok(Self {
            variant,
            device: device.clone(),
            dtype,
            vocab_size: config.vocab_size,
            max_seq_line: config.max_position_embeddings,
        });
    }

    #[inline]
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.variant
            .forward(input_ids, seqlen_offset)
            .map_err(|e| DebateLMError::ForwardPassFailed {
                step: seqlen_offset,
                message: e.to_string(),
            })
    }

    #[inline]
    pub fn clear_kv_cache(&mut self) {
        self.variant.clear_kv_cache();
    }
}

pub fn auto_dtype(device: &Device) -> DType {
    match device {
        Device::Cuda(_) => DType::BF16,
        Device::Metal(_) => DType::F32,
        Device::Cpu => DType::F32,
    }
}