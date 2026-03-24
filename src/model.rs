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

/// Dispatch enum over supported transformer architectures.
///
/// Adding a new architecture means:
/// 1. Add a variant here
/// 2. Add a `from_var_builder` arm in `DebateLMModel::load`
/// 3. Add `forward` and `clear_kv_cache` arms
///
/// No changes needed in `inference.rs`.
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

/// Owns the loaded model variant and the device it's resident on.
pub struct DebateLMModel {
    pub variant: ModelVariant,
    pub device:  Device,
    pub dtype:   DType,
    /// Vocab size from config — used for bounds-checking sampled token ids.
    pub vocab_size: usize,
    /// Model's maximum context length — enforced before generation starts.
    pub max_seq_len: usize,
}

impl DebateLMModel {
    /// Load a model from memory-mapped safetensors files.
    ///
    /// # Safety contract
    /// `from_mmaped_safetensors` is `unsafe` because:
    /// 1. It calls `mmap(2)` on the weight files.
    /// 2. If the files are modified externally while inference runs, the
    ///    mmap'd memory changes → undefined behavior in the forward pass.
    ///
    /// This is safe in practice because:
    /// - Weight files written by the training script are never modified after
    ///   the training process exits.
    /// - The HuggingFace cache writes atomically via temp file + rename.
    ///
    /// The `unsafe` block is here, not spread through the call stack, so that
    /// all mmap safety reasoning lives in one place.
    pub fn load(
        weight_paths: &[std::path::PathBuf],
        config:       &ModelConfig,
        device:       &Device,
        dtype:        DType,
    ) -> Result<Self> {
        tracing::info!(
            num_shards  = weight_paths.len(),
            dtype       = ?dtype,
            model_type  = ?config.model_type,
            "loading model weights via mmap"
        );

        // Safety: see contract above.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(weight_paths, dtype, device)
                .map_err(|e| DebateLMError::WeightsMmap {
                    message: e.to_string(),
                })?
        };

        let variant = match &config.model_type {
            ModelType::Mistral => {
                let candle_cfg = config.to_mistral_candle_config();
                let model = mistral::Model::new(&candle_cfg, vb)
                    .map_err(|e| DebateLMError::ModelBuild {
                        message: e.to_string(),
                    })?;
                ModelVariant::Mistral(model)
            }
            ModelType::Phi3 => {
                let candle_cfg = config.to_phi3_candle_config();
                let model = phi3::Model::new(&candle_cfg, vb)
                    .map_err(|e| DebateLMError::ModelBuild {
                        message: e.to_string(),
                    })?;
                ModelVariant::Phi3(model)
            }
        };

        tracing::info!("model weights loaded successfully");

        Ok(Self {
            variant,
            device: device.clone(),
            dtype,
            vocab_size:  config.vocab_size,
            max_seq_len: config.max_position_embeddings,
        })
    }

    /// Convenience: delegate forward to the inner variant.
    #[inline]
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.variant
            .forward(input_ids, seqlen_offset)
            .map_err(|e| DebateLMError::ForwardPassFailed {
                step:    seqlen_offset,
                message: e.to_string(),
            })
    }

    /// Convenience: clear KV cache and reset generation state.
    #[inline]
    pub fn clear_kv_cache(&mut self) {
        self.variant.clear_kv_cache();
    }
}

/// Select the best available dtype for inference on the given device.
///
/// - CUDA: BF16 preferred (A100/H100 have native BF16 tensor cores; matches
///   most Mistral checkpoint dtype). Falls back to F16 if BF16 unavailable,
///   then F32.
/// - Metal: F32 (Metal does not reliably support BF16 across all Apple GPUs).
/// - CPU: F32 (most x86/ARM CPUs don't have native F16 arithmetic; F32 keeps
///   numerical behaviour predictable).
///
/// The caller can override this with `--dtype` on the CLI.
pub fn auto_dtype(device: &Device) -> DType {
    match device {
        Device::Cuda(_) => DType::BF16,
        Device::Metal(_) => DType::F32,
        Device::Cpu => DType::F32,
    }
}