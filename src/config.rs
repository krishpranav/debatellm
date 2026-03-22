//! Model configuration — deserialization from HuggingFace config.json.
//!
//! Architecture: We define our own `RawConfig` that matches the superset of
//! fields from Mistral and Phi-3 config.json schemas. Both schemas share most
//! field names but differ on a few (e.g., Phi-3 uses `original_max_position_embeddings`).
//! We use `#[serde(default)]` and `Option<T>` liberally, then validate and
//! convert in `ModelConfig::from_raw`.
//!
//! Performance note: This struct is deserialized exactly once at startup.
//! No heap pressure on the hot inference path.

use serde::Deserialize;

use crate::error::{DebateLMError, Result};

// ---------------------------------------------------------------------------
// Public-facing model type discriminant
// ---------------------------------------------------------------------------

/// Which transformer architecture is being loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelType {
    Mistral,
    Phi3,
}

impl ModelType {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "mistral"                => Ok(Self::Mistral),
            "phi3" | "phi-3" | "phi_3" => Ok(Self::Phi3),
            other => Err(DebateLMError::UnsupportedModelType {
                model_type: other.to_string(),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Raw deserialization struct — matches HuggingFace config.json superset
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Clone)]
pub struct RawConfig {
    pub model_type: String,

    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,

    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    pub max_position_embeddings: usize,

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    // Mistral-specific
    #[serde(default)]
    pub rope_theta: Option<f64>,

    #[serde(default)]
    pub sliding_window: Option<usize>,

    // Optional per-head dimension override (Mistral v0.3+ adds this)
    #[serde(default)]
    pub head_dim: Option<usize>,

    // Phi-3 specific
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,

    #[serde(default)]
    pub rope_scaling: Option<candle_transformers::models::phi3::RopeScaling>,

    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,

    #[serde(default)]
    pub bos_token_id: Option<u32>,

    #[serde(default)]
    pub eos_token_id: Option<u32>,

    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
}

fn default_vocab_size() -> usize { 32_000 }
fn default_hidden_act()  -> String { "silu".to_string() }
fn default_rms_norm_eps() -> f64   { 1e-5 }

// ---------------------------------------------------------------------------
// Processed configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type:              ModelType,
    pub vocab_size:              usize,
    pub hidden_size:             usize,
    pub intermediate_size:       usize,
    pub num_hidden_layers:       usize,
    pub num_attention_heads:     usize,
    pub num_key_value_heads:     usize,
    pub head_dim:                Option<usize>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps:            f64,
    /// f64 throughout — Candle's Mistral Config uses f64 for rope_theta.
    pub rope_theta:              f64,
    pub sliding_window:          Option<usize>,
    pub use_flash_attn:          bool,
    // Phi-3 extras preserved for the phi3 config builder
    pub original_max_position_embeddings: Option<usize>,
    pub rope_scaling:            Option<candle_transformers::models::phi3::RopeScaling>,
    pub partial_rotary_factor:   Option<f64>,
    pub bos_token_id:            Option<u32>,
    pub eos_token_id:            Option<u32>,
    pub tie_word_embeddings:     bool,
}

impl ModelConfig {
    pub fn from_raw(raw: RawConfig) -> Result<Self> {
        let model_type = ModelType::from_str(&raw.model_type)?;

        let num_key_value_heads = raw
            .num_key_value_heads
            .unwrap_or(raw.num_attention_heads);

        // Rope theta: both Mistral and Phi-3 default to 10_000.0
        let rope_theta = raw.rope_theta.unwrap_or(10_000.0_f64);

        // Sliding window: Phi-3 doesn't expose this field
        let sliding_window = match &model_type {
            ModelType::Mistral => raw.sliding_window,
            ModelType::Phi3   => None,
        };

        let use_flash_attn = cfg!(feature = "flash-attn");

        Ok(Self {
            model_type,
            vocab_size:              raw.vocab_size,
            hidden_size:             raw.hidden_size,
            intermediate_size:       raw.intermediate_size,
            num_hidden_layers:       raw.num_hidden_layers,
            num_attention_heads:     raw.num_attention_heads,
            num_key_value_heads,
            head_dim:                raw.head_dim,
            max_position_embeddings: raw.max_position_embeddings,
            rms_norm_eps:            raw.rms_norm_eps,
            rope_theta,
            sliding_window,
            use_flash_attn,
            original_max_position_embeddings: raw.original_max_position_embeddings,
            rope_scaling:            raw.rope_scaling,
            partial_rotary_factor:   raw.partial_rotary_factor,
            bos_token_id:            raw.bos_token_id,
            eos_token_id:            raw.eos_token_id,
            tie_word_embeddings:     raw.tie_word_embeddings.unwrap_or(false),
        })
    }

    /// Build the Candle Mistral config.
    ///
    /// Fixes applied vs. original:
    /// - `rope_theta`    → f64 (Candle uses f64, not f32)
    /// - `sliding_window`→ Option<usize> passed directly (not unwrapped to usize)
    /// - `head_dim`      → Option<usize> forwarded; None means "derive from hidden_size / heads"
    pub fn to_mistral_candle_config(
        &self,
    ) -> candle_transformers::models::mistral::Config {
        candle_transformers::models::mistral::Config {
            vocab_size:              self.vocab_size,
            hidden_size:             self.hidden_size,
            intermediate_size:       self.intermediate_size,
            num_hidden_layers:       self.num_hidden_layers,
            num_attention_heads:     self.num_attention_heads,
            num_key_value_heads:     self.num_key_value_heads,
            hidden_act:              candle_nn::Activation::Silu,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps:            self.rms_norm_eps,
            rope_theta:              self.rope_theta,       // f64 ✓
            sliding_window:          self.sliding_window,  // Option<usize> ✓
            use_flash_attn:          self.use_flash_attn,
            head_dim:                self.head_dim,         // Option<usize> ✓
        }
    }

    /// Build the Candle Phi-3 config.
    ///
    /// Phi-3's Candle Config does NOT have `sliding_window` or `use_flash_attn`.
    /// It has its own set of fields: bos/eos token ids, rope_scaling,
    /// original_max_position_embeddings, partial_rotary_factor, tie_word_embeddings.
    pub fn to_phi3_candle_config(
        &self,
    ) -> candle_transformers::models::phi3::Config {
        candle_transformers::models::phi3::Config {
            vocab_size:                       self.vocab_size,
            hidden_size:                      self.hidden_size,
            intermediate_size:                self.intermediate_size,
            num_hidden_layers:                self.num_hidden_layers,
            num_attention_heads:              self.num_attention_heads,
            num_key_value_heads:              self.num_key_value_heads,
            hidden_act:                       candle_nn::Activation::Silu,
            max_position_embeddings:          self.max_position_embeddings,
            rms_norm_eps:                     self.rms_norm_eps,
            rope_theta:                       self.rope_theta,
            bos_token_id:                     self.bos_token_id,
            eos_token_id:                     self.eos_token_id,
            rope_scaling:                     self.rope_scaling.clone(), // Option<RopeScaling> ✓
            original_max_position_embeddings: self.original_max_position_embeddings,
            partial_rotary_factor:            self.partial_rotary_factor, // Option<f64> ✓ — pass directly, no unwrap
            tie_word_embeddings:              self.tie_word_embeddings,
        }
    }

    /// Load from a config.json file on disk.
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().display().to_string();
        let file = std::fs::File::open(&path).map_err(|e| DebateLMError::Io {
            path: path_str.clone(),
            source: e,
        })?;
        let raw: RawConfig =
            serde_json::from_reader(file).map_err(|e| DebateLMError::ConfigDeserialize {
                source: e,
            })?;
        Self::from_raw(raw)
    }
}

// ---------------------------------------------------------------------------
// Shard index — for multi-file safetensors models
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ShardIndex {
    pub weight_map: std::collections::HashMap<String, String>,
}

impl ShardIndex {
    pub fn unique_shards(&self) -> Vec<String> {
        let mut shards: Vec<String> = self
            .weight_map
            .values()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        shards.sort();
        shards
    }

    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().display().to_string();
        let file = std::fs::File::open(&path).map_err(|e| DebateLMError::Io {
            path: path_str.clone(),
            source: e,
        })?;
        serde_json::from_reader(file).map_err(|e| DebateLMError::ShardIndexMalformed {
            path: path_str,
            source: e,
        })
    }
}