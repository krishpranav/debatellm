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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelType {
    Mistral,
    Phi3,
}

impl ModelType {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "mistral" => Ok(Self::Mistral),
            "phi3" | "phi-3" | "phi_3" => Ok(Self::Phi3),

            other => Err(DebateLMError::UnsupportedModelType {
                model_type: other.to_string(),
            }),
        }
    }
}