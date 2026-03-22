//! Typed error hierarchy for DebateLM.
//!
//! Architecture decision: `thiserror` for the library layer (all modules except
//! `main.rs`), `anyhow` for the application layer (`main.rs` and `inference::Engine`
//! public API). This separation means library consumers get structured errors they
//! can pattern-match on, while the CLI layer gets ergonomic `?` propagation with
//! automatic context chaining via `anyhow::Context`.
//!
//! Rule: no module in this crate calls `.unwrap()` or `.expect()` in a code path
//! that can be reached by user input. The only permissible `.expect()` calls are
//! on compile-time invariants (e.g., building a regex from a literal).

use thiserror::Error;

/// All typed errors that can originate in the DebateLM library layer.
///
/// Each variant carries enough context to produce a diagnostic message without
/// requiring the caller to re-examine the offending value.
#[derive(Debug, Error)]
pub enum DebateLMError {
    // -----------------------------------------------------------------------
    // Configuration errors
    // -----------------------------------------------------------------------
    #[error("failed to deserialize model config: {source}")]
    ConfigDeserialize {
        #[source]
        source: serde_json::Error,
    },

    #[error("unsupported model type '{model_type}' — expected 'mistral' or 'phi3'")]
    UnsupportedModelType { model_type: String },

    #[error("missing required config field '{field}' for model type '{model_type}'")]
    MissingConfigField { field: &'static str, model_type: String },

    // -----------------------------------------------------------------------
    // Tokenizer errors
    // -----------------------------------------------------------------------
    #[error("failed to load tokenizer from file '{path}': {message}")]
    TokenizerLoad { path: String, message: String },

    #[error("tokenization failed for input of length {input_len}: {message}")]
    TokenizationFailed { input_len: usize, message: String },

    #[error("token decode failed for id {token_id}: {message}")]
    TokenDecodeFailed { token_id: u32, message: String },

    #[error("EOS token '</s>' not found in tokenizer vocabulary; cannot determine stop condition")]
    EosTokenNotFound,

    #[error("prompt is empty after tokenization — nothing to generate from")]
    EmptyPrompt,

    // -----------------------------------------------------------------------
    // Model loading errors
    // -----------------------------------------------------------------------
    #[error("safetensors weight file not found at '{path}'")]
    WeightsNotFound { path: String },

    #[error("failed to memory-map safetensors weights: {message}")]
    WeightsMmap { message: String },

    #[error("model construction failed: {message}")]
    ModelBuild { message: String },

    #[error("shard index file malformed at '{path}': {source}")]
    ShardIndexMalformed {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    // -----------------------------------------------------------------------
    // Inference / generation errors
    // -----------------------------------------------------------------------
    #[error("forward pass failed at step {step}: {message}")]
    ForwardPassFailed { step: usize, message: String },

    #[error("logit sampling failed: {message}")]
    SamplingFailed { message: String },

    #[error(
        "max_tokens ({max_tokens}) exceeds model's max_position_embeddings ({max_pos}) — \
         reduce --max-tokens or use a model with longer context"
    )]
    ContextLengthExceeded { max_tokens: usize, max_pos: usize },

    #[error("temperature must be in range [0.0, 2.0], got {value}")]
    InvalidTemperature { value: f64 },

    #[error("top_p must be in range (0.0, 1.0], got {value}")]
    InvalidTopP { value: f64 },

    // -----------------------------------------------------------------------
    // HuggingFace Hub errors
    // -----------------------------------------------------------------------
    #[error("failed to fetch '{filename}' from Hub repo '{repo_id}': {message}")]
    HubFetch {
        repo_id: String,
        filename: String,
        message: String,
    },

    // -----------------------------------------------------------------------
    // I/O errors
    // -----------------------------------------------------------------------
    #[error("I/O error reading '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

/// Convenience alias used throughout the library layer.
///
/// Using a named alias (rather than `Result<T, Box<dyn Error>>`) means
/// callers get named variants for match arms and IDEs can autocomplete.
pub type Result<T> = std::result::Result<T, DebateLMError>;