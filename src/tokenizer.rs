//! Tokenizer wrapper around the `tokenizers` crate.
//!
//! Design goal: completely hide the `tokenizers` crate behind this module.
//! All other modules work with `&str`, `String`, and `Vec<u32>` — never
//! with `tokenizers::Encoding` or `tokenizers::Tokenizer` directly.
//!
//! Performance notes:
//! - `Tokenizer::from_file` loads the tokenizer into heap memory once.
//!   There is no way to mmap a tokenizer.json with the `tokenizers` crate
//!   today, so we accept this one-time allocation.
//! - `encode` returns a borrowed `Encoding` which holds a reference into
//!   the tokenizer's internal buffers. We copy the IDs immediately (`to_vec`)
//!   to avoid lifetime entanglement with the tokenizer.
//! - `decode_token` allocates a `String` per token — unavoidable for streaming
//!   output, but this is off the critical batch-inference path.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::error::{DebateLMError, Result};

/// Wrapper that owns a HuggingFace tokenizer and exposes a clean interface.
pub struct DebateLMTokenizer {
    inner:        Tokenizer,
    /// Cached EOS token id — looked up once at construction time.
    eos_token_id: u32,
    /// Cached BOS token id — needed to prepend to prompts for some models.
    bos_token_id: Option<u32>,
}

impl DebateLMTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    ///
    /// Also reads `tokenizer_config.json` (sibling file) to extract EOS/BOS
    /// token strings, then maps them to IDs via the vocabulary.
    pub fn from_file<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self> {
        let path_str = tokenizer_path.as_ref().display().to_string();

        let inner =
            Tokenizer::from_file(tokenizer_path.as_ref()).map_err(|e| {
                DebateLMError::TokenizerLoad {
                    path: path_str.clone(),
                    message: e.to_string(),
                }
            })?;

        // Resolve EOS token id.
        // Mistral uses </s> (id=2), Phi-3 uses <|end|> (id varies).
        // We try candidates in priority order.
        let eos_candidates = ["</s>", "<|end|>", "<|endoftext|>", "<eos>"];
        let eos_token_id = eos_candidates
            .iter()
            .find_map(|tok| inner.token_to_id(tok))
            .ok_or(DebateLMError::EosTokenNotFound)?;

        // BOS is optional — some models don't require it prepended.
        let bos_candidates = ["<s>", "<|begin_of_text|>", "<bos>"];
        let bos_token_id = bos_candidates
            .iter()
            .find_map(|tok| inner.token_to_id(tok));

        tracing::debug!(
            eos_token_id,
            bos_token_id = ?bos_token_id,
            "tokenizer loaded"
        );

        Ok(Self {
            inner,
            eos_token_id,
            bos_token_id,
        })
    }

    /// Encode a text prompt into token IDs.
    ///
    /// `add_special_tokens = false` because we manage BOS/EOS insertion
    /// ourselves in `prompt.rs` to maintain precise control over the
    /// instruction format. Letting the tokenizer auto-insert special tokens
    /// creates subtle bugs when the prompt already contains them.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Err(DebateLMError::EmptyPrompt);
        }

        let encoding =
            self.inner
                .encode(text, false)
                .map_err(|e| DebateLMError::TokenizationFailed {
                    input_len: text.len(),
                    message: e.to_string(),
                })?;

        let ids = encoding.get_ids().to_vec();
        tracing::debug!(token_count = ids.len(), "encoded prompt");
        Ok(ids)
    }

    /// Encode with BOS prepended if the model uses one.
    ///
    /// Used for models that require `<s>` at the start of every sequence
    /// (Mistral instruction format prepends it before `[INST]`).
    pub fn encode_with_bos(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = self.encode(text)?;
        if let Some(bos_id) = self.bos_token_id {
            ids.insert(0, bos_id);
        }
        Ok(ids)
    }

    /// Decode a single token ID to its string representation.
    ///
    /// This is used for streaming output: each generated token is decoded
    /// and printed immediately. The `skip_special_tokens = false` flag is
    /// intentional — we want to detect EOS tokens in the output stream
    /// without a separate ID check.
    ///
    /// SentencePiece tokens use the `▁` prefix (U+2581 LOWER ONE EIGHTH BLOCK)
    /// to encode leading spaces. We replace it with ` ` for human-readable output.
    pub fn decode_token(&self, token_id: u32) -> Result<String> {
        self.inner
            .id_to_token(token_id)
            .map(|tok| tok.replace('▁', " "))
            .ok_or(DebateLMError::TokenDecodeFailed {
                token_id,
                message: "token id not in vocabulary".to_string(),
            })
    }

    /// Decode a full sequence of token IDs into a string.
    ///
    /// Prefer this over concatenating `decode_token` results — the tokenizer
    /// handles multi-token subword sequences correctly (e.g., emoji, Unicode
    /// edge cases) when decoding as a batch.
    pub fn decode_sequence(&self, token_ids: &[u32]) -> Result<String> {
        self.inner
            .decode(token_ids, true)
            .map_err(|e| DebateLMError::TokenDecodeFailed {
                token_id: token_ids.first().copied().unwrap_or(0),
                message: e.to_string(),
            })
    }

    /// Returns the EOS token ID for this tokenizer/model combination.
    #[inline]
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Returns the BOS token ID if this model uses one.
    #[inline]
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Returns the vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}