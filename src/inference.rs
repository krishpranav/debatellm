//! Text generation engine — the performance-critical core of DebateLM.
//!
//! This module owns the complete generation pipeline:
//!   load weights → tokenize prompt → prefill KV cache → decode tokens →
//!   sample → decode tokens → yield string
//!
//! Performance-critical design decisions (all justified in comments):
//! 1. mmap weight loading (no heap copy of 14GB weights)
//! 2. KV cache reuse within a request (O(1) decode steps)
//! 3. Top-p sampling via sorted cumulative sum (O(V log V) per token,
//!    acceptable at inference latency scale)
//! 4. Single batch dimension (batch_size=1) — streaming UX requires
//!    immediate token output, incompatible with batching
//! 5. Tensor created on device (no host→device copy per decode step)

use std::{
    io::Write,
    path::PathBuf,
};

use anyhow::Context;
use candle_core::{DType, Device, IndexOp, Tensor};
use hf_hub::api::tokio::Api;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{
    distributions::{Distribution, WeightedIndex},
    rngs::SmallRng,
    SeedableRng,
};

use crate::{
    config::{ModelConfig, ShardIndex},
    model::{auto_dtype, DebateLMModel},
    prompt,
    tokenizer::DebateLMTokenizer,
};

// ---------------------------------------------------------------------------
// Generation parameters
// ---------------------------------------------------------------------------

/// Parameters controlling the generation strategy.
///
/// Validated once in `Engine::generate`, not on every token.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate (not including the prompt).
    pub max_new_tokens: usize,
    /// Sampling temperature.
    /// - 0.0: greedy decoding (deterministic, highest-probability token)
    /// - 0.7: balanced (recommended for argumentation tasks)
    /// - 1.0: unconstrained sampling (more creative, less coherent)
    /// - >1.0: increasingly chaotic (not recommended)
    pub temperature: f64,
    /// Nucleus (top-p) sampling threshold.
    /// - 1.0: disabled (sample from full distribution)
    /// - 0.9: sample from top-90% probability mass (recommended)
    /// - 0.5: very conservative (safe but repetitive)
    /// Must be in (0.0, 1.0].
    pub top_p: f64,
    /// Random seed for reproducible sampling. `None` = use OS entropy.
    pub seed: Option<u64>,
    /// Whether to stream tokens to stdout as they are generated.
    pub stream: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature:    0.7,
            top_p:          0.9,
            seed:           None,
            stream:         true,
        }
    }
}

impl GenerationConfig {
    /// Validate parameter ranges.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.temperature < 0.0 || self.temperature > 2.0 {
            return Err(crate::error::DebateLMError::InvalidTemperature {
                value: self.temperature,
            });
        }
        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(crate::error::DebateLMError::InvalidTopP {
                value: self.top_p,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Engine — owns the loaded model and tokenizer
// ---------------------------------------------------------------------------

/// The inference engine. Load once, call `steelman` many times.
///
/// `Engine` is `Send` but not `Sync` — the KV cache is mutably updated on
/// every forward pass. Use separate `Engine` instances for concurrent requests.
pub struct Engine {
    model:     DebateLMModel,
    tokenizer: DebateLMTokenizer,
    config:    ModelConfig,
}

impl Engine {
    /// Load the engine from a HuggingFace Hub model repository.
    ///
    /// Downloads tokenizer.json, config.json, and weight shards if not already
    /// in the local HuggingFace cache (`~/.cache/huggingface/hub`).
    ///
    /// On second run, all files are served from the local cache — no network.
    pub async fn load(model_id: &str, dtype_override: Option<DType>) -> anyhow::Result<Self> {
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .unwrap(), // literal string — always valid
        );
        spinner.set_message(format!("Connecting to HuggingFace Hub ({model_id})…"));
        spinner.enable_steady_tick(std::time::Duration::from_millis(80));

        // Build the Hub API client (uses HUGGINGFACE_TOKEN env var if set).
        let api = Api::new().context("failed to initialise HuggingFace Hub API")?;
        let repo = api.model(model_id.to_string());

        // ----------------------------------------------------------------
        // Download config.json
        // ----------------------------------------------------------------
        spinner.set_message("fetching config.json…");
        let config_path = repo
            .get("config.json")
            .await
            .with_context(|| format!("cannot fetch config.json from {model_id}"))?;

        let config = ModelConfig::from_file(&config_path)
            .with_context(|| "failed to parse model config")?;

        // ----------------------------------------------------------------
        // Resolve device and dtype
        // ----------------------------------------------------------------
        let device = Device::cuda_if_available(0)
            .context("failed to initialise device")?;

        let dtype = dtype_override.unwrap_or_else(|| auto_dtype(&device));

        tracing::info!(
            model_id = model_id,
            device   = ?device,
            dtype    = ?dtype,
            "engine configuration resolved"
        );

        // ----------------------------------------------------------------
        // Download tokenizer
        // ----------------------------------------------------------------
        spinner.set_message("fetching tokenizer.json…");
        let tokenizer_path = repo
            .get("tokenizer.json")
            .await
            .with_context(|| format!("cannot fetch tokenizer.json from {model_id}"))?;

        let tokenizer = DebateLMTokenizer::from_file(&tokenizer_path)
            .with_context(|| "failed to load tokenizer")?;

        // ----------------------------------------------------------------
        // Download model weights — handle both single-file and sharded
        // ----------------------------------------------------------------
        spinner.set_message("resolving weight files…");
        let weight_paths = Self::resolve_weight_paths(&repo).await
            .with_context(|| "failed to resolve model weight files")?;

        spinner.set_message(format!(
            "loading {} weight shard(s) via mmap…",
            weight_paths.len()
        ));

        let model = DebateLMModel::load(&weight_paths, &config, &device, dtype)
            .with_context(|| "failed to load model weights")?;

        spinner.finish_with_message("✓ DebateLM engine ready");

        Ok(Self { model, tokenizer, config })
    }

    /// Load the engine from a local directory (no network required).
    ///
    /// Expects the directory to contain: `config.json`, `tokenizer.json`,
    /// and one or more `.safetensors` files.
    pub fn load_local(path: &std::path::Path, dtype_override: Option<DType>) -> anyhow::Result<Self> {
        let config = ModelConfig::from_file(path.join("config.json"))
            .context("failed to parse config.json")?;

        let device = Device::cuda_if_available(0)
            .context("failed to initialise device")?;

        let dtype = dtype_override.unwrap_or_else(|| auto_dtype(&device));

        let tokenizer = DebateLMTokenizer::from_file(path.join("tokenizer.json"))
            .context("failed to load tokenizer")?;

        let weight_paths = Self::resolve_local_weight_paths(path)
            .context("failed to find weight files in local directory")?;

        let model = DebateLMModel::load(&weight_paths, &config, &device, dtype)
            .context("failed to load model weights")?;

        tracing::info!(path = ?path, "engine loaded from local path");
        Ok(Self { model, tokenizer, config })
    }

    // -----------------------------------------------------------------------
    // Weight path resolution helpers
    // -----------------------------------------------------------------------

    /// Resolve weight file paths from a HuggingFace Hub repo.
    ///
    /// Strategy:
    /// 1. Try `model.safetensors.index.json` → sharded model.
    /// 2. Fall back to `model.safetensors` → single-file model.
    async fn resolve_weight_paths(
        repo: &hf_hub::api::tokio::ApiRepo,
    ) -> anyhow::Result<Vec<PathBuf>> {
        // Try sharded index first.
        let index_result = repo.get("model.safetensors.index.json").await;

        if let Ok(index_path) = index_result {
            let shard_index = ShardIndex::from_file(&index_path)
                .context("failed to parse shard index")?;
            let shard_names = shard_index.unique_shards();

            tracing::info!(num_shards = shard_names.len(), "downloading sharded model");

            let mut paths = Vec::with_capacity(shard_names.len());
            for shard_name in &shard_names {
                let shard_path = repo
                    .get(shard_name)
                    .await
                    .with_context(|| format!("failed to download shard '{shard_name}'"))?;
                paths.push(shard_path);
            }
            return Ok(paths);
        }

        // Fall back to single-file.
        let single_path = repo
            .get("model.safetensors")
            .await
            .context("model has neither 'model.safetensors.index.json' nor 'model.safetensors'")?;

        Ok(vec![single_path])
    }

    /// Resolve weight file paths from a local directory.
    fn resolve_local_weight_paths(dir: &std::path::Path) -> anyhow::Result<Vec<PathBuf>> {
        // Check for shard index
        let index_path = dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let shard_index = ShardIndex::from_file(&index_path)
                .context("failed to parse local shard index")?;
            let paths: Vec<PathBuf> = shard_index
                .unique_shards()
                .into_iter()
                .map(|s| dir.join(s))
                .collect();
            return Ok(paths);
        }

        // Single file
        let single = dir.join("model.safetensors");
        if single.exists() {
            return Ok(vec![single]);
        }

        // Glob for any .safetensors files
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .context("cannot read local model directory")?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
            .collect();

        if paths.is_empty() {
            anyhow::bail!("no .safetensors files found in {}", dir.display());
        }
        paths.sort();
        Ok(paths)
    }

    // -----------------------------------------------------------------------
    // Generation API
    // -----------------------------------------------------------------------

    /// Generate a steel-man counter-argument for the given argument.
    ///
    /// This is the primary public API. It handles:
    /// - Prompt formatting (model-specific template)
    /// - Tokenization
    /// - Context length validation
    /// - KV cache management (prefill + decode)
    /// - Sampling (greedy / temperature / top-p)
    /// - Streaming output (if `gen_config.stream = true`)
    /// - KV cache cleanup after generation
    pub fn steelman(
        &mut self,
        argument:   &str,
        gen_config: &GenerationConfig,
    ) -> anyhow::Result<String> {
        // Validate generation parameters first — fail fast.
        gen_config.validate().context("invalid generation parameters")?;

        // Format the instruction-tuned prompt.
        let prompt_text = prompt::format_steelman(argument, &self.config.model_type);

        // Tokenize (with BOS for Mistral).
        let prompt_ids = self
            .tokenizer
            .encode_with_bos(&prompt_text)
            .context("failed to tokenize prompt")?;

        let prompt_len = prompt_ids.len();

        // Validate context length BEFORE starting generation.
        // max_seq_len is the model's hard limit; we need room for both the
        // prompt and the generated tokens.
        let total_budget = self.model.max_seq_len;
        if prompt_len + gen_config.max_new_tokens > total_budget {
            anyhow::bail!(
                "prompt ({} tokens) + max_new_tokens ({}) = {} exceeds model context ({} tokens). \
                 Reduce --max-tokens or shorten the input argument.",
                prompt_len,
                gen_config.max_new_tokens,
                prompt_len + gen_config.max_new_tokens,
                total_budget,
            );
        }

        tracing::debug!(
            prompt_len      = prompt_len,
            max_new_tokens  = gen_config.max_new_tokens,
            temperature     = gen_config.temperature,
            top_p           = gen_config.top_p,
            "starting generation"
        );

        // Always clear KV cache before a new request.
        self.model.clear_kv_cache();

        // Build RNG.
        let mut rng = match gen_config.seed {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None       => SmallRng::from_entropy(),
        };

        // Run the generation loop.
        let output = self
            .generate_tokens(&prompt_ids, gen_config, &mut rng)
            .context("generation failed")?;

        Ok(output)
    }

    /// Core autoregressive generation loop.
    ///
    /// Phase 1 — Prefill:
    ///   Pass all prompt tokens at once. This fills the KV cache and gives us
    ///   logits for the last token (the first token to sample).
    ///   Cost: O(prompt_len²) due to full attention over the prompt.
    ///
    /// Phase 2 — Decode:
    ///   Pass one token at a time, incrementing `seqlen_offset` each step.
    ///   The KV cache from prefill is reused — each decode step costs O(1)
    ///   in attention (one query against prompt_len + generated_so_far keys).
    ///
    /// This two-phase structure is what makes autoregressive LLM inference
    /// fast. Without the KV cache, each decode step would be O(seq_len²).
    fn generate_tokens(
        &mut self,
        prompt_ids: &[u32],
        gen_config: &GenerationConfig,
        rng:        &mut SmallRng,
    ) -> anyhow::Result<String> {
        let device      = self.model.device.clone();
        let eos_id      = self.tokenizer.eos_token_id();
        let prompt_len  = prompt_ids.len();

        // ----------------------------------------------------------------
        // Phase 1: Prefill
        // ----------------------------------------------------------------
        // Build input tensor: shape [1, prompt_len], dtype u32.
        // `.to_device` is a no-op for CPU; for CUDA it copies host→device.
        let input_tensor = Tensor::new(prompt_ids, &device)
            .context("failed to build prompt input tensor")?
            .unsqueeze(0) // [prompt_len] → [1, prompt_len]
            .context("failed to add batch dimension")?;

        // Forward pass with seqlen_offset=0 (fresh KV cache).
        // Returns: [1, prompt_len, vocab_size]
        let prefill_logits = self
            .model
            .forward(&input_tensor, 0)
            .context("prefill forward pass failed")?;

        // Extract logits for the last prompt token: [vocab_size]
        // This gives us the distribution over the first token to generate.
        let last_logits = prefill_logits
            .i((0, prompt_len - 1, ..))
            .context("failed to slice prefill logits")?;

        // Sample the first generated token.
        let first_token = sample_token(&last_logits, gen_config, rng, self.model.vocab_size)
            .context("failed to sample first token")?;

        if first_token == eos_id {
            return Ok(String::new());
        }

        // Stream first token immediately if enabled.
        let mut output_tokens: Vec<u32> = Vec::with_capacity(gen_config.max_new_tokens);
        output_tokens.push(first_token);

        if gen_config.stream {
            let text = self
                .tokenizer
                .decode_token(first_token)
                .context("failed to decode first token")?;
            print!("{text}");
            std::io::stdout().flush().ok(); // Non-fatal if flush fails
        }

        // ----------------------------------------------------------------
        // Phase 2: Decode loop
        // ----------------------------------------------------------------
        // `seqlen_offset` starts at prompt_len (the KV cache has prompt_len
        // entries from prefill). Each iteration:
        //   1. Build a [1, 1] tensor with the last generated token.
        //   2. Forward pass → [1, 1, vocab_size] logits.
        //   3. Sample next token.
        //   4. Check EOS; if found, stop.
        //   5. Stream the decoded token to stdout.
        let mut seqlen_offset = prompt_len;

        for step in 1..gen_config.max_new_tokens {
            let last_token_id = *output_tokens
                .last()
                .expect("output_tokens is non-empty after first_token push"); // invariant

            // Build single-token input: [1, 1]
            let decode_input = Tensor::new(&[last_token_id][..], &device)
                .context("failed to build decode input tensor")?
                .unsqueeze(0)
                .context("failed to add batch dimension to decode input")?;

            // Forward pass with offset = prompt_len + tokens_generated_so_far
            let decode_logits = self
                .model
                .forward(&decode_input, seqlen_offset)
                .with_context(|| format!("decode forward pass failed at step {step}"))?;

            // Extract logits: [1, 1, vocab_size] → [vocab_size]
            let step_logits = decode_logits
                .i((0, 0, ..))
                .context("failed to slice decode logits")?;

            // Sample next token
            let next_token = sample_token(&step_logits, gen_config, rng, self.model.vocab_size)
                .with_context(|| format!("sampling failed at decode step {step}"))?;

            // Check EOS
            if next_token == eos_id {
                tracing::debug!(step, "EOS token generated — stopping");
                break;
            }

            output_tokens.push(next_token);
            seqlen_offset += 1;

            // Stream output
            if gen_config.stream {
                let text = self
                    .tokenizer
                    .decode_token(next_token)
                    .with_context(|| format!("failed to decode token at step {step}"))?;
                print!("{text}");
                std::io::stdout().flush().ok();
            }
        }

        if gen_config.stream {
            println!(); // Final newline after streamed output
        }

        // Decode the complete sequence for the return value.
        // Even in streaming mode we return the full string for programmatic use.
        let full_output = self
            .tokenizer
            .decode_sequence(&output_tokens)
            .context("failed to decode output sequence")?;

        tracing::debug!(
            tokens_generated = output_tokens.len(),
            "generation complete"
        );

        Ok(full_output)
    }
}

// ---------------------------------------------------------------------------
// Sampling implementations
// ---------------------------------------------------------------------------

/// Sample the next token ID from a logit vector.
///
/// Implements three strategies, selected by `gen_config.temperature`:
/// - `temperature == 0.0`: greedy — always pick the highest-logit token.
///   Deterministic, fast (no softmax needed, just argmax).
/// - `0.0 < temperature <= 2.0, top_p == 1.0`: temperature sampling.
///   Apply temperature scaling, softmax, then sample from full distribution.
/// - `0.0 < temperature <= 2.0, top_p < 1.0`: nucleus (top-p) sampling.
///   Apply temperature scaling, softmax, then restrict to top-p probability
///   mass before sampling. This is the production-quality path.
///
/// # Arguments
/// - `logits`: raw unnormalized logit vector, shape `[vocab_size]`, dtype F32.
/// - `gen_config`: generation parameters (temperature, top_p).
/// - `rng`: mutable reference to a seeded RNG (per-request, for reproducibility).
/// - `vocab_size`: upper bound on valid token IDs (for bounds checking).
///
/// # Complexity
/// - Greedy: O(V) — single argmax scan.
/// - Temperature only: O(V) — softmax + one call to `WeightedIndex::new`.
/// - Top-p: O(V log V) — requires sorting. Acceptable at inference scale
///   (vocab_size=32000, generate=512 tokens; total sort cost is well under 1ms
///    per token on modern hardware).
fn sample_token(
    logits:     &Tensor,
    gen_config: &GenerationConfig,
    rng:        &mut SmallRng,
    vocab_size:  usize,
) -> anyhow::Result<u32> {
    let temperature = gen_config.temperature;

    // ----------------------------------------------------------------
    // Greedy decoding (temperature == 0.0 or very close to 0)
    // ----------------------------------------------------------------
    if temperature < 1e-9 {
        let logits_vec = logits
            .to_vec1::<f32>()
            .context("failed to extract logits for greedy sampling")?;
        let best = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| anyhow::anyhow!("empty logit vector"))?;
        return Ok(best);
    }

    // ----------------------------------------------------------------
    // Temperature scaling → softmax → probability vector
    // ----------------------------------------------------------------
    // Division by temperature: low temp → sharpens distribution (peaky),
    // high temp → flattens distribution (more uniform).
    // We operate on the raw logits before softmax — this is the canonical
    // and numerically stable approach. Never apply temperature after softmax.
    let scaled_logits = (logits / temperature)
        .context("temperature scaling failed")?;

    // log-sum-exp softmax for numerical stability.
    // candle_nn::ops::softmax handles the subtract-max trick internally.
    let probs_tensor = candle_nn::ops::softmax(&scaled_logits, 0)
        .context("softmax failed")?;

    let mut probs: Vec<f32> = probs_tensor
        .to_vec1::<f32>()
        .context("failed to extract probability vector")?;

    // Sanity check: all probs should be non-negative after softmax.
    // NaN or negative values indicate upstream issues (e.g., BF16 overflow).
    if probs.iter().any(|p| p.is_nan() || *p < 0.0) {
        anyhow::bail!(
            "NaN or negative probability detected after softmax — \
             consider using --dtype f32 or checking model weights"
        );
    }

    // ----------------------------------------------------------------
    // Top-p (nucleus) filtering
    // ----------------------------------------------------------------
    // Only runs when top_p < 1.0 (i.e., in production mode).
    if gen_config.top_p < 1.0 {
        nucleus_filter(&mut probs, gen_config.top_p);
    }

    // ----------------------------------------------------------------
    // Weighted sampling
    // ----------------------------------------------------------------
    let dist = WeightedIndex::new(&probs)
        .context("failed to build weighted distribution (all probabilities may be zero)")?;

    let sampled_idx = dist.sample(rng) as u32;

    // Bounds check — WeightedIndex can't return out-of-bounds, but
    // validate anyway because model weights could produce unexpected vocab sizes.
    if sampled_idx as usize >= vocab_size {
        anyhow::bail!(
            "sampled token id {} is out of vocab bounds ({})",
            sampled_idx,
            vocab_size
        );
    }

    Ok(sampled_idx)
}

/// Apply nucleus (top-p) filtering to a probability vector in-place.
///
/// Algorithm:
/// 1. Find the top-p token set: sort by descending probability, accumulate
///    until cumulative sum >= top_p. Everything outside this set is zeroed.
/// 2. Renormalize: after zeroing, the remaining probs no longer sum to 1.
///    Divide by their new sum so that `WeightedIndex` gets a valid distribution.
///
/// # Complexity
/// O(V log V) for the sort. For Mistral's vocab_size=32000, this is
/// ~32000 * log2(32000) ≈ 480000 operations — negligible vs. the forward pass.
///
/// # Numerical notes
/// - We use f64 for the accumulation to avoid float32 rounding at small top_p values.
/// - Renormalization handles the case where floating-point accumulation causes
///   the cumsum to overshoot top_p by a tiny epsilon, zeroing one extra token.
fn nucleus_filter(probs: &mut Vec<f32>, top_p: f64) {
    // Build sorted (probability, original_index) pairs.
    let mut indexed: Vec<(f32, usize)> = probs
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, p)| (p, i))
        .collect();

    // Sort descending by probability.
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Find nucleus boundary.
    let mut cumulative: f64 = 0.0;
    let mut keep_count = indexed.len(); // default: keep all (fallback if top_p=1.0)
    for (i, (prob, _)) in indexed.iter().enumerate() {
        cumulative += *prob as f64;
        if cumulative >= top_p {
            keep_count = i + 1;
            break;
        }
    }

    // Build the set of indices to keep (the nucleus).
    let nucleus: std::collections::HashSet<usize> = indexed
        .iter()
        .take(keep_count)
        .map(|(_, idx)| *idx)
        .collect();

    // Zero out tokens outside the nucleus.
    for (i, p) in probs.iter_mut().enumerate() {
        if !nucleus.contains(&i) {
            *p = 0.0;
        }
    }

    // Renormalize.
    let total: f32 = probs.iter().sum();
    if total > 1e-8 {
        let scale = 1.0 / total;
        for p in probs.iter_mut() {
            *p *= scale;
        }
    }
    // If total <= 1e-8, all probs are effectively zero — WeightedIndex::new
    // will return an error, caught in sample_token. This should never happen
    // for valid top_p > 0 with a non-degenerate distribution.
}