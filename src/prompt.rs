//! Prompt formatting for instruction-following models.
//!
//! Each model family has its own instruction template. Getting this exactly
//! right is critical — even one extra space or a missing `[/INST]` marker
//! causes the model to treat the instruction as data rather than a directive,
//! severely degrading output quality.
//!
//! References:
//! - Mistral instruction format: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
//! - Phi-3 instruction format:   https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

/// Mistral instruction template:
/// `<s>[INST] {system}\n\n{user} [/INST]`
///
/// Note:
/// - `<s>` is prepended by `tokenizer.encode_with_bos()` — NOT here.
///   We only format the text; BOS insertion is tokenizer-level.
/// - There is a space before `[/INST]` — this is load-bearing.
///   Missing it causes the model to hallucinate extra instruction text.
/// - No `</s>` at the end — the model generates until it predicts EOS.
const MISTRAL_SYSTEM_PROMPT: &str = "\
You are DebateLM, an expert in steel-manning arguments. \
Your task is to produce the strongest possible counter-argument to the position given. \
Do not strawman. Do not dismiss. Generate the most intellectually rigorous, \
evidence-grounded, and logically coherent opposing case possible. \
Lead with the most powerful objection, then support it with specific reasoning, \
empirical evidence where available, and philosophical/theoretical grounding. \
Maintain intellectual honesty throughout.";

/// Format a steel-manning prompt in Mistral's instruction format.
///
/// Output format:
/// ```text
/// [INST] You are DebateLM...
///
/// Steel-man the following argument:
///
/// {argument}
///  [/INST]
/// ```
///
/// The caller is responsible for tokenizing with `encode_with_bos` so that
/// `<s>` is prepended at the token level rather than the text level.
pub fn format_steelman_mistral(argument: &str) -> String {
    format!(
        "[INST] {system}\n\nSteel-man the following argument:\n\n{argument} [/INST]",
        system   = MISTRAL_SYSTEM_PROMPT,
        argument = argument.trim(),
    )
}

/// Phi-3 instruction template:
/// `<|user|>\n{message}<|end|>\n<|assistant|>`
///
/// Phi-3 uses a chat-template format with role tokens rather than
/// `[INST]`/`[/INST]`. The `<|end|>` token signals turn end.
/// The `<|assistant|>` token at the end primes the model to generate
/// the assistant's response.
const PHI3_SYSTEM_PROMPT: &str = "\
You are DebateLM, an expert in steel-manning arguments. \
Produce the strongest possible counter-argument to the position given. \
Do not strawman. Generate the most rigorous, evidence-grounded opposing case possible.";

/// Format a steel-manning prompt in Phi-3's instruction format.
pub fn format_steelman_phi3(argument: &str) -> String {
    format!(
        "<|system|>\n{system}<|end|>\n<|user|>\nSteel-man the following argument:\n\n{argument}<|end|>\n<|assistant|>",
        system   = PHI3_SYSTEM_PROMPT,
        argument = argument.trim(),
    )
}

/// Format a steel-manning prompt for the given model type.
///
/// This is the single public entry point used by `inference.rs`.
/// Centralising dispatch here means `inference.rs` stays model-agnostic.
pub fn format_steelman(argument: &str, model_type: &crate::config::ModelType) -> String {
    match model_type {
        crate::config::ModelType::Mistral => format_steelman_mistral(argument),
        crate::config::ModelType::Phi3    => format_steelman_phi3(argument),
    }
}

/// Validate that an argument string is suitable for processing.
///
/// Returns `Err` if the argument is too short (likely noise) or too long
/// (would overflow the context window before generation even starts).
///
/// # Arguments
/// - `argument`: the raw user-provided argument string
/// - `max_input_chars`: derived from `max_position_embeddings * ~4` chars/token
pub fn validate_argument(argument: &str, max_input_chars: usize) -> crate::error::Result<()> {
    let trimmed = argument.trim();
    if trimmed.is_empty() {
        return Err(crate::error::DebateLMError::EmptyPrompt);
    }
    // Rough char-based upper bound; tokenizer gives exact count but we
    // do this pre-tokenization check to fail fast on obviously huge inputs.
    if trimmed.len() > max_input_chars {
        return Err(crate::error::DebateLMError::ContextLengthExceeded {
            max_tokens: trimmed.len() / 4,
            max_pos:    max_input_chars / 4,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelType;

    #[test]
    fn mistral_prompt_contains_inst_markers() {
        let prompt = format_steelman("Climate policy hurts growth", &ModelType::Mistral);
        assert!(prompt.contains("[INST]"));
        assert!(prompt.contains("[/INST]"));
        assert!(prompt.contains("Climate policy hurts growth"));
        // Space before [/INST] is load-bearing
        assert!(prompt.contains(" [/INST]"));
    }

    #[test]
    fn phi3_prompt_contains_role_tokens() {
        let prompt = format_steelman("AI will destroy jobs", &ModelType::Phi3);
        assert!(prompt.contains("<|user|>"));
        assert!(prompt.contains("<|assistant|>"));
        assert!(prompt.contains("<|end|>"));
        assert!(prompt.contains("AI will destroy jobs"));
    }

    #[test]
    fn argument_is_trimmed() {
        let p1 = format_steelman("  trimmed  ", &ModelType::Mistral);
        let p2 = format_steelman("trimmed", &ModelType::Mistral);
        assert_eq!(p1, p2);
    }

    #[test]
    fn validate_rejects_empty() {
        assert!(validate_argument("", 10_000).is_err());
        assert!(validate_argument("   ", 10_000).is_err());
    }

    #[test]
    fn validate_accepts_normal_input() {
        assert!(validate_argument("Climate policy hurts growth.", 10_000).is_ok());
    }
}