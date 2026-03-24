//! DebateLM CLI entry point.
//!
//! All CLI logic lives here. All inference logic lives in `inference.rs`.
//! `main.rs` is intentionally thin — it parses arguments, initialises
//! tracing, calls the engine, and handles top-level errors.
//!
//! Error handling contract:
//! - `main` returns `anyhow::Result<()>`.
//! - `anyhow` automatically formats the error chain on exit.
//! - No `.unwrap()` or `.expect()` except on compile-time invariants
//!   (the `.unwrap()` in the ProgressStyle literal is such an invariant —
//!   the format string is a constant and is valid at compile time).

mod config;
mod error;
mod inference;
mod model;
mod prompt;
mod tokenizer;

use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
};

use anyhow::{Context, Result};
use candle_core::DType;
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};

use crate::inference::{Engine, GenerationConfig};

// ---------------------------------------------------------------------------
// CLI definition — clap derive macros
// ---------------------------------------------------------------------------

/// Steel-man any argument using DebateLM.
///
/// Generates the strongest possible counter-argument to any position.
/// Not a strawman. Not a dismissal. The most powerful opposing case.
///
/// Examples:
///   debatelm --argument "Remote work reduces team productivity"
///   debatelm --interactive
///   debatelm --argument "UBI reduces work incentive" --temperature 0.8 --top-p 0.95
#[derive(Parser, Debug)]
#[command(name = "debatelm")]
#[command(version)]
#[command(author = "DebateLM Contributors")]
#[command(propagate_version = true)]
#[command(arg_required_else_help = true)]
struct Cli {
    /// The argument to steel-man (enclose in quotes if it contains spaces).
    ///
    /// Example: --argument "Universal healthcare reduces innovation"
    #[arg(short, long, value_name = "ARGUMENT")]
    argument: Option<String>,

    /// Run in interactive REPL mode.
    ///
    /// Accepts arguments one per line from stdin. Type 'exit' or press Ctrl+D to quit.
    #[arg(short, long, default_value_t = false)]
    interactive: bool,

    /// HuggingFace Hub model ID (e.g., "your-username/debatelm").
    ///
    /// Can also be set via the DEBATELM_MODEL_ID environment variable.
    /// Not needed if --local-path is set.
    #[arg(long, env = "DEBATELM_MODEL_ID", value_name = "MODEL_ID")]
    model_id: Option<String>,

    /// Load model from a local directory instead of HuggingFace Hub.
    ///
    /// The directory must contain config.json, tokenizer.json, and
    /// one or more .safetensors weight files.
    #[arg(long, value_name = "PATH")]
    local_path: Option<PathBuf>,

    /// Maximum number of tokens to generate.
    ///
    /// Higher values allow longer arguments but increase latency.
    /// Typical steel-man arguments are 300–600 tokens.
    #[arg(long, default_value_t = 512, value_name = "N")]
    max_tokens: usize,

    /// Sampling temperature.
    ///
    /// 0.0 = greedy (deterministic, conservative)
    /// 0.7 = balanced (recommended for argumentation)
    /// 1.0 = creative (more varied, less reliable)
    #[arg(long, default_value_t = 0.7, value_name = "TEMP")]
    temperature: f64,

    /// Top-p (nucleus) sampling threshold.
    ///
    /// 0.9 = sample from top 90% of probability mass (recommended)
    /// 1.0 = disabled (sample from full distribution)
    /// 0.5 = conservative (less diverse outputs)
    #[arg(long, default_value_t = 0.9, value_name = "P")]
    top_p: f64,

    /// Random seed for reproducible outputs.
    ///
    /// If not set, uses OS entropy (different output each run).
    #[arg(long, value_name = "SEED")]
    seed: Option<u64>,

    /// Model weight dtype.
    ///
    /// "auto" selects BF16 on CUDA, F32 on CPU/Metal (recommended).
    /// Override only if you have specific precision requirements.
    #[arg(long, default_value = "auto", value_name = "DTYPE")]
    dtype: DtypeArg,

    /// Suppress streaming output (print full response when complete instead).
    ///
    /// Useful when piping output to another program.
    #[arg(long, default_value_t = false)]
    no_stream: bool,

    /// Enable verbose/debug logging.
    #[arg(long, short = 'v', default_value_t = false)]
    verbose: bool,
}

/// Dtype argument enum for clap.
#[derive(Debug, Clone, ValueEnum)]
enum DtypeArg {
    Auto,
    F32,
    F16,
    Bf16,
}

impl DtypeArg {
    fn to_candle_dtype(&self) -> Option<DType> {
        match self {
            Self::Auto => None,
            Self::F32  => Some(DType::F32),
            Self::F16  => Some(DType::F16),
            Self::Bf16 => Some(DType::BF16),
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialise tracing — respects RUST_LOG env var.
    // In non-verbose mode: only WARN and ERROR.
    // In verbose mode: DEBUG and above.
    let log_level = if cli.verbose { "debug" } else { "warn" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level)),
        )
        .with_target(false)
        .without_time()
        .init();

    // Validate that we have a model source.
    if cli.model_id.is_none() && cli.local_path.is_none() {
        anyhow::bail!(
            "no model specified.\n\n\
             Use --model-id your-username/debatelm (HuggingFace Hub)\n\
             or --local-path /path/to/model/ (local directory)\n\
             or set DEBATELM_MODEL_ID environment variable"
        );
    }

    // Validate that interactive and argument are not both set.
    if cli.interactive && cli.argument.is_some() {
        anyhow::bail!(
            "--interactive and --argument are mutually exclusive. \
             Use one or the other."
        );
    }

    // Load the engine.
    let dtype_override = cli.dtype.to_candle_dtype();

    let mut engine = if let Some(local_path) = &cli.local_path {
        println!("Loading DebateLM from local path: {}", local_path.display());
        Engine::load_local(local_path, dtype_override)
            .context("failed to load model from local path")?
    } else {
        let model_id = cli
            .model_id
            .as_deref()
            .unwrap_or("your-username/debatelm");
        Engine::load(model_id, dtype_override)
            .await
            .with_context(|| format!("failed to load model from Hub: {model_id}"))?
    };

    // Build generation config.
    let gen_config = GenerationConfig {
        max_new_tokens: cli.max_tokens,
        temperature:    cli.temperature,
        top_p:          cli.top_p,
        seed:           cli.seed,
        stream:         !cli.no_stream,
    };

    // Dispatch to single-shot or interactive mode.
    if cli.interactive {
        run_interactive(&mut engine, &gen_config)
            .context("interactive session terminated with error")?;
    } else if let Some(argument) = cli.argument {
        run_single(&mut engine, &argument, &gen_config)
            .context("single-shot generation failed")?;
    }
    // The `arg_required_else_help = true` flag on the Cli struct ensures
    // we never reach here without one of the above branches being taken.

    Ok(())
}

// ---------------------------------------------------------------------------
// Single-shot mode
// ---------------------------------------------------------------------------

fn run_single(engine: &mut Engine, argument: &str, gen_config: &GenerationConfig) -> Result<()> {
    println!("\n🎯 Steel-manning: \"{argument}\"\n");
    println!("{}", "─".repeat(60));
    println!();

    if gen_config.stream {
        // In stream mode, inference.rs prints tokens directly and returns
        // the full string. We just need to print the closing banner.
        let _result = engine
            .steelman(argument, gen_config)
            .context("generation failed")?;
        println!();
        println!("{}", "─".repeat(60));
    } else {
        // Non-streaming: collect full response, then print.
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} Generating…")
                .expect("valid spinner template"), // compile-time constant
        );
        spinner.enable_steady_tick(std::time::Duration::from_millis(80));

        let result = engine
            .steelman(argument, gen_config)
            .context("generation failed")?;

        spinner.finish_and_clear();
        println!("{result}");
        println!();
        println!("{}", "─".repeat(60));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Interactive REPL mode
// ---------------------------------------------------------------------------

fn run_interactive(engine: &mut Engine, gen_config: &GenerationConfig) -> Result<()> {
    print_banner();

    let stdin  = io::stdin();
    let stdout = io::stdout();

    loop {
        // Print prompt.
        {
            let mut out = stdout.lock();
            write!(out, "\n> ").context("stdout write failed")?;
            out.flush().context("stdout flush failed")?;
        }

        // Read line.
        let mut line = String::new();
        let bytes_read = stdin
            .lock()
            .read_line(&mut line)
            .context("failed to read from stdin")?;

        // EOF (Ctrl+D) → clean exit.
        if bytes_read == 0 {
            println!("\n\nGoodbye.");
            break;
        }

        let argument = line.trim();

        // Handle special commands.
        match argument.to_lowercase().as_str() {
            "exit" | "quit" | "q" => {
                println!("Goodbye.");
                break;
            }
            "help" | "?" => {
                print_help();
                continue;
            }
            "" => {
                // Empty input — re-prompt without error.
                continue;
            }
            _ => {}
        }

        // Generate steel-man.
        println!("\n{}", "─".repeat(60));
        println!();

        match engine.steelman(argument, gen_config) {
            Ok(_) => {
                // In stream mode, output was already printed.
                // In non-stream mode, we'd print here — but interactive
                // mode always streams for responsiveness.
                println!();
                println!("{}", "─".repeat(60));
            }
            Err(e) => {
                // Print error but don't crash the REPL — let user try again.
                eprintln!("\n❌ Error: {e:#}");
                eprintln!("Try a shorter argument or adjust --max-tokens.");
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

fn print_banner() {
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║                      DebateLM v{}{}║", env!("CARGO_PKG_VERSION"), " ".repeat(42 - env!("CARGO_PKG_VERSION").len()));
    println!("║         Steel-man any argument. No strawmen allowed.     ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Enter an argument to steel-man, or type 'help' / 'exit' ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
}

fn print_help() {
    println!();
    println!("DEBATELM INTERACTIVE MODE");
    println!("─────────────────────────");
    println!("Type any argument or position and press Enter.");
    println!("DebateLM will generate the strongest possible counter-argument.");
    println!();
    println!("Commands:");
    println!("  exit / quit / q  — exit DebateLM");
    println!("  help / ?         — show this message");
    println!();
    println!("Examples:");
    println!("  Universal Basic Income reduces the incentive to work");
    println!("  Open source software is less secure than proprietary software");
    println!("  Stricter gun control reduces violent crime");
    println!("  Remote work permanently damages company culture");
    println!();
}