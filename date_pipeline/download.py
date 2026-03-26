#!/usr/bin/env python3
"""
Download all DebateLM training datasets from HuggingFace Hub and other sources.

Usage:
    python data_pipeline/download.py
    python data_pipeline/download.py --data-dir data/raw --hf-token $HF_TOKEN

All datasets are saved to disk as JSONL files. Existing files are skipped
unless --force is passed. This makes the script idempotent — safe to re-run.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Each entry defines how to fetch one dataset.
# "loader" is either "hf_datasets" (uses the datasets library) or "custom"
# (uses a bespoke fetch function defined below).
#
# "text_fields" defines which JSON keys contain the text we care about.
# The format.py script uses these to produce the unified JSONL format.

DATASETS: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------
    # Primary datasets (from doc section 3.1)
    # ------------------------------------------------------------------
    {
        "name":         "changemyview",
        "loader":       "hf_datasets",
        "hf_id":        "MegrezZhu/changemyview",
        "split":        "train",
        "config":       None,
        "text_fields":  ["title", "selftext"],
        "output_file":  "changemyview.jsonl",
        "description":  "Reddit r/ChangeMyView — humans changing opinions",
    },
    {
        "name":         "argument_mining",
        "loader":       "hf_datasets",
        "hf_id":        "debatelab/argument-mining",
        "split":        "train",
        "config":       None,
        "text_fields":  ["premise", "claim"],
        "output_file":  "argument_mining.jsonl",
        "description":  "Annotated argument structures: claims, premises, rebuttals",
    },
    {
        "name":         "fever",
        "loader":       "hf_datasets",
        "hf_id":        "fever",
        "split":        "train",
        "config":       "v1.0",
        "text_fields":  ["claim", "evidence_sentence"],
        "output_file":  "fever.jsonl",
        "description":  "Claims with supporting and refuting evidence pairs",
    },
    {
        "name":         "arguana",
        "loader":       "hf_datasets",
        "hf_id":        "BeIR/arguana",
        "split":        "test",
        "config":       None,
        "text_fields":  ["text"],
        "output_file":  "arguana.jsonl",
        "description":  "Counter-argument retrieval dataset",
    },
    {
        "name":         "philosophy_se",
        "loader":       "hf_datasets",
        "hf_id":        "philstackexchange",
        "split":        "train",
        "config":       None,
        "text_fields":  ["question", "answer"],
        "output_file":  "philosophy_se.jsonl",
        "description":  "Philosophy StackExchange Q&A pairs",
    },
    {
        "name":         "eli5",
        "loader":       "hf_datasets",
        "hf_id":        "eli5",
        "split":        "train_eli5",
        "config":       None,
        "text_fields":  ["title", "answers"],
        "output_file":  "eli5.jsonl",
        "description":  "Explanation quality training data",
    },
    {
        "name":         "kialo",
        "loader":       "hf_datasets",
        "hf_id":        "Anthropic/kialo-arguments",
        "split":        "train",
        "config":       None,
        "text_fields":  ["pro_argument", "con_argument", "thesis"],
        "output_file":  "kialo.jsonl",
        "description":  "Structured pro/con debate trees on thousands of topics",
    },
    {
        "name":         "un_debates",
        "loader":       "hf_datasets",
        "hf_id":        "qanastek/un-general-debates",
        "split":        "train",
        "config":       None,
        "text_fields":  ["text"],
        "output_file":  "un_debates.jsonl",
        "description":  "UN General Debate corpus — formal policy debate language",
    },
    {
        "name":         "persuasive_essays",
        "loader":       "hf_datasets",
        "hf_id":        "ukp-project/arguana-counter",
        "split":        "train",
        "config":       None,
        "text_fields":  ["argument", "counter_argument"],
        "output_file":  "persuasive_essays.jsonl",
        "description":  "Persuasive essays with argument component annotations",
    },
    {
        "name":         "legal_briefs",
        "loader":       "hf_datasets",
        "hf_id":        "pile-of-law/pile-of-law",
        "split":        "train",
        "config":       "us_courts",
        "text_fields":  ["text"],
        "output_file":  "legal_briefs.jsonl",
        "description":  "Legal briefs — professional steel-manning context",
        # Large dataset: limit to 50k examples for manageable training
        "max_examples": 50_000,
    },
]


# ---------------------------------------------------------------------------
# Fetch functions
# ---------------------------------------------------------------------------

def fetch_hf_dataset(
        dataset_cfg: Dict[str, Any],
        output_path: Path,
        hf_token:    Optional[str],
        max_examples: Optional[int],
) -> int:
    """
    Download a HuggingFace dataset and write it as JSONL.

    Returns the number of examples written.
    """
    from datasets import load_dataset  # type: ignore

    hf_id  = dataset_cfg["hf_id"]
    split  = dataset_cfg.get("split", "train")
    config = dataset_cfg.get("config")

    log.info(f"  Downloading {hf_id} (split={split}, config={config})…")

    try:
        load_kwargs: Dict[str, Any] = {
            "path":  hf_id,
            "split": split,
        }
        if config:
            load_kwargs["name"] = config
        if hf_token:
            load_kwargs["token"] = hf_token

        dataset = load_dataset(**load_kwargs)
    except Exception as e:
        log.error(f"  ✗ Failed to download {hf_id}: {e}")
        log.error("    Skipping — re-run after fixing the error above.")
        return 0

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            if max_examples and i >= max_examples:
                break
            # Write the raw example as-is; format.py will shape it.
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    log.info(f"  ✓ {count:,} examples written to {output_path.name}")
    return count


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def download_all(
        data_dir:     Path,
        hf_token:     Optional[str],
        force:        bool,
        skip_on_fail: bool,
) -> Dict[str, int]:
    """
    Download all registered datasets.

    Returns a dict mapping dataset name → example count.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, int] = {}
    total_start = time.time()

    for i, cfg in enumerate(DATASETS, 1):
        name        = cfg["name"]
        output_file = data_dir / cfg["output_file"]
        max_ex      = cfg.get("max_examples")

        log.info(f"[{i}/{len(DATASETS)}] {name} — {cfg['description']}")

        if output_file.exists() and not force:
            size = output_file.stat().st_size
            log.info(f"  ↷ Already exists ({size / 1024**2:.1f} MB) — skipping. Use --force to re-download.")
            # Count lines for reporting
            with output_file.open() as f:
                count = sum(1 for _ in f)
            results[name] = count
            continue

        loader = cfg.get("loader", "hf_datasets")

        try:
            if loader == "hf_datasets":
                count = fetch_hf_dataset(cfg, output_file, hf_token, max_ex)
            else:
                log.warning(f"  Unknown loader '{loader}' for {name} — skipping")
                count = 0
        except Exception as e:
            log.error(f"  ✗ Unexpected error downloading {name}: {e}")
            if not skip_on_fail:
                raise
            count = 0

        results[name] = count

    elapsed = time.time() - total_start
    total_examples = sum(results.values())

    log.info("─" * 60)
    log.info(f"Download complete in {elapsed:.1f}s")
    log.info(f"Total examples: {total_examples:,}")
    log.info("Per-dataset breakdown:")
    for name, count in results.items():
        status = "✓" if count > 0 else "✗"
        log.info(f"  {status} {name:<30} {count:>10,}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DebateLM training datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save raw dataset files",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HUGGINGFACE_TOKEN"),
        help="HuggingFace API token (or set HUGGINGFACE_TOKEN env var)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download datasets even if files already exist",
    )
    parser.add_argument(
        "--skip-on-fail",
        action="store_true",
        default=True,
        help="Continue downloading remaining datasets if one fails",
    )
    parser.add_argument(
        "--dataset",
        choices=[d["name"] for d in DATASETS],
        help="Download only a specific dataset (default: all)",
    )

    args = parser.parse_args()

    # Filter to single dataset if requested.
    if args.dataset:
        global DATASETS
        DATASETS = [d for d in DATASETS if d["name"] == args.dataset]
        if not DATASETS:
            log.error(f"Dataset '{args.dataset}' not found in registry.")
            sys.exit(1)

    results = download_all(
        data_dir     = args.data_dir,
        hf_token     = args.hf_token,
        force        = args.force,
        skip_on_fail = args.skip_on_fail,
    )

    # Exit with error code if no data was downloaded.
    if all(count == 0 for count in results.values()):
        log.error("No data was downloaded. Check your HuggingFace token and network.")
        sys.exit(1)


if __name__ == "__main__":
    main()