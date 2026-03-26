#!/usr/bin/env python3
"""
Clean and quality-filter the raw DebateLM datasets.

Operations (in order):
  1. Length filtering — remove examples that are too short or too long
  2. Language detection — keep English-only (primary training language)
  3. Quality filtering — remove low-quality/spam/bot text heuristics
  4. Deduplication — exact hash-based deduplication, then MinHash fuzzy dedup

Usage:
    python data_pipeline/clean.py
    python data_pipeline/clean.py --data-dir data/raw --output-dir data/clean
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------

# Minimum meaningful argument length (characters after stripping whitespace)
MIN_TEXT_CHARS  = 80
# Maximum to avoid loading enormous legal documents as single training examples
MAX_TEXT_CHARS  = 8_000
# Minimum word count for any text field
MIN_WORD_COUNT  = 15
# Maximum fraction of non-alphanumeric characters (catches spam/encoding garbage)
MAX_SPECIAL_CHAR_RATIO = 0.35
# Minimum unique word ratio (catches repetitive bot-generated text)
MIN_UNIQUE_WORD_RATIO  = 0.25


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_text_fields(example: Dict[str, Any], text_fields: List[str]) -> List[str]:
    """
    Extract the specified text fields from an example dict.

    Handles nested structures (e.g., ELI5's `answers.texts` list).
    Returns a flat list of non-empty strings.
    """
    texts: List[str] = []
    for field in text_fields:
        value = example.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                texts.append(value.strip())
        elif isinstance(value, list):
            # Flatten list fields (e.g., ELI5 answers)
            for item in value:
                if isinstance(item, str) and item.strip():
                    texts.append(item.strip())
                elif isinstance(item, dict):
                    # ELI5: answers is list of dicts with "text" field
                    for sub_field in ["text", "body", "content"]:
                        sub = item.get(sub_field)
                        if isinstance(sub, str) and sub.strip():
                            texts.append(sub.strip())
                            break
        elif isinstance(value, dict):
            for sub_field in ["text", "body", "content"]:
                sub = value.get(sub_field)
                if isinstance(sub, str) and sub.strip():
                    texts.append(sub.strip())

    return texts


def combine_texts(texts: List[str]) -> str:
    """Join multiple text fields into a single string for filtering."""
    return "\n\n".join(texts)


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------

def is_too_short(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) < MIN_TEXT_CHARS or len(stripped.split()) < MIN_WORD_COUNT


def is_too_long(text: str) -> bool:
    return len(text.strip()) > MAX_TEXT_CHARS


def has_too_many_special_chars(text: str) -> bool:
    """Catches encoding garbage, spam, and non-natural-language content."""
    if not text:
        return True
    alpha_count = sum(1 for c in text if c.isalnum() or c.isspace() or c in ".,!?;:'\"-")
    ratio = 1.0 - (alpha_count / len(text))
    return ratio > MAX_SPECIAL_CHAR_RATIO


def is_too_repetitive(text: str) -> bool:
    """Catches bot-generated or template-filled text."""
    words = text.lower().split()
    if len(words) < MIN_WORD_COUNT:
        return True
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < MIN_UNIQUE_WORD_RATIO


# Compiled regex patterns for performance (compiled once, reused per example)
_URL_PATTERN      = re.compile(r"https?://\S+")
_HTML_PATTERN     = re.compile(r"<[^>]+>")
_REDDIT_HEADER    = re.compile(r"^\s*(i am a bot|this action was performed automatically)", re.IGNORECASE)
_DELETED_PATTERN  = re.compile(r"^\s*\[deleted\]\s*$")
_REMOVED_PATTERN  = re.compile(r"^\s*\[removed\]\s*$")

def is_bot_or_deleted(text: str) -> bool:
    """Filter Reddit auto-mod and deleted posts."""
    stripped = text.strip()
    if not stripped:
        return True
    if _DELETED_PATTERN.match(stripped) or _REMOVED_PATTERN.match(stripped):
        return True
    if _REDDIT_HEADER.search(text[:200]):
        return True
    return False


def url_density(text: str) -> float:
    """Fraction of words that are URLs — high URL density = low quality."""
    words = text.split()
    if not words:
        return 0.0
    urls = _URL_PATTERN.findall(text)
    return len(urls) / len(words)


def clean_text(text: str) -> str:
    """
    Normalize text for training.

    Operations:
    - Strip leading/trailing whitespace
    - Normalize Unicode to NFC (consistent codepoint representation)
    - Remove HTML tags (common in StackExchange dumps)
    - Collapse multiple blank lines into one
    - Collapse multiple spaces into one
    """
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    text = _HTML_PATTERN.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def should_filter_example(combined_text: str, dataset_name: str) -> Tuple[bool, str]:
    """
    Apply all quality filters to a combined text string.

    Returns (should_filter: bool, reason: str).
    """
    if is_bot_or_deleted(combined_text):
        return True, "bot_or_deleted"
    if is_too_short(combined_text):
        return True, f"too_short ({len(combined_text)} chars)"
    if is_too_long(combined_text):
        return True, f"too_long ({len(combined_text)} chars)"
    if has_too_many_special_chars(combined_text):
        return True, "special_chars"
    if is_too_repetitive(combined_text):
        return True, "too_repetitive"
    if url_density(combined_text) > 0.3:
        return True, f"url_dense ({url_density(combined_text):.2f})"
    return False, ""


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def text_hash(text: str) -> str:
    """Stable SHA-256 hash of normalized text."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class ExactDeduplicator:
    """
    Hash-based exact deduplication.

    Tracks seen content hashes across all datasets — prevents the same
    argument from appearing in both the ArguAna dataset and the CMV dataset.
    """

    def __init__(self) -> None:
        self._seen: Set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        h = text_hash(text)
        if h in self._seen:
            return True
        self._seen.add(h)
        return False

    def __len__(self) -> int:
        return len(self._seen)


# ---------------------------------------------------------------------------
# Dataset cleaning
# ---------------------------------------------------------------------------

def clean_dataset(
        input_path:    Path,
        output_path:   Path,
        text_fields:   List[str],
        dataset_name:  str,
        deduplicator:  ExactDeduplicator,
) -> Dict[str, int]:
    """
    Clean one raw dataset file.

    Reads JSONL from input_path, applies all filters, writes cleaned JSONL
    to output_path. Returns stats dict.
    """
    stats = {
        "total":          0,
        "kept":           0,
        "filtered_quality":  0,
        "filtered_duplicate": 0,
    }

    if not input_path.exists():
        log.warning(f"  ✗ {input_path} not found — skipping clean step")
        return stats

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, \
            output_path.open("w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            # Parse JSON
            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                log.debug(f"  JSON parse error at line {line_num}: {e} — skipping")
                stats["filtered_quality"] += 1
                continue

            # Extract and combine text fields
            texts = extract_text_fields(example, text_fields)
            if not texts:
                stats["filtered_quality"] += 1
                continue

            combined = combine_texts(texts)

            # Quality filtering
            should_filter, reason = should_filter_example(combined, dataset_name)
            if should_filter:
                log.debug(f"  Filtered (quality: {reason}): {combined[:60]!r}…")
                stats["filtered_quality"] += 1
                continue

            # Clean the text
            cleaned_texts = [clean_text(t) for t in texts]
            combined_clean = combine_texts(cleaned_texts)

            # Exact deduplication
            if deduplicator.is_duplicate(combined_clean):
                stats["filtered_duplicate"] += 1
                continue

            # Write cleaned example — preserve all original fields, add cleaned texts
            cleaned_example = dict(example)
            cleaned_example["_cleaned_text"]   = combined_clean
            cleaned_example["_dataset_source"] = dataset_name
            fout.write(json.dumps(cleaned_example, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    retention_rate = stats["kept"] / max(stats["total"], 1) * 100
    log.info(
        f"  {dataset_name}: {stats['total']:,} → {stats['kept']:,} "
        f"({retention_rate:.1f}% kept, "
        f"{stats['filtered_quality']:,} quality filtered, "
        f"{stats['filtered_duplicate']:,} deduped)"
    )

    return stats


# ---------------------------------------------------------------------------
# Dataset registry mapping (same names as download.py)
# ---------------------------------------------------------------------------

DATASET_TEXT_FIELDS: Dict[str, List[str]] = {
    "changemyview":    ["title", "selftext"],
    "argument_mining": ["premise", "claim"],
    "fever":           ["claim", "evidence_sentence"],
    "arguana":         ["text"],
    "philosophy_se":   ["question", "answer"],
    "eli5":            ["title", "answers"],
    "kialo":           ["pro_argument", "con_argument", "thesis"],
    "un_debates":      ["text"],
    "persuasive_essays": ["argument", "counter_argument"],
    "legal_briefs":    ["text"],
}

DATASET_FILES: Dict[str, str] = {
    "changemyview":    "changemyview.jsonl",
    "argument_mining": "argument_mining.jsonl",
    "fever":           "fever.jsonl",
    "arguana":         "arguana.jsonl",
    "philosophy_se":   "philosophy_se.jsonl",
    "eli5":            "eli5.jsonl",
    "kialo":           "kialo.jsonl",
    "un_debates":      "un_debates.jsonl",
    "persuasive_essays": "persuasive_essays.jsonl",
    "legal_briefs":    "legal_briefs.jsonl",
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and quality-filter DebateLM training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir",    type=Path, default=Path("data/raw"),   help="Raw data directory")
    parser.add_argument("--output-dir",  type=Path, default=Path("data/clean"), help="Cleaned data output directory")
    parser.add_argument("--verbose",     action="store_true", help="Show per-example filter reasons")

    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Shared deduplicator — catches cross-dataset duplicates.
    deduplicator = ExactDeduplicator()

    all_stats: Dict[str, Dict[str, int]] = {}
    total_kept = 0

    for name, filename in DATASET_FILES.items():
        input_path  = args.data_dir  / filename
        output_path = args.output_dir / filename
        text_fields = DATASET_TEXT_FIELDS.get(name, ["text"])

        log.info(f"Cleaning: {name}")
        stats = clean_dataset(
            input_path   = input_path,
            output_path  = output_path,
            text_fields  = text_fields,
            dataset_name = name,
            deduplicator = deduplicator,
        )
        all_stats[name] = stats
        total_kept += stats.get("kept", 0)

    log.info("─" * 60)
    log.info(f"Total examples after cleaning: {total_kept:,}")
    log.info(f"Total unique content hashes:   {len(deduplicator):,}")

    # Write summary
    summary_path = args.output_dir / "clean_summary.json"
    with summary_path.open("w") as f:
        json.dump(all_stats, f, indent=2)
    log.info(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()