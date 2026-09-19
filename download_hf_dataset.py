#!/usr/bin/env python3
"""
Hugging Face Dataset Downloader for Jev System One Benchmarking
--------------------------------------------------------------
Downloads and formats real-world evaluation datasets from Hugging Face:
  1. AI Safety & Jailbreaks: 'lmsys/toxic-chat' (ungated, real-world LMSYS user prompts)
     - Captures prompt, jailbreaking flag, toxicity flag, and safety categories
  2. High-Cardinality Intent Triage: 'mteb/banking77' (ungated, 77 fine-grained classes)
     - Captures customer inquiry text, intent label ID, and human-readable intent name

Usage:
  python3 download_hf_dataset.py --task guardrails --samples 100
  python3 download_hf_dataset.py --task banking --samples 100
  python3 download_hf_dataset.py --task all --samples 200
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

try:
    from datasets import load_dataset
except ImportError:
    print("[!] Error: 'datasets' package is required. Install via `pip install datasets`.")
    sys.exit(1)


def download_guardrails(num_samples: int, split: str = "test", output_dir: str = "data") -> str:
    """Downloads AI Safety & Jailbreak prompts from lmsys/toxic-chat."""
    print(f"[*] Fetching '{split}' split from 'lmsys/toxic-chat' (config: toxicchat0124)...")
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split=split, streaming=True)

    formatted_samples: List[Dict[str, Any]] = []
    harmful_count = 0
    jailbreak_count = 0

    print("[*] Filtering and formatting prompt safety samples...")
    for idx, row in enumerate(ds):
        user_input = row.get("user_input", "").strip()
        if not user_input or len(user_input) < 10:
            continue

        is_jailbreak = int(row.get("jailbreaking", 0))
        is_toxic = int(row.get("toxicity", 0))
        is_harmful = 1 if (is_jailbreak or is_toxic) else 0

        if is_jailbreak:
            jailbreak_count += 1
        if is_harmful:
            harmful_count += 1

        category = "benign"
        if is_jailbreak:
            category = "jailbreak"
        elif is_toxic:
            category = "toxicity"

        formatted_samples.append({
            "id": row.get("conv_id", f"sample_{idx}"),
            "prompt": user_input,
            "is_harmful": is_harmful,
            "is_jailbreak": is_jailbreak,
            "is_toxic": is_toxic,
            "category": category,
            "severity": 2 if is_jailbreak else (1 if is_toxic else 0),
        })

        if len(formatted_samples) >= num_samples:
            break

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"guardrails_toxicchat_{len(formatted_samples)}.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(formatted_samples, f, indent=2, ensure_ascii=False)

    print(f"[+] Downloaded {len(formatted_samples)} guardrail samples.")
    print(f"    - Harmful/Adversarial: {harmful_count}")
    print(f"    - Jailbreak Attempts:  {jailbreak_count}")
    print(f"    - Benign / Safe:       {len(formatted_samples) - harmful_count}")
    print(f"[+] Saved to: {out_file}")
    return out_file


def download_banking(num_samples: int, split: str = "test", output_dir: str = "data") -> str:
    """Downloads 77-class customer intent classification dataset from mteb/banking77."""
    print(f"[*] Fetching '{split}' split from 'mteb/banking77'...")
    ds = load_dataset("mteb/banking77", split=split, streaming=False)

    formatted_samples: List[Dict[str, Any]] = []
    total_to_take = min(num_samples, len(ds))

    print(f"[*] Processing {total_to_take} customer banking inquiries...")
    unique_intents = set()

    for idx in range(total_to_take):
        row = ds[idx]
        intent_text = row["label_text"]
        unique_intents.add(intent_text)

        # Infer basic urgency indicator based on intent semantics
        is_urgent = any(word in intent_text.lower() for word in ["stolen", "lost", "fraud", "blocked", "compromised"])

        formatted_samples.append({
            "id": f"bank_{idx}",
            "text": row["text"],
            "intent": intent_text,
            "intent_id": int(row["label"]),
            "urgency": 2 if is_urgent else 0,
            "escalate": 1 if is_urgent else 0,
        })

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"banking77_{len(formatted_samples)}.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(formatted_samples, f, indent=2, ensure_ascii=False)

    print(f"[+] Downloaded {len(formatted_samples)} banking inquiries.")
    print(f"    - Unique Intent Classes Covered: {len(unique_intents)} / 77")
    print(f"[+] Saved to: {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Download and format Hugging Face datasets for Jev benchmarks.")
    parser.add_argument("--task", choices=["guardrails", "banking", "all"], default="all",
                        help="Dataset task to download: guardrails, banking, or all (default: all).")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to download per dataset (default: 100).")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to pull from (default: test).")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory to save formatted JSON (default: data).")

    args = parser.parse_args()

    print("=" * 65)
    print("HUGGING FACE DATASET DOWNLOADER FOR JEV EVALUATION")
    print("=" * 65)

    if args.task in ("guardrails", "all"):
        download_guardrails(num_samples=args.samples, split=args.split, output_dir=args.output_dir)
        print("-" * 65)

    if args.task in ("banking", "all"):
        download_banking(num_samples=args.samples, split=args.split, output_dir=args.output_dir)
        print("-" * 65)

    print("[✔] All requested datasets successfully downloaded and structured in ./data/")


if __name__ == "__main__":
    main()
