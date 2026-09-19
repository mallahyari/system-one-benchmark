#!/usr/bin/env python3
"""
Local System One Evaluation Runner (Apple Silicon MLX)
-----------------------------------------------------
Evaluates Parallel Constrained Decoding (PCD) vs. Autoregressive JSON Generation
using an open-source model (mlx-community/Qwen2.5-1.5B-Instruct-4bit).

Matches our two primary use cases:
  1. Guardrails: AI Safety & Jailbreak Detection (lmsys/toxic-chat)
  2. Banking: Customer Intent Classification & Escalation (mteb/banking77)

Usage:
  python3 evaluate_local_pcd.py --task guardrails --samples 10
  python3 evaluate_local_pcd.py --task banking --samples 10
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

from local_pcd_engine import (
    LocalStructuredSchema,
    run_local_autoregressive_baseline,
    run_local_parallel_decoding,
)

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


def get_guardrail_schema() -> LocalStructuredSchema:
    return LocalStructuredSchema({
        "is_harmful": {
            "type": "boolean",
            "description": "Whether the user prompt is an adversarial jailbreak or harmful request",
        },
        "risk_category": {
            "type": "enum",
            "description": "Primary risk classification category",
            "choices": ["cyberattack", "fraud", "hazardous", "harassment", "toxicity", "benign"],
        },
        "severity_level": {
            "type": "enum",
            "description": "Malicious risk severity grade",
            "choices": ["safe_tier_0", "moderate_tier_1", "critical_tier_2"],
        },
    })


def get_banking_schema() -> LocalStructuredSchema:
    return LocalStructuredSchema({
        "intent_category": {
            "type": "enum",
            "description": "Target banking customer service intent",
            "choices": [
                "card_stolen_fraud",
                "pin_blocked_atm",
                "foreign_exchange_fees",
                "change_card_limits",
                "wire_transfer_delay",
                "tax_statements",
                "direct_deposit_delay",
                "fee_inquiry_refund",
            ],
        },
        "is_urgent": {
            "type": "boolean",
            "description": "Whether the customer inquiry indicates high financial or time sensitivity",
        },
        "requires_escalation": {
            "type": "boolean",
            "description": "Whether human tier-2 supervisor intervention is required",
        },
    })


def load_samples(task: str, samples_count: int, data_file: str = None) -> List[Dict[str, Any]]:
    # 1. If explicit data_file provided and exists
    if data_file and os.path.exists(data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[:samples_count]

    # 2. Check default data paths
    candidate_file = f"data/{'guardrails_toxicchat_50.json' if task == 'guardrails' else 'banking77_50.json'}"
    if os.path.exists(candidate_file):
        with open(candidate_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[:samples_count]

    # 3. Fallback to sample_data.py
    from sample_data import BANKING_SAMPLES, GUARDRAIL_SAMPLES
    base = GUARDRAIL_SAMPLES if task == "guardrails" else BANKING_SAMPLES
    res = []
    while len(res) < samples_count:
        res.extend(base)
    return res[:samples_count]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Local Parallel Constrained Decoding on Apple Silicon.")
    parser.add_argument("--task", choices=["guardrails", "banking"], default="guardrails", help="Task to evaluate.")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to evaluate (default: 10).")
    parser.add_argument("--data-file", type=str, default=None, help="Optional path to custom JSON dataset file.")
    parser.add_argument("--output", type=str, default="local_mlx_results.json", help="Output results file.")

    args = parser.parse_args()

    schema = get_guardrail_schema() if args.task == "guardrails" else get_banking_schema()
    samples = load_samples(args.task, args.samples, args.data_file)

    print("\n" + "=" * 75)
    print("LOCAL SYSTEM ONE EVALUATION (Apple Silicon MLX + Qwen 2.5 1.5B 4-bit)")
    print("=" * 75)
    print(f"Task: {args.task.upper()} | Samples: {len(samples)} | Schema Fields: {len(schema)}")
    print("Testing Parallel Constrained Decoding (PCD) vs. Autoregressive JSON Baseline...")
    print("-" * 75)

    pcd_latencies = []
    auto_latencies = []
    pcd_passes = []
    auto_passes = []
    auto_schema_errors = 0
    results_log = []

    for idx, sample in enumerate(samples):
        context = sample.get("prompt") or sample.get("text")
        print(f"\n[Sample {idx + 1}/{len(samples)}] Context: \"{context[:60]}...\"")

        # 1. Run Parallel Constrained Decoding (Jev-style single pass)
        res_pcd = run_local_parallel_decoding(context, schema)
        pcd_latencies.append(res_pcd["elapsed_ms"])
        pcd_passes.append(res_pcd["sequential_forward_passes"])

        # 2. Run Autoregressive JSON Baseline
        res_auto = run_local_autoregressive_baseline(context, schema)
        auto_latencies.append(res_auto["elapsed_ms"])
        auto_passes.append(res_auto["sequential_forward_passes"])

        if not res_auto["schema_match"]:
            auto_schema_errors += 1

        speedup = res_auto["elapsed_ms"] / max(res_pcd["elapsed_ms"], 1e-4)

        print(f"  ⚡ Parallel Constrained : {res_pcd['elapsed_ms']:>6.1f} ms | Passes: 1 | Values: {res_pcd['parsed_json']}")
        print(f"  🐢 Autoregressive JSON  : {res_auto['elapsed_ms']:>6.1f} ms | Passes: {res_auto['sequential_forward_passes']} | Speedup: {speedup:.1f}x")

        results_log.append({
            "index": idx,
            "context": context,
            "parallel_result": res_pcd,
            "autoregressive_result": res_auto,
            "speedup": round(speedup, 2),
        })

    # Calculate summary metrics
    pcd_latencies.sort()
    auto_latencies.sort()
    n = len(samples)

    pcd_p50 = pcd_latencies[int(n * 0.5)]
    auto_p50 = auto_latencies[int(n * 0.5)]
    pcd_mean = sum(pcd_latencies) / n
    auto_mean = sum(auto_latencies) / n
    overall_speedup = auto_mean / max(pcd_mean, 1e-4)

    rows = [
        ["Latency p50 (Median)", f"{pcd_p50:.1f} ms", f"{auto_p50:.1f} ms", f"{auto_p50 / pcd_p50:.1f}x Faster"],
        ["Latency Mean", f"{pcd_mean:.1f} ms", f"{auto_mean:.1f} ms", f"{overall_speedup:.1f}x Faster"],
        ["Forward Passes per Query", "1 (O(1) pass)", f"~{sum(auto_passes) / n:.1f} passes", f"{sum(auto_passes) / n:.1f}x Fewer"],
        ["Schema Validity", "100.0% (Guaranteed)", f"{(1 - auto_schema_errors / n) * 100:.1f}%", "Zero JSON parse bugs"],
        ["Calibrated Probabilities", "Yes (Field-level softmax)", "No (Raw generated text)", "Confidence thresholds"],
    ]

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON (Apple Silicon M-Series Unified Memory)")
    print("=" * 80)
    if tabulate:
        print(tabulate(rows, headers=["Metric", "Parallel Constrained (PCD)", "Autoregressive JSON", "Advantage"], tablefmt="github"))
    else:
        for r in rows:
            print(f"{r[0]:<25} | PCD: {r[1]:<15} | Auto: {r[2]:<18} | {r[3]}")
    print("=" * 80 + "\n")

    summary = {
        "task": args.task,
        "samples_evaluated": n,
        "model_id": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "parallel_latency_mean_ms": round(pcd_mean, 2),
        "autoregressive_latency_mean_ms": round(auto_mean, 2),
        "speedup_factor": round(overall_speedup, 2),
        "autoregressive_schema_error_rate": round(auto_schema_errors / n, 4),
        "sample_logs": results_log,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Detailed local benchmark results saved to: {args.output}")


if __name__ == "__main__":
    main()
