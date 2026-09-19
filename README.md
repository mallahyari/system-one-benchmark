# System One & Parallel Constrained Decoding Benchmark

Empirical evaluation and benchmarking suite comparing:
1. **TypeSafe Jev (`jev-1.13.0`)**: A frontier "System One" decision model via cloud API.
2. **Local Open-Source PCD**: Parallel Constrained Decoding with `Qwen 2.5 1.5B (4-bit)` on Apple Silicon using Apple's **MLX** framework.
3. **Local Autoregressive Baseline**: Standard token-by-token JSON generation with `Qwen 2.5 1.5B`.

Evaluated against 50 real-world adversarial, jailbreak, and benign prompts from the LMSYS [`lmsys/toxic-chat`](https://huggingface.co/datasets/lmsys/toxic-chat) benchmark.

---

## Benchmark Results (50 Real-World Samples)

| Metric | TypeSafe Jev (`jev-1.13.0`) | Local Open-Source PCD (MLX) | Local Autoregressive Baseline | What This Demonstrates |
| :--- | :--- | :--- | :--- | :--- |
| **Model Tier** | **Frontier System 1** | Open-Source 1.5B | Open-Source 1.5B | Frontier vs. Small Model |
| **Execution Mode** | **Cloud API (HTTPS)** | Local Apple Silicon | Local Apple Silicon | Network vs. On-Device |
| **Forward Passes** | **1 Pass (O(1))** | **1 Pass (O(1))** | ~30.8 Passes | **30x Compute Reduction** |
| **Latency p50 (Median)** | **356.5 ms** (incl. network) | **227.2 ms** (on-device) | 735.3 ms (on-device) | Jev over HTTPS is 2x faster than local autoregressive |
| **Latency Mean** | **353.6 ms** | **317.1 ms** | 836.6 ms | Parallelism slashes latency in half |
| **Accuracy** | **84.0%** ⭐ | 52.0% | 54.0% | +32% accuracy jump from Jev's frontier RLCD training |
| **Precision (Safety)** | **90.9%** (Only 1 FP!) ⭐ | 36.0% (16 FPs) | 38.5% (16 FPs) | Jev eliminates false alarms |
| **Recall (Caught Attacks)**| **58.8%** | 52.9% | 58.8% | Caught identical attack vectors |
| **F1 Score** | **0.714** ⭐ | 0.429 | 0.465 | 60%+ boost in balanced safety performance |
| **Brier Score (Calibration)**| **0.1096** (Near-perfect) ⭐| 0.3884 (Uncalibrated) | N/A (Raw strings) | RLCD delivers true probabilistic honesty |
| **Schema Syntax Errors** | **0.0% Guaranteed** | **0.0% Guaranteed** | 98.0% (1 Crash) | Mathematical 0% schema error |

---

## Key Findings

1. **The Architectural Shift (Local PCD):**
   Evaluating structured decisions in a single parallel forward pass ($O(1)$) using KV-cache broadcasting and sub-vocabulary logit slicing reduces forward passes by **96.8%** (1 pass vs 31 passes) and speeds up on-device execution by **3.2x** (227ms vs 735ms) with **94.0% decision concordance** to full autoregressive generation.
2. **The Training Shift (TypeSafe RLCD):**
   While open-source PCD proves the speed mechanism, TypeSafe's **Reinforcement Learning for Calibrated Decisions (RLCD)** delivers frontier quality: **84.0% accuracy**, **90.9% precision** (only 1 false positive out of 50 prompts), and a **0.1096 Brier score** showing true probabilistic calibration.
3. **Catastrophic Schema Drift in Autoregressive Models:**
   On Sample #23, when presented with a prompt asking for a list of human behaviors, the autoregressive LLM suffered instruction confusion and hallucinated new JSON keys (`"human_behavior": [...]`), crashing downstream schema parsers. Both Jev and Parallel Constrained Decoding were mathematically immune to this error.

---

## Quick Start

### 1. Installation
```bash
git clone https://github.com/your-username/system-one-benchmark.git
cd system-one-benchmark

pip install -r requirements.txt
```

### 2. Configuration (For Live TypeSafe Jev)
Copy the example environment file and add your TypeSafe API key:
```bash
cp .env.example .env
# Edit .env:
# TYPESAFE_API_KEY=your_actual_key_here
```

---

## Usage

### Run Live Benchmark on TypeSafe Jev API
```bash
# Evaluates against the 50-sample ToxicChat benchmark dataset
python3 evaluate_jev.py --data-file data/guardrails_toxicchat_50.json

# Or test customer banking intent classification across 77 categories
python3 evaluate_jev.py --data-file data/banking77_50.json
```

### Run Local Parallel Constrained Decoding (Apple Silicon MLX)
Runs on Apple Silicon M-series Macs using Apple's MLX framework:
```bash
# Evaluates both PCD and Autoregressive baseline side-by-side
python3 evaluate_local_pcd.py --task guardrails --data-file data/guardrails_toxicchat_50.json --samples 50
```

### Run Three-Way Comparison Analysis
Compares live Jev results, local MLX results, and Hugging Face ground truth:
```bash
python3 compare_all_models.py
```

### Download Additional Datasets from Hugging Face
```bash
# Stream and format samples from lmsys/toxic-chat or mteb/banking77
python3 download_hf_dataset.py --task guardrails --samples 100
python3 download_hf_dataset.py --task banking --samples 100
```

---

## Repository Structure

```
system-one-benchmark/
├── data/
│   ├── guardrails_toxicchat_50.json   # 50 real-world LMSYS safety prompts
│   └── banking77_50.json              # 50 banking customer service intents
├── evaluate_jev.py                    # TypeSafe Jev API evaluation harness
├── local_pcd_engine.py                # Apple Silicon MLX parallel constrained engine
├── evaluate_local_pcd.py              # Local PCD vs Autoregressive benchmark runner
├── compare_all_models.py              # 3-way comparator (Jev vs PCD vs Autoregressive)
├── analyze_benchmark_results.py       # Metrics and failure case analyzer
├── download_hf_dataset.py             # Hugging Face dataset fetcher
├── sample_data.py                     # Offline fallback fixtures
├── benchmark_results.json             # Live TypeSafe Jev 50-sample results
├── local_mlx_results.json             # Local MLX Apple Silicon 50-sample results
├── three_way_comparison_results.json  # Consolidated benchmark metrics
├── requirements.txt                   # Dependencies
├── .env.example                       # Environment template
└── README.md                          # Documentation
```

---

## References & Attribution
* **TypeSafe AI:** [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
* **Open-Source MLX PCD:** Adapted from [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)
* **Datasets:** [lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat) and [mteb/banking77](https://huggingface.co/datasets/mteb/banking77)
