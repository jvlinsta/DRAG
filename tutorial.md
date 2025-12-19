# DRAG: Document Retrieval with Agentic Grounding

A tutorial on finetuning Vision-Language Models to generate better search queries through trajectory learning.

---

## Table of Contents

1. [Introduction](#introduction)
2. [The Core Idea](#the-core-idea)
3. [Architecture Overview](#architecture-overview)
4. [Step 1: Collecting Trajectories](#step-1-collecting-trajectories)
5. [Step 2: Processing & Filtering](#step-2-processing--filtering)
6. [Step 3: Formatting Training Data](#step-3-formatting-training-data)
7. [Step 4: LoRA Finetuning](#step-4-lora-finetuning)
8. [Step 5: Evaluation](#step-5-evaluation)
9. [Results & Analysis](#results--analysis)
10. [Appendix: Infrastructure Setup](#appendix-infrastructure-setup)

---

## Introduction

Document retrieval is a critical component of Retrieval-Augmented Generation (RAG) systems. However, turning natural language questions into effective search queries remains challenging—especially for sparse (keyword-based) retrieval systems like BM25 or Whoosh.

This tutorial demonstrates how to **finetune a Vision-Language Model (VLM) to generate better search queries** by learning from its own successful search trajectories. The key insight is that we can use the **normalized rank of ground truth documents** as a reward signal to train the model to produce more effective keywords.

### Why This Matters

- **Sparse search is fast and interpretable** but struggles with vocabulary mismatch
- **VLMs can understand documents visually** and reason about content
- **Iterative refinement** allows models to learn from their search successes and failures

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (8×A100 recommended for training, smaller GPUs work for inference)
- Access to a document QA dataset with ground truth locations
- vLLM for efficient model serving

---

## The Core Idea

The method is inspired by [STaR-SQL: Self-Taught Reasoner for Text-to-SQL](https://arxiv.org/abs/2502.13550), adapted for multimodal document retrieval.

### The Problem

Given a question like:
> *"What is the maintenance interval for the Apache helicopter's rotor assembly?"*

We need to generate search keywords that will retrieve the correct document page from a corpus. A naive approach might search for `"apache helicopter maintenance"`, but the correct page might use different terminology like `"AH-64 rotor service schedule"`.

### The Solution: Trajectory Learning

1. **Run a VLM agent** that iteratively searches and refines queries
2. **Track which queries** successfully retrieve the ground truth document
3. **Train the model** to generate the successful queries directly

The key innovation is the **normalized rank score**:

```
rank_score = (top_k - rank + 1) / top_k
```

| Rank | Score (k=3) | Interpretation |
|------|-------------|----------------|
| 1    | 1.00        | Perfect hit    |
| 2    | 0.67        | Good           |
| 3    | 0.33        | Found but low  |
| >3   | 0.00        | Not found      |

This score serves as the reward signal for training.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DRAG Training Pipeline                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │   1. COLLECT    │    │   2. PROCESS    │    │   3. FORMAT     │     │
│  │   Trajectories  │───►│   & Filter      │───►│   Training Data │     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│         │                      │                       │               │
│         ▼                      ▼                       ▼               │
│    trajectories.jsonl    processed.jsonl        training_data/        │
│    - Full reasoning      - GT in top-k only     - sft_*.jsonl         │
│    - Search queries      - Rank scores          - dpo_*.jsonl         │
│    - Results & ranks     - Best queries         - reward_*.jsonl      │
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐                            │
│  │   4. FINETUNE   │    │   5. EVALUATE   │                            │
│  │   LoRA + SFT    │───►│   Compare Base  │                            │
│  └─────────────────┘    └─────────────────┘                            │
│         │                      │                                        │
│         ▼                      ▼                                        │
│    checkpoints/          results/*.json                                │
│    - LoRA adapters       - Success rate                                │
│    - Merged model        - Iterations to success                       │
│                          - Hit@k, MRR                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Collecting Trajectories

The first step is to run the VLM agent on your document QA dataset and log everything.

### What We Capture

For each question, we record:
- **Reasoning tokens** (`<think>...</think>` blocks)
- **Search queries** generated by the model
- **Search results** with ranks
- **Ground truth matching** (did GT appear? at what rank?)

### Running the Collector

```bash
# Start the vLLM server
vllm serve Qwen/Qwen3-VL-8B-Thinking \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.95 \
  --api-key "your-key" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes

# Collect trajectories (parallel mode for throughput)
python trajectory_collector.py \
  --output trajectories.jsonl \
  --ocr-file /path/to/ocr_output.jsonl \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --sampling-config default \
  --concurrency 16 \
  --limit 500
```

### Sampling Configurations

Different sampling parameters produce different trajectory diversity:

| Config | Temperature | Top-p | Best For |
|--------|-------------|-------|----------|
| `greedy` | 0.0 | 1.0 | Deterministic baseline |
| `default` | 0.7 | 0.95 | Balanced exploration |
| `creative` | 1.2 | 0.95 | Maximum diversity |
| `diverse` | 0.9 | 0.98 | Variety with repetition penalty |

> **Observation**: Greedy decoding causes repetition in thinking models. Temperature 0.7 produces much cleaner reasoning traces.

### Trajectory Structure

Each trajectory is a JSON object:

```json
{
  "id": "traj_q123",
  "question": "What is the maximum speed of the CH-47?",
  "ground_truth": {
    "file": "specs/ch47_manual.pdf",
    "page": 42,
    "answers": ["170 knots"]
  },
  "iterations": [
    {
      "step": 1,
      "reasoning": "The user is asking about CH-47 speed specifications...",
      "action": "search",
      "query": "CH-47 maximum speed knots",
      "results": [...],
      "gt_found": true,
      "gt_rank": 1,
      "rank_score": 1.0
    }
  ],
  "best_rank_score": 1.0,
  "gt_ever_found": true
}
```

---

## Step 2: Processing & Filtering

Not all trajectories are useful for training. We filter to keep only those where the model successfully retrieved the ground truth document.

### Why Filter?

If the ground truth document was never in the search results, we have:
- ❌ No positive signal about what worked
- ❌ No way to compute meaningful rank scores
- ❌ Potentially noisy/misleading training data

### Running the Processor

```bash
python process_trajectories.py \
  --input trajectories.jsonl \
  --output processed.jsonl \
  --stats stats.json
```

### Processing Statistics

From our experiments with ~350 collected trajectories:

```
TRAJECTORY PROCESSING STATISTICS

Trajectories:
  Total: 351
  Valid (GT found): 209 (59.5%)
  Filtered (GT not found): 142 (40.5%)

Iterations to First Success:
  Step 1: 189 (90.4%)
  Step 2: 15 (7.2%)
  Step 3: 5 (2.4%)

Best Rank Distribution:
  Rank 1: 195 (93.3%)
  Rank 2: 14 (6.7%)
```

> **Key Insight**: The model is effective on the first try (90%+ find GT at step 1). This means most training signal comes from single-step trajectories.

---

## Step 3: Formatting Training Data

We convert processed trajectories into multiple training formats to support different finetuning approaches.

### Supported Formats

| Format | Input | Output | Use Case |
|--------|-------|--------|----------|
| `sft_query` | Question | Best query | Direct query generation |
| `sft_trajectory` | Question | Full reasoning trace | Learn reasoning patterns |
| `sft_context` | Question + prev attempts | Better query | Iterative improvement |
| `dpo` | Prompt | (chosen, rejected) pair | Preference learning |
| `reward` | Query | Score | Reward model training |

### Running the Formatter

```bash
python format_training_data.py \
  --input processed.jsonl \
  --output-dir training_data/ \
  --formats all \
  --val-split 0.1
```

### Training Data Structure

**SFT Query Format** (simplest, most direct):
```json
{
  "messages": [
    {"role": "user", "content": "Search query for: What is the maintenance interval?"},
    {"role": "assistant", "content": "<search>AH-64 rotor maintenance schedule</search>"}
  ],
  "weight": 1.0
}
```

**DPO Format** (contrastive learning):
```json
{
  "prompt": "Search query for: What is the maintenance interval?",
  "chosen": "<search>AH-64 rotor maintenance schedule</search>",
  "rejected": "<search>helicopter maintenance</search>"
}
```

### DPO Data Scarcity

A challenge we observed: **DPO requires paired comparisons**, but most trajectories succeed on the first query. This creates few contrast pairs:

- 351 total trajectories
- Only ~50 have multiple search iterations
- Only ~20 produce valid DPO pairs (same question, different quality queries)

> **Recommendation**: Start with SFT on successful queries. Use DPO only if you have sufficient comparison data.

---

## Step 4: LoRA Finetuning

We use LoRA (Low-Rank Adaptation) to efficiently finetune the VLM while preserving its base capabilities.

### Why LoRA?

- **Memory efficient**: ~10% of full finetuning memory
- **Fast training**: Fewer parameters to update
- **Composable**: Can serve multiple adapters with vLLM
- **Safe**: Easy to compare with base model

### Training Configuration

```bash
# Basic SFT with query format
python finetune.py sft \
  --train-data training_data/sft_best_train.jsonl \
  --val-data training_data/sft_best_val.jsonl \
  --output-dir ./checkpoints/sft \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --epochs 3 \
  --batch-size 4 \
  --lr 2e-4 \
  --lora-r 16
```

### Advanced: Label Masking & Sample Weighting

Focus the loss on what matters:

```bash
python finetune.py sft \
  --train-data training_data/sft_best_train.jsonl \
  --output-dir ./checkpoints/sft_masked \
  --mask-strategy search-only \
  --weight-scheme rank-score
```

**Mask Strategies:**
- `none`: Standard loss on all tokens
- `search-only`: Only compute loss on `<search>...</search>` tokens
- `assistant`: Loss on all assistant tokens

**Weight Schemes:**
- `none`: Equal weight for all samples
- `rank-score`: Weight by normalized rank (better queries weighted more)
- `binary`: 1.0 for rank-1 hits, 0.5 otherwise

### Multi-GPU Training

For 8×A100 setup:

```bash
torchrun --nproc_per_node=8 finetune.py sft \
  --train-data training_data/sft_best_train.jsonl \
  --output-dir ./checkpoints/sft \
  --model Qwen/Qwen3-VL-8B-Thinking
```

### Merging the Adapter

After training, you can merge the LoRA weights into the base model:

```bash
python finetune.py merge \
  --adapter-path ./checkpoints/sft/final \
  --output-path ./merged_model
```

Or serve with vLLM using dynamic adapter loading (recommended):

```bash
vllm serve Qwen/Qwen3-VL-8B-Thinking \
  --enable-lora \
  --lora-modules my-adapter=checkpoints/sft/final \
  --trust-remote-code
```

---

## Step 5: Evaluation

Compare the finetuned model against the base model on a held-out test set.

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Success Rate** | % of questions answered correctly |
| **Iterations to Success** | Average searches needed to find GT |
| **First-Hit Rank** | GT document rank on first search |
| **MRR** | Mean Reciprocal Rank (1/rank averaged) |
| **Hit@k** | % of queries with GT in top-k |

### Running Evaluation

```bash
# Evaluate base model
python evaluate.py \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --include-json splits/test.json \
  --ocr-file data/ocr_output.jsonl \
  --output results/base_results.json

# Evaluate finetuned model (adapter mode)
python evaluate.py \
  --model my-adapter \
  --include-json splits/test.json \
  --ocr-file data/ocr_output.jsonl \
  --output results/adapter_results.json
```

---

## Results & Analysis

### Comparison: Base vs. Finetuned

| Metric | Base Model | Finetuned | Change |
|--------|------------|-----------|--------|
| Success Rate | 68.0% | 66.0% | -2.0% |
| Avg Iterations | 4.14 | 3.98 | **-3.9%** |
| Iterations to Success | 1.97 | **1.85** | **-6.1%** |
| First-Hit Rate | 50.0% | **52.0%** | **+4.0%** |
| First-Hit Rank | 1.24 | **1.15** | **-7.3%** |
| First Search MRR | 0.893 | **0.936** | **+4.8%** |
| Hit@1 | 40.0% | **46.0%** | **+15.0%** |
| Hit@3 | 50.0% | **52.0%** | **+4.0%** |

### Key Observations

1. **Improved First Search Quality**: The finetuned model generates better initial queries (MRR +4.8%, Hit@1 +15%)

2. **Fewer Iterations Needed**: When successful, the finetuned model needs fewer search iterations (1.85 vs 1.97)

3. **Trade-off on Success Rate**: Slight decrease in overall success rate (-2%), likely due to the model being more "decisive" and less exploratory

4. **Best Rank Improvement**: First-hit rank improved from 1.24 to 1.15, meaning the GT document appears higher in results

### Interpretation

The finetuning achieved its primary goal: **better first-try search queries**. The model learned to generate more effective keywords that retrieve the ground truth document higher in the results.

The slight success rate decrease suggests room for improvement:
- More training data (we only had ~200 valid trajectories)
- Longer training (3 epochs may be insufficient)
- Different training formats (trajectory-based vs query-only)

---

## Appendix: Infrastructure Setup

### AWS EC2 Setup (8×A100 80GB)

```bash
# Mount ephemeral NVMe storage
sudo mkfs.ext4 -F /dev/nvme1n1
sudo mount /dev/nvme1n1 /data
sudo chown ubuntu:ubuntu /data

# Setup Python environment
source ~/.local/bin/env
mkdir -p /data/cache
export UV_CACHE_DIR=/data/cache
uv venv /data/vllm-env --python 3.11
source /data/vllm-env/bin/activate
uv pip install vllm
```

### SSH Port Forwarding

```bash
# ~/.ssh/config
Host gpu-server
    HostName <your-ip>
    User ubuntu
    IdentityFile ~/.ssh/your-key.pem
    LocalForward 8000 localhost:8000
    LocalForward 8888 localhost:8888
```

### Syncing Code to Server

```bash
rsync -avz --exclude '__pycache__' --exclude '.venv' \
  ./agentic-retrieval-finetuning gpu-server:/data/
```

### vLLM Serve Commands

**Base Model:**
```bash
vllm serve Qwen/Qwen3-VL-8B-Thinking \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.95 \
  --api-key "abc123" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-num-seqs 32
```

**With LoRA Adapter:**
```bash
vllm serve Qwen/Qwen3-VL-8B-Thinking \
  --tensor-parallel-size 8 \
  --enable-lora \
  --lora-modules my-adapter=checkpoints/sft/final \
  --trust-remote-code
```

### Monitoring GPU Utilization

```bash
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv'
```

---

## References

- [STaR-SQL: Self-Taught Reasoner for Text-to-SQL](https://arxiv.org/abs/2502.13550)
- [Qwen3-VL Model Card](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [TRL: Transformer Reinforcement Learning](https://huggingface.co/docs/trl)

---

## License

MIT

---

*Tutorial created as part of the Agentic Document AI benchmark project.*

