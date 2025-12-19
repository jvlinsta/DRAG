# DRAG: Document Retrieval with Agentic Grounding

Train Vision-Language Models to generate better search queries through trajectory learning.

**📖 [Read the full tutorial →](tutorial.md)**

## Overview

This pipeline improves sparse search by finetuning a vision-language model (Qwen-VL) on successful search trajectories. The key insight is that we can use the **normalized rank of ground truth documents** as a reward signal to train the model to generate more effective search queries.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Training Pipeline                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Collect       2. Process        3. Format         4. Finetune   │
│  Trajectories     Trajectories      Training Data     Model         │
│                                                                     │
│  ┌─────────┐     ┌─────────────┐   ┌────────────┐   ┌────────────┐ │
│  │ Run VLM │ ──► │ Filter GT   │ ─►│ SFT / DPO  │ ─►│ LoRA       │ │
│  │ Agent   │     │ in top-k    │   │ Formats    │   │ Training   │ │
│  └─────────┘     └─────────────┘   └────────────┘   └────────────┘ │
│       │                │                 │                │        │
│       ▼                ▼                 ▼                ▼        │
│  trajectories.   processed.         training_data/   checkpoints/  │
│  jsonl           jsonl              *.jsonl          final/        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/DRAG.git
cd DRAG

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Navigate to the pipeline directory
cd agentic-retrieval-finetuning
```

## Quick Start

> **Note**: All pipeline scripts are in the `agentic-retrieval-finetuning/` directory.

### 1. Start vLLM Server

```bash
# For trajectory collection and evaluation
vllm serve Qwen/Qwen3-VL-8B-Thinking \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

### 2. Collect Trajectories

```bash
cd agentic-retrieval-finetuning

python trajectory_collector.py \
  --output trajectories.jsonl \
  --ocr-file /path/to/ocr_output.jsonl \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --sampling-config default \
  --concurrency 8 \
  --limit 500
```

### 3. Process Trajectories

```bash
python process_trajectories.py \
  --input trajectories.jsonl \
  --output processed.jsonl \
  --stats stats.json
```

### 4. Format Training Data

```bash
python format_training_data.py \
  --input processed.jsonl \
  --output-dir training_data/ \
  --formats all
```

### 5. Finetune

```bash
# SFT on best queries
python finetune.py sft \
  --train-data training_data/sft_best_train.jsonl \
  --val-data training_data/sft_best_val.jsonl \
  --output-dir checkpoints/sft \
  --model Qwen/Qwen3-VL-8B-Thinking

# Or DPO on query pairs
python finetune.py dpo \
  --train-data training_data/dpo_train.jsonl \
  --output-dir checkpoints/dpo

# Merge adapter with base model
python finetune.py merge \
  --adapter-path checkpoints/sft/final \
  --output-path merged_model/
```

### 6. Evaluate

```bash
# Serve model with LoRA adapter
vllm serve Qwen/Qwen3-VL-8B-Thinking \
  --port 8000 \
  --enable-lora \
  --lora-modules my-adapter=checkpoints/sft/final

# Evaluate base model
python evaluate.py \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --ocr-file /path/to/ocr_output.jsonl \
  --output results/eval_base.json

# Evaluate finetuned adapter
python evaluate.py \
  --model my-adapter \
  --ocr-file /path/to/ocr_output.jsonl \
  --output results/eval_adapter.json

# Compare results
python evaluate.py --compare results/eval_base.json results/eval_adapter.json
```

## Key Concepts

### Normalized Rank Score

The reward signal for training is computed as:

```
rank_score = (top_k - rank + 1) / top_k
```

| Rank | Score (k=3) | Interpretation |
|------|-------------|----------------|
| 1    | 1.00        | Perfect hit    |
| 2    | 0.67        | Good           |
| 3    | 0.33        | Found but low  |
| >3   | 0.00        | Not found      |

### Training Data Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `sft_best` | question → best_query | Direct query generation |
| `sft_trajectory` | question → full_trace | Learn reasoning patterns |
| `sft_context` | question + prev_attempts → better_query | Iterative improvement |
| `dpo` | (prompt, chosen_query, rejected_query) | Preference learning |
| `reward` | (query, score) | Reward model training |

### Sampling Configurations

Experiment with different sampling parameters to generate diverse trajectories:

| Config | Temperature | Top-p | Use Case |
|--------|-------------|-------|----------|
| `greedy` | 0.0 | 1.0 | Deterministic baseline |
| `low_temp` | 0.3 | 0.9 | Focused, less random |
| `default` | 0.7 | 0.95 | Balanced |
| `high_temp` | 1.0 | 0.95 | More diverse |
| `creative` | 1.2 | 0.95 | Maximum diversity |

## Evaluation Metrics

- **Success Rate**: % of questions where GT document was found
- **Iterations to Success**: Average searches needed to find GT document
- **First-Hit Rank**: GT document rank on first search
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank across queries
- **Hit@K**: % of queries where GT is in top-K results
- **ANLS\***: Answer accuracy metric
- **Citation F1**: Precision/recall of cited documents

## Results

After finetuning on ~200 successful trajectories:

| Metric | Base | Finetuned | Δ |
|--------|------|-----------|---|
| Hit@1 | 40% | **46%** | +15% |
| First Search MRR | 0.893 | **0.936** | +4.8% |
| Iterations to Success | 1.97 | **1.85** | -6.1% |

## Project Structure

```
DRAG/
├── pyproject.toml              # Project dependencies
├── README.md
│
├── agentic-retrieval-finetuning/
│   ├── trajectory_collector.py # Step 1: Collect agent trajectories
│   ├── process_trajectories.py # Step 2: Filter and score
│   ├── format_training_data.py # Step 3: Create training data
│   ├── finetune.py             # Step 4: LoRA finetuning (SFT/DPO)
│   ├── evaluate.py             # Step 5: Evaluation
│   ├── inference_test.py       # Quick inference testing
│   ├── search_engine.py        # Whoosh sparse search
│   └── utils.py                # PDF/image utilities
│
├── training_data/              # Generated training files (gitignored)
├── results/                    # Evaluation results (gitignored)
└── checkpoints/                # Model checkpoints (gitignored)
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (8×A100 recommended for training)
- vLLM for model serving
- Access to document dataset with OCR (coming with release of Document AI 

## References

- [STaR-SQL: Self-Taught Reasoner for Text-to-SQL](https://arxiv.org/abs/2502.13550)
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

## License

MIT
