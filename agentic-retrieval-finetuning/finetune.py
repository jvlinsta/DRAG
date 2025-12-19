#!/usr/bin/env python3
"""
Finetune Qwen3-VL for Better Sparse Search

Based on: https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune/qwenvl/train

Supports:
- SFT (Supervised Fine-Tuning) on successful trajectories
- DPO (Direct Preference Optimization) on query pairs
- LoRA for efficient finetuning
- Label masking to focus loss on search query tokens only
- Component-wise training (LLM, vision tower, MLP projector)
- Separate learning rates for different components
- Image/video processing configuration
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

import torch
from datasets import Dataset
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)


class LabelMaskStrategy(Enum):
    """Strategy for masking labels during SFT training."""
    NONE = "none"                    # No masking, compute loss on all tokens
    SEARCH_ONLY = "search-only"      # Only compute loss on <search>...</search> tokens
    SEARCH_AND_THINK = "search-think"  # Loss on <search> and <think> tokens
    ASSISTANT_ONLY = "assistant"     # Loss on assistant response only (mask user input)


class LossWeightScheme(Enum):
    """Scheme for weighting loss based on sample quality."""
    NONE = "none"                    # No weighting, all samples equal
    RANK_SCORE = "rank-score"        # Weight by rank_score (0-1), higher = more weight
    RANK_SCORE_SQUARED = "rank-sq"   # Weight by rank_score^2 (amplify high scores)
    INVERSE_EFFORT = "inv-effort"    # Weight by cumulative_effort_score if available
    BINARY = "binary"                # 1.0 for rank 1, 0.5 for others


@dataclass
class FinetuneConfig:
    """Configuration for finetuning Qwen3-VL.
    
    Based on QwenLM/Qwen3-VL training framework.
    """
    # Model
    model_name: str = "Qwen/Qwen3-VL-8B-Thinking"
    
    # Component-wise training control (Qwen3-VL specific)
    tune_mm_llm: bool = True       # Fine-tune the language model backbone
    tune_mm_vision: bool = False   # Fine-tune the vision tower (usually False for efficiency)
    tune_mm_mlp: bool = True       # Fine-tune the multimodal projector
    
    # LoRA - target modules for Qwen3-VL architecture
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        # LLM modules
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        # Vision modules (if tune_mm_vision=True)
        # "qkv", "proj", "fc1", "fc2"  # Uncomment if training vision
    ])
    
    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    
    # Training
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 2048
    
    # Component-specific learning rates (Qwen3-VL best practice)
    # Vision tower LR should be 5-10x smaller than LLM LR
    mm_projector_lr: Optional[float] = None  # Defaults to learning_rate if None
    vision_tower_lr: Optional[float] = None  # Should be ~learning_rate/10
    
    # Image/Video processing (Qwen3-VL specific)
    max_pixels: int = 1280 * 28 * 28  # Maximum pixels for images
    min_pixels: int = 256 * 28 * 28   # Minimum pixels for images
    video_max_frames: int = 32        # Max frames for video
    video_min_frames: int = 4         # Min frames for video
    video_fps: float = 2.0            # Target FPS for video sampling
    
    # Data processing
    data_flatten: bool = True         # Flatten nested data structures
    data_packing: bool = False        # Use packed data for efficiency
    
    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # DPO specific
    dpo_beta: float = 0.1


def find_tag_spans(text: str, tag: str) -> List[tuple]:
    """Find all spans of <tag>...</tag> in text.
    
    Returns list of (start_char, end_char) tuples for the content inside tags.
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    spans = []
    for match in re.finditer(pattern, text, re.DOTALL):
        # Include the tags themselves in the span
        spans.append((match.start(), match.end()))
    return spans


def create_masked_labels(
    input_ids: List[int],
    text: str,
    tokenizer,
    mask_strategy: LabelMaskStrategy,
    assistant_start_marker: str = "<|im_start|>assistant\n"
) -> List[int]:
    """Create labels with masking based on strategy.
    
    Args:
        input_ids: Tokenized input IDs
        text: Original text (for finding tag positions)
        tokenizer: Tokenizer for encoding substrings
        mask_strategy: Which tokens to keep for loss computation
        assistant_start_marker: Marker indicating start of assistant response
    
    Returns:
        List of labels with -100 for masked positions
    """
    IGNORE_INDEX = -100
    labels = [IGNORE_INDEX] * len(input_ids)
    
    if mask_strategy == LabelMaskStrategy.NONE:
        return input_ids.copy()
    
    if mask_strategy == LabelMaskStrategy.ASSISTANT_ONLY:
        # Find where assistant response starts and unmask from there
        assistant_start = text.find(assistant_start_marker)
        if assistant_start == -1:
            return input_ids.copy()
        
        # Get character position after the marker
        content_start = assistant_start + len(assistant_start_marker)
        
        # Convert character position to token position
        prefix = text[:content_start]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        token_start = len(prefix_tokens)
        
        # Unmask from assistant content to end
        for i in range(token_start, len(labels)):
            labels[i] = input_ids[i]
        
        return labels
    
    # For SEARCH_ONLY and SEARCH_AND_THINK, find specific tag spans
    spans_to_unmask = []
    
    if mask_strategy in [LabelMaskStrategy.SEARCH_ONLY, LabelMaskStrategy.SEARCH_AND_THINK]:
        spans_to_unmask.extend(find_tag_spans(text, "search"))
    
    if mask_strategy == LabelMaskStrategy.SEARCH_AND_THINK:
        spans_to_unmask.extend(find_tag_spans(text, "think"))
    
    if not spans_to_unmask:
        # No tags found, fall back to all tokens
        return input_ids.copy()
    
    # Convert character spans to token positions
    # We need to map each span to token indices
    for char_start, char_end in spans_to_unmask:
        # Encode the text up to the span start to find token position
        prefix = text[:char_start]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        token_start = len(prefix_tokens)
        
        # Encode up to span end
        prefix_with_span = text[:char_end]
        prefix_with_span_tokens = tokenizer.encode(prefix_with_span, add_special_tokens=False)
        token_end = len(prefix_with_span_tokens)
        
        # Unmask these tokens
        for i in range(token_start, min(token_end, len(labels))):
            labels[i] = input_ids[i]
    
    return labels


def compute_sample_weight(
    example: Dict[str, Any],
    scheme: LossWeightScheme,
    min_weight: float = 0.1
) -> float:
    """Compute sample weight based on quality metrics.
    
    Args:
        example: Data example with potential rank_score, best_rank_score, 
                 cumulative_effort_score fields
        scheme: Weighting scheme to use
        min_weight: Minimum weight to avoid completely ignoring samples
    
    Returns:
        Weight in range [min_weight, 1.0]
    """
    if scheme == LossWeightScheme.NONE:
        return 1.0
    
    # Try to get rank score from various possible field names
    rank_score = (
        example.get("rank_score") or 
        example.get("best_rank_score") or 
        0.0
    )
    
    if scheme == LossWeightScheme.RANK_SCORE:
        # Linear: weight = rank_score, clamped to [min_weight, 1.0]
        return max(min_weight, float(rank_score))
    
    elif scheme == LossWeightScheme.RANK_SCORE_SQUARED:
        # Quadratic: amplify high scores, attenuate low scores
        return max(min_weight, float(rank_score) ** 2)
    
    elif scheme == LossWeightScheme.INVERSE_EFFORT:
        # Use cumulative effort score if available
        effort_score = example.get("cumulative_effort_score", rank_score)
        return max(min_weight, float(effort_score))
    
    elif scheme == LossWeightScheme.BINARY:
        # Binary: full weight for rank 1, half weight for others
        gt_rank = example.get("gt_rank")
        if gt_rank == 1:
            return 1.0
        return 0.5
    
    return 1.0


def load_sft_dataset(
    data_path: str, 
    tokenizer, 
    max_length: int = 2048,
    mask_strategy: LabelMaskStrategy = LabelMaskStrategy.NONE,
    weight_scheme: LossWeightScheme = LossWeightScheme.NONE
) -> Dataset:
    """Load and tokenize SFT dataset with optional label masking and sample weighting.
    
    Args:
        data_path: Path to JSONL file with 'input' and 'output' fields
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        mask_strategy: Strategy for masking labels
            - NONE: Compute loss on all output tokens
            - SEARCH_ONLY: Only compute loss on <search>...</search> tokens
            - SEARCH_AND_THINK: Loss on <search> and <think> tokens
            - ASSISTANT_ONLY: Loss on entire assistant response (mask user input)
        weight_scheme: Scheme for weighting samples by quality
            - NONE: All samples weighted equally
            - RANK_SCORE: Weight by rank_score (higher = more weight)
            - RANK_SCORE_SQUARED: Weight by rank_score^2
            - INVERSE_EFFORT: Weight by cumulative_effort_score
            - BINARY: 1.0 for rank 1, 0.5 for others
    """
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Precompute weights if needed
    weights = []
    if weight_scheme != LossWeightScheme.NONE:
        for example in data:
            weights.append(compute_sample_weight(example, weight_scheme))
        
        # Log weight statistics
        avg_weight = sum(weights) / len(weights) if weights else 0
        min_w, max_w = min(weights) if weights else 0, max(weights) if weights else 0
        print(f"Sample weights: avg={avg_weight:.3f}, min={min_w:.3f}, max={max_w:.3f}")
    
    dataset = Dataset.from_list(data)
    
    def tokenize_function(examples):
        texts = []
        for inp, out in zip(examples['input'], examples['output']):
            text = f"<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
            texts.append(text)
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        
        # Apply label masking
        if mask_strategy == LabelMaskStrategy.NONE:
            tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        else:
            masked_labels = []
            for input_ids, text in zip(tokenized["input_ids"], texts):
                labels = create_masked_labels(
                    input_ids, text, tokenizer, mask_strategy
                )
                masked_labels.append(labels)
            tokenized["labels"] = masked_labels
        
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing SFT data (mask={mask_strategy.value}, weight={weight_scheme.value})"
    )
    
    # Add weights to dataset if using weighting
    if weight_scheme != LossWeightScheme.NONE and weights:
        tokenized_dataset = tokenized_dataset.add_column("sample_weight", weights)
    
    # Log masking statistics
    if mask_strategy != LabelMaskStrategy.NONE:
        total_tokens = 0
        masked_tokens = 0
        for example in tokenized_dataset:
            labels = example["labels"]
            total_tokens += len(labels)
            masked_tokens += sum(1 for l in labels if l == -100)
        
        if total_tokens > 0:
            pct_masked = 100 * masked_tokens / total_tokens
            print(f"Label masking: {pct_masked:.1f}% of tokens masked "
                  f"({masked_tokens}/{total_tokens})")
    
    return tokenized_dataset


def load_dpo_dataset(data_path: str, tokenizer, max_length: int = 2048) -> Dataset:
    """Load DPO dataset."""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    dataset = Dataset.from_list(data)
    
    def format_dpo(examples):
        formatted = {
            "prompt": [],
            "chosen": [],
            "rejected": []
        }
        
        for prompt, chosen, rejected in zip(
            examples['prompt'], 
            examples['chosen'], 
            examples['rejected']
        ):
            formatted["prompt"].append(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
            formatted["chosen"].append(f"{chosen}<|im_end|>")
            formatted["rejected"].append(f"{rejected}<|im_end|>")
        
        return formatted
    
    formatted_dataset = dataset.map(
        format_dpo,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Formatting DPO data"
    )
    
    return formatted_dataset


class WeightedSFTTrainer(Trainer):
    """Custom trainer that supports sample-level loss weighting.
    
    Expects dataset to have 'sample_weight' column with per-sample weights.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute weighted cross-entropy loss."""
        # Extract sample weights if present
        sample_weights = inputs.pop("sample_weight", None)
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        
        if labels is None:
            # Fall back to default loss
            loss = outputs.loss
        else:
            # Compute per-token cross-entropy loss
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            
            per_token_loss = loss_fct(flat_logits, flat_labels)
            
            # Reshape to (batch_size, seq_len)
            batch_size = shift_labels.size(0)
            seq_len = shift_labels.size(1)
            per_token_loss = per_token_loss.view(batch_size, seq_len)
            
            # Compute per-sample loss (mean of non-ignored tokens)
            mask = (shift_labels != -100).float()
            per_sample_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            
            # Apply sample weights if provided
            if sample_weights is not None:
                sample_weights = sample_weights.to(per_sample_loss.device).float()
                weighted_loss = per_sample_loss * sample_weights
                loss = weighted_loss.mean()
            else:
                loss = per_sample_loss.mean()
        
        return (loss, outputs) if return_outputs else loss


def configure_component_training(model, config: FinetuneConfig):
    """Configure which components to train based on Qwen3-VL architecture.
    
    Args:
        model: The Qwen3-VL model
        config: Training configuration with tune_mm_* flags
    """
    # Freeze/unfreeze vision tower
    if hasattr(model, 'visual'):
        for param in model.visual.parameters():
            param.requires_grad = config.tune_mm_vision
        print(f"Vision tower: {'trainable' if config.tune_mm_vision else 'frozen'}")
    
    # Freeze/unfreeze multimodal projector (merger in Qwen3-VL)
    if hasattr(model, 'merger'):
        for param in model.merger.parameters():
            param.requires_grad = config.tune_mm_mlp
        print(f"MM projector (merger): {'trainable' if config.tune_mm_mlp else 'frozen'}")
    
    # Language model is controlled by LoRA, but we can freeze base if needed
    if hasattr(model, 'model'):  # LLM backbone
        if not config.tune_mm_llm:
            for name, param in model.model.named_parameters():
                if 'lora' not in name.lower():
                    param.requires_grad = False
            print("LLM backbone: frozen (LoRA only)")
        else:
            print("LLM backbone: trainable via LoRA")


def get_lora_target_modules(config: FinetuneConfig) -> List[str]:
    """Get LoRA target modules based on what components we're training.
    
    Qwen3-VL architecture:
    - LLM: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    - Vision: qkv, proj, fc1, fc2 (in visual encoder)
    - Merger: Dense layers in the multimodal projector
    """
    target_modules = []
    
    if config.tune_mm_llm:
        # LLM attention and MLP modules
        target_modules.extend([
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ])
    
    if config.tune_mm_vision:
        # Vision encoder modules
        target_modules.extend([
            "qkv", "proj", "fc1", "fc2"
        ])
    
    # If custom modules specified, use those instead
    if config.lora_target_modules:
        return config.lora_target_modules
    
    return target_modules if target_modules else [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]


def is_distributed() -> bool:
    """Check if running in distributed mode (torchrun/accelerate)."""
    return (
        torch.distributed.is_initialized() or 
        "WORLD_SIZE" in os.environ or
        "LOCAL_RANK" in os.environ
    )


def get_local_rank() -> int:
    """Get local rank for distributed training."""
    return int(os.environ.get("LOCAL_RANK", 0))


def create_model_and_processor(config: FinetuneConfig):
    """Create Qwen3-VL model with LoRA and processor.
    
    Uses Qwen3VLForConditionalGeneration and AutoProcessor for proper
    vision-language model support.
    
    Supports both single-GPU (device_map="auto") and multi-GPU DDP training.
    """
    distributed = is_distributed()
    local_rank = get_local_rank()
    
    if config.use_4bit:
        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )
    else:
        bnb_config = None
    
    print(f"Loading Qwen3-VL model: {config.model_name}")
    print(f"  tune_mm_llm={config.tune_mm_llm}, tune_mm_vision={config.tune_mm_vision}, tune_mm_mlp={config.tune_mm_mlp}")
    print(f"  distributed={distributed}, local_rank={local_rank}")
    
    # For distributed training (torchrun), use device_map to current device
    # For single-GPU or model parallelism, use device_map="auto"
    if distributed:
        if config.use_4bit:
            # 4-bit models need explicit device mapping to current GPU
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.model_name,
                quantization_config=bnb_config,
                device_map={"": f"cuda:{local_rank}"},
                trust_remote_code=True,
            )
        else:
            # Non-quantized: load to CPU then move to correct GPU
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            model = model.to(f"cuda:{local_rank}")
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not config.use_4bit else None
        )
    
    # Load processor (includes tokenizer + image/video processor)
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Configure processor for image/video handling
    if hasattr(processor, 'image_processor'):
        processor.image_processor.size = {
            "longest_edge": config.max_pixels,
            "shortest_edge": config.min_pixels
        }
        print(f"Image processor configured: max_pixels={config.max_pixels}, min_pixels={config.min_pixels}")
    
    # Get tokenizer from processor
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure component-wise training
    configure_component_training(model, config)
    
    # Get target modules based on training config
    target_modules = get_lora_target_modules(config)
    print(f"LoRA target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, processor, tokenizer


# Backwards compatibility alias
def create_model_and_tokenizer(config: FinetuneConfig):
    """Backwards compatible wrapper - returns model and tokenizer only."""
    model, processor, tokenizer = create_model_and_processor(config)
    return model, tokenizer


def get_optimizer_grouped_parameters(model, config: FinetuneConfig) -> List[Dict[str, Any]]:
    """Create optimizer parameter groups with different learning rates.
    
    Qwen3-VL best practice: vision tower LR should be 5-10x smaller than LLM LR.
    """
    # Default LRs
    llm_lr = config.learning_rate
    projector_lr = config.mm_projector_lr or config.learning_rate
    vision_lr = config.vision_tower_lr or (config.learning_rate / 10)
    
    # Separate parameters by component
    llm_params = []
    vision_params = []
    projector_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'visual' in name or 'vision' in name.lower():
            vision_params.append(param)
        elif 'merger' in name or 'projector' in name.lower():
            projector_params.append(param)
        else:
            llm_params.append(param)
    
    param_groups = []
    
    if llm_params:
        param_groups.append({
            "params": llm_params,
            "lr": llm_lr,
            "name": "llm"
        })
        print(f"LLM parameters: {len(llm_params)}, lr={llm_lr}")
    
    if projector_params:
        param_groups.append({
            "params": projector_params,
            "lr": projector_lr,
            "name": "projector"
        })
        print(f"Projector parameters: {len(projector_params)}, lr={projector_lr}")
    
    if vision_params:
        param_groups.append({
            "params": vision_params,
            "lr": vision_lr,
            "name": "vision"
        })
        print(f"Vision parameters: {len(vision_params)}, lr={vision_lr}")
    
    return param_groups


def train_sft(
    config: FinetuneConfig,
    train_data_path: str,
    val_data_path: Optional[str] = None,
    mask_strategy: LabelMaskStrategy = LabelMaskStrategy.NONE,
    weight_scheme: LossWeightScheme = LossWeightScheme.NONE
):
    """Run SFT training for Qwen3-VL.
    
    Args:
        config: Training configuration
        train_data_path: Path to training JSONL
        val_data_path: Optional path to validation JSONL
        mask_strategy: Label masking strategy for focusing loss on specific tokens
        weight_scheme: Sample weighting scheme based on quality metrics
    """
    model, processor, tokenizer = create_model_and_processor(config)
    
    print(f"Loading training data from {train_data_path}")
    print(f"Label masking strategy: {mask_strategy.value}")
    print(f"Loss weight scheme: {weight_scheme.value}")
    
    train_dataset = load_sft_dataset(
        train_data_path, tokenizer, config.max_seq_length, 
        mask_strategy, weight_scheme
    )
    
    eval_dataset = None
    if val_data_path:
        print(f"Loading validation data from {val_data_path}")
        eval_dataset = load_sft_dataset(
            val_data_path, tokenizer, config.max_seq_length, 
            mask_strategy, weight_scheme
        )
    
    distributed = is_distributed()
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        save_total_limit=config.save_total_limit,
        bf16=True,
        gradient_checkpointing=True,
        # Fix DDP + gradient checkpointing + LoRA compatibility
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False if distributed else None,
        optim="paged_adamw_32bit",
        report_to="tensorboard",
        remove_unused_columns=False,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    # Use weighted trainer if sample weighting is enabled
    trainer_class = WeightedSFTTrainer if weight_scheme != LossWeightScheme.NONE else Trainer
    
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print("Starting SFT training...")
    trainer.train()
    
    final_path = Path(config.output_dir) / "final"
    print(f"Saving final model to {final_path}")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    # Also save processor for inference
    processor.save_pretrained(str(final_path))
    
    return trainer


def train_dpo(
    config: FinetuneConfig,
    train_data_path: str,
    val_data_path: Optional[str] = None
):
    """Run DPO training for Qwen3-VL."""
    try:
        from trl import DPOTrainer, DPOConfig
    except ImportError:
        raise ImportError("Please install trl: pip install trl")
    
    model, processor, tokenizer = create_model_and_processor(config)
    
    print("Loading reference model (Qwen3-VL)...")
    distributed = is_distributed()
    local_rank = get_local_rank()
    
    if config.use_4bit:
        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )
    else:
        bnb_config = None
    
    if distributed:
        if config.use_4bit:
            ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.model_name,
                quantization_config=bnb_config,
                device_map={"": f"cuda:{local_rank}"},
                trust_remote_code=True,
            )
        else:
            ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            ref_model = ref_model.to(f"cuda:{local_rank}")
    else:
        ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not config.use_4bit else None
        )
    
    print(f"Loading training data from {train_data_path}")
    train_dataset = load_dpo_dataset(train_data_path, tokenizer, config.max_seq_length)
    
    eval_dataset = None
    if val_data_path:
        print(f"Loading validation data from {val_data_path}")
        eval_dataset = load_dpo_dataset(val_data_path, tokenizer, config.max_seq_length)
    
    dpo_config = DPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        save_total_limit=config.save_total_limit,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        report_to="tensorboard",
        beta=config.dpo_beta,
        max_length=config.max_seq_length,
        max_prompt_length=config.max_seq_length // 2,
    )
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    print("Starting DPO training...")
    trainer.train()
    
    final_path = Path(config.output_dir) / "final"
    print(f"Saving final model to {final_path}")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))
    
    return trainer


def merge_and_save(
    adapter_path: str,
    output_path: str,
    model_name: str = "Qwen/Qwen3-VL-8B-Thinking"
):
    """Merge LoRA adapter with base Qwen3-VL model and save."""
    print(f"Loading base Qwen3-VL model: {model_name}")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging adapter with base model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)
    
    # Save processor (includes tokenizer + image processor)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    processor.save_pretrained(output_path)
    
    print("Done!")


def add_common_qwen_vl_args(parser):
    """Add Qwen3-VL specific arguments to a parser."""
    # Component-wise training
    parser.add_argument(
        "--tune-llm", 
        action="store_true", 
        default=True,
        help="Fine-tune the language model backbone (default: True)"
    )
    parser.add_argument(
        "--no-tune-llm", 
        action="store_true",
        help="Disable fine-tuning the language model backbone"
    )
    parser.add_argument(
        "--tune-vision", 
        action="store_true", 
        default=False,
        help="Fine-tune the vision tower (default: False, usually not needed)"
    )
    parser.add_argument(
        "--tune-mlp", 
        action="store_true", 
        default=True,
        help="Fine-tune the multimodal projector (default: True)"
    )
    parser.add_argument(
        "--no-tune-mlp", 
        action="store_true",
        help="Disable fine-tuning the multimodal projector"
    )
    
    # Component-specific learning rates
    parser.add_argument(
        "--vision-lr", 
        type=float, 
        default=None,
        help="Learning rate for vision tower (default: lr/10)"
    )
    parser.add_argument(
        "--projector-lr", 
        type=float, 
        default=None,
        help="Learning rate for multimodal projector (default: same as lr)"
    )
    
    # Image/video processing
    parser.add_argument(
        "--max-pixels", 
        type=int, 
        default=1280 * 28 * 28,
        help="Maximum pixels for image processing"
    )
    parser.add_argument(
        "--min-pixels", 
        type=int, 
        default=256 * 28 * 28,
        help="Minimum pixels for image processing"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Finetune Qwen3-VL for sparse search (based on QwenLM/Qwen3-VL)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # SFT command
    sft_parser = subparsers.add_parser("sft", help="Run SFT training")
    sft_parser.add_argument("--train-data", required=True, help="Training data JSONL")
    sft_parser.add_argument("--val-data", help="Validation data JSONL")
    sft_parser.add_argument("--output-dir", default="./checkpoints/sft", help="Output directory")
    sft_parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking", help="Base model")
    sft_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    sft_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    sft_parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    sft_parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    sft_parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    sft_parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    sft_parser.add_argument(
        "--mask-strategy",
        type=str,
        choices=["none", "search-only", "search-think", "assistant"],
        default="none",
        help="""Label masking strategy to focus loss on specific tokens:
  none:         Compute loss on all tokens (default)
  search-only:  Only loss on <search>...</search> tokens
  search-think: Loss on <search> and <think> tokens
  assistant:    Loss on entire assistant response (mask user input)"""
    )
    sft_parser.add_argument(
        "--weight-scheme",
        type=str,
        choices=["none", "rank-score", "rank-sq", "inv-effort", "binary"],
        default="none",
        help="""Sample weighting scheme based on quality:
  none:       All samples weighted equally (default)
  rank-score: Weight by rank_score (0-1, higher = more weight)
  rank-sq:    Weight by rank_score^2 (amplify high scores)
  inv-effort: Weight by cumulative_effort_score
  binary:     1.0 for rank 1, 0.5 for others"""
    )
    add_common_qwen_vl_args(sft_parser)
    
    # DPO command
    dpo_parser = subparsers.add_parser("dpo", help="Run DPO training")
    dpo_parser.add_argument("--train-data", required=True, help="Training data JSONL")
    dpo_parser.add_argument("--val-data", help="Validation data JSONL")
    dpo_parser.add_argument("--output-dir", default="./checkpoints/dpo", help="Output directory")
    dpo_parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking", help="Base model")
    dpo_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    dpo_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    dpo_parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    dpo_parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    dpo_parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    dpo_parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    dpo_parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    add_common_qwen_vl_args(dpo_parser)
    
    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge LoRA adapter with base model")
    merge_parser.add_argument("--adapter-path", required=True, help="Path to LoRA adapter")
    merge_parser.add_argument("--output-path", required=True, help="Output path for merged model")
    merge_parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking", help="Base model")
    
    args = parser.parse_args()
    
    if args.command == "sft":
        # Resolve boolean flag conflicts
        tune_llm = not args.no_tune_llm if hasattr(args, 'no_tune_llm') else True
        tune_mlp = not args.no_tune_mlp if hasattr(args, 'no_tune_mlp') else True
        tune_vision = args.tune_vision if hasattr(args, 'tune_vision') else False
        
        config = FinetuneConfig(
            model_name=args.model,
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            max_seq_length=args.max_length,
            use_4bit=not args.no_4bit,
            # Qwen3-VL specific
            tune_mm_llm=tune_llm,
            tune_mm_vision=tune_vision,
            tune_mm_mlp=tune_mlp,
            vision_tower_lr=getattr(args, 'vision_lr', None),
            mm_projector_lr=getattr(args, 'projector_lr', None),
            max_pixels=getattr(args, 'max_pixels', 1280 * 28 * 28),
            min_pixels=getattr(args, 'min_pixels', 256 * 28 * 28),
        )
        
        # Parse mask strategy
        mask_strategy_map = {
            "none": LabelMaskStrategy.NONE,
            "search-only": LabelMaskStrategy.SEARCH_ONLY,
            "search-think": LabelMaskStrategy.SEARCH_AND_THINK,
            "assistant": LabelMaskStrategy.ASSISTANT_ONLY,
        }
        mask_strategy = mask_strategy_map[args.mask_strategy]
        
        # Parse weight scheme
        weight_scheme_map = {
            "none": LossWeightScheme.NONE,
            "rank-score": LossWeightScheme.RANK_SCORE,
            "rank-sq": LossWeightScheme.RANK_SCORE_SQUARED,
            "inv-effort": LossWeightScheme.INVERSE_EFFORT,
            "binary": LossWeightScheme.BINARY,
        }
        weight_scheme = weight_scheme_map[args.weight_scheme]
        
        train_sft(config, args.train_data, args.val_data, mask_strategy, weight_scheme)
    
    elif args.command == "dpo":
        # Resolve boolean flag conflicts
        tune_llm = not args.no_tune_llm if hasattr(args, 'no_tune_llm') else True
        tune_mlp = not args.no_tune_mlp if hasattr(args, 'no_tune_mlp') else True
        tune_vision = args.tune_vision if hasattr(args, 'tune_vision') else False
        
        config = FinetuneConfig(
            model_name=args.model,
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            max_seq_length=args.max_length,
            use_4bit=not args.no_4bit,
            dpo_beta=args.beta,
            # Qwen3-VL specific
            tune_mm_llm=tune_llm,
            tune_mm_vision=tune_vision,
            tune_mm_mlp=tune_mlp,
            vision_tower_lr=getattr(args, 'vision_lr', None),
            mm_projector_lr=getattr(args, 'projector_lr', None),
            max_pixels=getattr(args, 'max_pixels', 1280 * 28 * 28),
            min_pixels=getattr(args, 'min_pixels', 256 * 28 * 28),
        )
        train_dpo(config, args.train_data, args.val_data)
    
    elif args.command == "merge":
        merge_and_save(args.adapter_path, args.output_path, args.model)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
