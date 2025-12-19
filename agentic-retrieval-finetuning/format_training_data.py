#!/usr/bin/env python3
"""
Format Training Data for Finetuning

Converts processed trajectories into various training data formats:
- SFT: Supervised fine-tuning on successful queries/trajectories
- DPO: Direct preference optimization with query pairs
- Reward: Reward model training with scores
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from tqdm import tqdm


@dataclass
class SFTExample:
    """Supervised fine-tuning example."""
    input_text: str
    output_text: str
    metadata: Dict[str, Any]


@dataclass
class DPOExample:
    """Direct preference optimization example."""
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    metadata: Dict[str, Any]


@dataclass
class RewardExample:
    """Reward model training example."""
    input_text: str
    score: float
    metadata: Dict[str, Any]


class TrainingDataFormatter:
    """Formats processed trajectories into training data."""
    
    def __init__(
        self,
        include_reasoning: bool = True,
        include_context: bool = False,
        max_context_searches: int = 2
    ):
        self.include_reasoning = include_reasoning
        self.include_context = include_context
        self.max_context_searches = max_context_searches
    
    def format_sft_query_generation(
        self, 
        trajectory: Dict[str, Any]
    ) -> List[SFTExample]:
        """Format for query generation: question -> best query."""
        examples: List[SFTExample] = []
        best_search = trajectory.get("best_search")
        
        if not best_search or not best_search.get("query"):
            return examples
        
        question = trajectory["question"]
        query = best_search["query"]
        
        input_text = f"<question>{question}</question>"
        
        if self.include_reasoning and best_search.get("reasoning"):
            output_text = f"<think>{best_search['reasoning']}</think>\n<search>{query}</search>"
        else:
            output_text = f"<search>{query}</search>"
        
        examples.append(SFTExample(
            input_text=input_text,
            output_text=output_text,
            metadata={
                "trajectory_id": trajectory["id"],
                "rank_score": best_search.get("rank_score"),
                "format": "query_generation"
            }
        ))
        
        return examples
    
    def format_sft_full_trajectory(
        self, 
        trajectory: Dict[str, Any]
    ) -> List[SFTExample]:
        """Format for full trajectory: question -> complete reasoning trace."""
        examples: List[SFTExample] = []
        
        if not trajectory.get("searches"):
            return examples
        
        question = trajectory["question"]
        output_parts = []
        
        best_step = None
        if trajectory.get("best_search"):
            best_step = trajectory["best_search"].get("step")
        
        for search in trajectory["searches"]:
            if self.include_reasoning and search.get("reasoning"):
                output_parts.append(f"<think>{search['reasoning']}</think>")
            
            output_parts.append(f"<search>{search['query']}</search>")
            
            if search.get("gt_found"):
                output_parts.append(f"<result>Found relevant document at rank {search.get('gt_rank', '?')}</result>")
            else:
                output_parts.append("<result>No relevant results</result>")
            
            if best_step and search.get("step") == best_step:
                break
        
        if trajectory.get("final_answer"):
            answers = trajectory["final_answer"]
            if isinstance(answers, list):
                answers = ", ".join(answers)
            output_parts.append(f"<answer>{answers}</answer>")
        
        input_text = f"Question: {question}"
        output_text = "\n".join(output_parts)
        
        examples.append(SFTExample(
            input_text=input_text,
            output_text=output_text,
            metadata={
                "trajectory_id": trajectory["id"],
                "best_rank_score": trajectory.get("best_rank_score"),
                "total_searches": trajectory.get("total_searches"),
                "format": "full_trajectory"
            }
        ))
        
        return examples
    
    def format_sft_query_with_context(
        self, 
        trajectory: Dict[str, Any]
    ) -> List[SFTExample]:
        """Format with context: question + previous attempts -> better query."""
        examples: List[SFTExample] = []
        searches = trajectory.get("searches", [])
        
        if len(searches) < 2:
            return examples
        
        question = trajectory["question"]
        
        for i, search in enumerate(searches[1:], start=1):
            if not search.get("gt_found") or search.get("rank_score", 0) <= 0:
                continue
            
            context_searches = searches[max(0, i - self.max_context_searches):i]
            context_parts = []
            
            for prev_search in context_searches:
                context_parts.append(f"Previous query: {prev_search['query']}")
                if prev_search.get("gt_found"):
                    context_parts.append(f"  Result: Found at rank {prev_search.get('gt_rank')}")
                else:
                    context_parts.append("  Result: Not found")
            
            input_text = f"<question>{question}</question>\n\n" + "\n".join(context_parts)
            
            if self.include_reasoning and search.get("reasoning"):
                output_text = f"<think>{search['reasoning']}</think>\n<search>{search['query']}</search>"
            else:
                output_text = f"<search>{search['query']}</search>"
            
            examples.append(SFTExample(
                input_text=input_text,
                output_text=output_text,
                metadata={
                    "trajectory_id": trajectory["id"],
                    "step": search.get("step"),
                    "rank_score": search.get("rank_score"),
                    "context_length": len(context_searches),
                    "format": "query_with_context"
                }
            ))
        
        return examples
    
    def format_dpo_pairs(
        self, 
        trajectory: Dict[str, Any]
    ) -> List[DPOExample]:
        """Format DPO pairs: better query vs worse query."""
        examples: List[DPOExample] = []
        
        question = trajectory["question"]
        query_pairs = trajectory.get("query_pairs", [])
        
        for pair in query_pairs:
            better = pair.get("better", {})
            worse = pair.get("worse", {})
            
            if not better.get("query") or not worse.get("query"):
                continue
            
            prompt = f"<question>{question}</question>\n\nGenerate a search query to find the answer:"
            
            if self.include_reasoning and better.get("reasoning"):
                chosen = f"<think>{better['reasoning']}</think>\n<search>{better['query']}</search>"
            else:
                chosen = f"<search>{better['query']}</search>"
            
            if self.include_reasoning and worse.get("reasoning"):
                rejected = f"<think>{worse['reasoning']}</think>\n<search>{worse['query']}</search>"
            else:
                rejected = f"<search>{worse['query']}</search>"
            
            examples.append(DPOExample(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                chosen_score=better.get("rank_score", 0.0) or 0.0,
                rejected_score=worse.get("rank_score", 0.0) or 0.0,
                metadata={
                    "trajectory_id": trajectory["id"],
                    "better_query": better["query"],
                    "worse_query": worse["query"]
                }
            ))
        
        return examples
    
    def format_reward_examples(
        self, 
        trajectory: Dict[str, Any]
    ) -> List[RewardExample]:
        """Format reward model examples: query -> score."""
        examples: List[RewardExample] = []
        
        question = trajectory["question"]
        
        for search in trajectory.get("searches", []):
            if not search.get("query"):
                continue
            
            query = search["query"]
            score = search.get("rank_score", 0.0) or 0.0
            
            input_text = f"Question: {question}\nQuery: {query}"
            
            examples.append(RewardExample(
                input_text=input_text,
                score=score,
                metadata={
                    "trajectory_id": trajectory["id"],
                    "step": search.get("step"),
                    "gt_found": search.get("gt_found", False),
                    "gt_rank": search.get("gt_rank")
                }
            ))
        
        return examples
    
    def format_all(
        self,
        input_path: str,
        output_dir: str,
        formats: Optional[List[str]] = None,
        train_ratio: float = 0.9,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Format all training data from processed trajectories."""
        if formats is None:
            formats = ["sft_query", "sft_trajectory", "sft_context", "dpo", "reward"]
        
        random.seed(seed)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_examples: Dict[str, List] = {
            "sft_query": [],
            "sft_trajectory": [],
            "sft_context": [],
            "dpo": [],
            "reward": []
        }
        
        with open(input_path, 'r', encoding='utf-8') as f:
            trajectories = [json.loads(line) for line in f if line.strip()]
        
        for traj in tqdm(trajectories, desc="Formatting training data"):
            if "sft_query" in formats:
                all_examples["sft_query"].extend(self.format_sft_query_generation(traj))
            
            if "sft_trajectory" in formats:
                all_examples["sft_trajectory"].extend(self.format_sft_full_trajectory(traj))
            
            if "sft_context" in formats:
                all_examples["sft_context"].extend(self.format_sft_query_with_context(traj))
            
            if "dpo" in formats:
                all_examples["dpo"].extend(self.format_dpo_pairs(traj))
            
            if "reward" in formats:
                all_examples["reward"].extend(self.format_reward_examples(traj))
        
        stats: Dict[str, Any] = {"total_trajectories": len(trajectories)}
        
        for format_name, examples in all_examples.items():
            if format_name not in formats or not examples:
                continue
            
            random.shuffle(examples)
            split_idx = int(len(examples) * train_ratio)
            train_examples = examples[:split_idx]
            val_examples = examples[split_idx:]
            
            if format_name.startswith("sft"):
                train_data = [
                    {"input": ex.input_text, "output": ex.output_text, **ex.metadata}
                    for ex in train_examples
                ]
                val_data = [
                    {"input": ex.input_text, "output": ex.output_text, **ex.metadata}
                    for ex in val_examples
                ]
            elif format_name == "dpo":
                train_data = [
                    {
                        "prompt": ex.prompt,
                        "chosen": ex.chosen,
                        "rejected": ex.rejected,
                        "chosen_score": ex.chosen_score,
                        "rejected_score": ex.rejected_score,
                        **ex.metadata
                    }
                    for ex in train_examples
                ]
                val_data = [
                    {
                        "prompt": ex.prompt,
                        "chosen": ex.chosen,
                        "rejected": ex.rejected,
                        "chosen_score": ex.chosen_score,
                        "rejected_score": ex.rejected_score,
                        **ex.metadata
                    }
                    for ex in val_examples
                ]
            elif format_name == "reward":
                train_data = [
                    {"input": ex.input_text, "score": ex.score, **ex.metadata}
                    for ex in train_examples
                ]
                val_data = [
                    {"input": ex.input_text, "score": ex.score, **ex.metadata}
                    for ex in val_examples
                ]
            else:
                continue
            
            train_file = output_path / f"{format_name}_train.jsonl"
            val_file = output_path / f"{format_name}_val.jsonl"
            
            with open(train_file, 'w', encoding='utf-8') as f:
                for item in train_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            with open(val_file, 'w', encoding='utf-8') as f:
                for item in val_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            stats[format_name] = {
                "total": len(examples),
                "train": len(train_examples),
                "val": len(val_examples),
                "train_path": str(train_file),
                "val_path": str(val_file)
            }
            
            print(f"\n{format_name}: {len(examples)} examples "
                  f"(train: {len(train_examples)}, val: {len(val_examples)})")
        
        stats_file = output_path / "stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Format training data for finetuning")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--formats", nargs="+",
                       choices=["sft_query", "sft_trajectory", "sft_context", "dpo", "reward", "all"],
                       default=["all"], help="Formats to generate")
    parser.add_argument("--include-reasoning", action="store_true", default=True)
    parser.add_argument("--no-reasoning", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    formats = args.formats
    if "all" in formats:
        formats = ["sft_query", "sft_trajectory", "sft_context", "dpo", "reward"]
    
    formatter = TrainingDataFormatter(
        include_reasoning=not args.no_reasoning
    )
    
    stats = formatter.format_all(
        input_path=args.input,
        output_dir=args.output_dir,
        formats=formats,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    print(f"\n{'='*60}")
    print("Training data formatting complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"Total trajectories: {stats['total_trajectories']}")


if __name__ == "__main__":
    main()
