#!/usr/bin/env python3
"""
Process Trajectories for Training Data

Loads collected trajectories, filters valid training samples (where ground truth
was found in top-k), and computes metrics for training data generation.

Supports multiple output formats:
- dpo: Query pairs for Direct Preference Optimization
- sft-best: Best query per trajectory for Supervised Fine-Tuning
- sft-trajectory: Full trajectory with cumulative effort score
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

from tqdm import tqdm


def extract_thinking_from_raw(raw_response: Optional[str]) -> Optional[str]:
    """Extract thinking/reasoning from raw model response.
    
    Handles multiple formats:
    - <think>...</think> (standard format)
    - Everything before </think> (when opening tag is missing)
    """
    if not raw_response:
        return None
    
    # Try standard <think>...</think> format
    think_match = re.search(r'<think>(.*?)</think>', raw_response, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    
    # Handle missing opening tag (common in Qwen-Thinking)
    if '</think>' in raw_response:
        think_content = raw_response.split('</think>')[0]
        think_content = think_content.strip()
        if think_content:
            return think_content
    
    return None


class OutputFormat(Enum):
    DPO = "dpo"
    SFT_BEST = "sft-best"
    SFT_TRAJECTORY = "sft-trajectory"
    ALL = "all"  # Generate all formats into separate files


@dataclass
class ProcessedIteration:
    """Processed iteration with training-relevant information."""
    step: int
    query: str
    reasoning: Optional[str]
    results: List[Dict[str, Any]]
    gt_found: bool
    gt_rank: Optional[int]
    rank_score: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "query": self.query,
            "reasoning": self.reasoning,
            "results": self.results,
            "gt_found": self.gt_found,
            "gt_rank": self.gt_rank,
            "rank_score": self.rank_score
        }


@dataclass
class ProcessedTrajectory:
    """Processed trajectory ready for training data formatting."""
    id: str
    question: str
    ground_truth: Dict[str, Any]
    
    # All search iterations
    searches: List[ProcessedIteration] = field(default_factory=list)
    
    # Best search (highest rank score)
    best_search: Optional[ProcessedIteration] = None
    best_rank_score: float = 0.0
    
    # First successful search (GT found)
    first_success_search: Optional[ProcessedIteration] = None
    first_success_step: Optional[int] = None
    
    # Trajectory-level metrics
    total_searches: int = 0
    gt_ever_found: bool = False
    final_answer: Optional[List[str]] = None
    answer_correct: Optional[bool] = None
    
    # Cumulative effort score for SFT-trajectory
    # Normalized score: 1.0 = found at rank 1 in first iteration, 0.0 = never found
    cumulative_effort_score: float = 0.0
    
    # For DPO: pairs of better vs worse queries
    query_pairs: List[Tuple[ProcessedIteration, ProcessedIteration]] = field(default_factory=list)
    
    # Metadata
    model: str = ""
    sampling_config: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "searches": [s.to_dict() for s in self.searches],
            "best_search": self.best_search.to_dict() if self.best_search else None,
            "best_rank_score": self.best_rank_score,
            "first_success_search": self.first_success_search.to_dict() if self.first_success_search else None,
            "first_success_step": self.first_success_step,
            "total_searches": self.total_searches,
            "gt_ever_found": self.gt_ever_found,
            "cumulative_effort_score": self.cumulative_effort_score,
            "final_answer": self.final_answer,
            "answer_correct": self.answer_correct,
            "query_pairs": [
                {"better": b.to_dict(), "worse": w.to_dict()}
                for b, w in self.query_pairs
            ],
            "model": self.model,
            "sampling_config": self.sampling_config
        }


class TrajectoryProcessor:
    """Processes raw trajectories into training-ready format."""
    
    def __init__(self, top_k: int = 3, min_rank_score: float = 0.0):
        """
        Args:
            top_k: Number of results per search (for rank score computation)
            min_rank_score: Minimum rank score to consider a search "successful"
        """
        self.top_k = top_k
        self.min_rank_score = min_rank_score
    
    def compute_cumulative_effort_score(
        self, 
        searches: List[ProcessedIteration],
        first_success_idx: Optional[int]
    ) -> float:
        """Compute normalized cumulative effort score.
        
        Score represents how efficiently GT was found across all search attempts.
        - 1.0 = GT found at rank 1 in first iteration (best possible)
        - 0.0 = GT never found (worst possible)
        
        Formula: 1 - (cumulative_position - 1) / (max_possible_positions)
        
        Example: 2 iterations, top_k=3, GT at rank 2 in iteration 2
        - Cumulative position = 3 (iter 1) + 2 (iter 2) = 5
        - Max positions = 2 * 3 = 6
        - Score = 1 - (5-1)/6 = 1 - 4/6 = 0.333
        """
        if first_success_idx is None or not searches:
            return 0.0
        
        # Count positions checked before finding GT
        cumulative_position = 0
        for i, search in enumerate(searches):
            if i < first_success_idx:
                # Full iteration before success - count all top_k positions
                cumulative_position += self.top_k
            elif i == first_success_idx:
                # Success iteration - count up to GT rank
                if search.gt_rank is not None:
                    cumulative_position += search.gt_rank
                else:
                    # Fallback: assume found at last position
                    cumulative_position += self.top_k
                break
        
        # Max possible positions = total_searches * top_k
        max_positions = len(searches) * self.top_k
        
        if max_positions == 0:
            return 0.0
        
        # Normalize: 1.0 = found immediately, 0.0 = found at very last position
        # (cumulative_position - 1) because rank 1 in iter 1 should give score 1.0
        score = 1.0 - (cumulative_position - 1) / max_positions
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def process_trajectory(
        self, 
        raw_trajectory: Dict[str, Any],
        include_failed: bool = False
    ) -> Optional[ProcessedTrajectory]:
        """Process a single trajectory.
        
        Args:
            raw_trajectory: Raw trajectory data from collection
            include_failed: If True, include trajectories where GT was never found
                           (with cumulative_effort_score=0.0)
        
        Returns:
            ProcessedTrajectory or None if filtered out
        """
        processed = ProcessedTrajectory(
            id=raw_trajectory["id"],
            question=raw_trajectory["question"],
            ground_truth=raw_trajectory["ground_truth"],
            final_answer=raw_trajectory.get("final_answer"),
            model=raw_trajectory.get("model", ""),
            sampling_config=raw_trajectory.get("sampling_config")
        )
        
        # Process each iteration
        best_score = -1.0
        first_success_idx = None
        
        for idx, it in enumerate(raw_trajectory.get("iterations", [])):
            if it.get("action") != "search" or not it.get("query"):
                continue
            
            # Extract reasoning - try direct field first, then raw_response
            reasoning = it.get("reasoning")
            if reasoning is None and it.get("raw_response"):
                reasoning = extract_thinking_from_raw(it.get("raw_response"))
            
            proc_it = ProcessedIteration(
                step=it["step"],
                query=it["query"],
                reasoning=reasoning,
                results=it.get("results", []),
                gt_found=it.get("gt_found", False),
                gt_rank=it.get("gt_rank"),
                rank_score=it.get("rank_score")
            )
            
            search_idx = len(processed.searches)
            processed.searches.append(proc_it)
            processed.total_searches += 1
            
            if proc_it.gt_found:
                processed.gt_ever_found = True
                
                if processed.first_success_search is None:
                    processed.first_success_search = proc_it
                    processed.first_success_step = proc_it.step
                    first_success_idx = search_idx
                
                if proc_it.rank_score is not None and proc_it.rank_score > best_score:
                    best_score = proc_it.rank_score
                    processed.best_search = proc_it
                    processed.best_rank_score = proc_it.rank_score
        
        # Compute cumulative effort score
        processed.cumulative_effort_score = self.compute_cumulative_effort_score(
            processed.searches, first_success_idx
        )
        
        # Filter out trajectories where GT was never found (unless include_failed)
        if not processed.gt_ever_found and not include_failed:
            return None
        
        # Generate query pairs for DPO
        processed.query_pairs = self._generate_query_pairs(processed.searches)
        
        return processed
    
    def _generate_query_pairs(
        self, 
        searches: List[ProcessedIteration]
    ) -> List[Tuple[ProcessedIteration, ProcessedIteration]]:
        """Generate pairs of better vs worse queries for DPO training."""
        pairs = []
        
        for i, search_i in enumerate(searches):
            for j, search_j in enumerate(searches):
                if i >= j:
                    continue
                
                score_i = search_i.rank_score if search_i.rank_score is not None else -1
                score_j = search_j.rank_score if search_j.rank_score is not None else -1
                
                if score_i > score_j and score_i > 0:
                    pairs.append((search_i, search_j))
                elif score_j > score_i and score_j > 0:
                    pairs.append((search_j, search_i))
        
        return pairs
    
    def format_for_dpo(self, trajectory: ProcessedTrajectory) -> List[Dict[str, Any]]:
        """Format trajectory for DPO training (query pairs).
        
        Output format is directly compatible with finetune.py DPO training:
        - prompt: The question
        - chosen: The better search query
        - rejected: The worse search query
        """
        examples = []
        for better, worse in trajectory.query_pairs:
            # Format with <think> tags if reasoning is available
            if better.reasoning:
                chosen = f"<think>{better.reasoning}</think>\n<search>{better.query}</search>"
            else:
                chosen = f"<search>{better.query}</search>"
            
            if worse.reasoning:
                rejected = f"<think>{worse.reasoning}</think>\n<search>{worse.query}</search>"
            else:
                rejected = f"<search>{worse.query}</search>"
            
            examples.append({
                "prompt": f"<question>{trajectory.question}</question>\n\nGenerate a search query to find the answer:",
                "chosen": chosen,
                "rejected": rejected,
                "chosen_score": better.rank_score or 0.0,
                "rejected_score": worse.rank_score or 0.0,
                "trajectory_id": trajectory.id,
            })
        return examples
    
    def format_for_sft_best(self, trajectory: ProcessedTrajectory) -> Optional[Dict[str, Any]]:
        """Format trajectory for SFT training (best query only).
        
        Output format is directly compatible with finetune.py SFT training:
        - input: The question
        - output: The best search query (with optional reasoning)
        
        Uses the best successful query (highest rank score) as the target.
        For failed trajectories, returns None.
        """
        if not trajectory.gt_ever_found or trajectory.best_search is None:
            return None
        
        # Format output with <think> tags if reasoning is available
        if trajectory.best_search.reasoning:
            output = f"<think>{trajectory.best_search.reasoning}</think>\n<search>{trajectory.best_search.query}</search>"
        else:
            output = f"<search>{trajectory.best_search.query}</search>"
        
        return {
            "input": f"<question>{trajectory.question}</question>",
            "output": output,
            "trajectory_id": trajectory.id,
            "rank_score": trajectory.best_rank_score,
            "gt_rank": trajectory.best_search.gt_rank,
        }
    
    def format_for_sft_trajectory(
        self, 
        trajectory: ProcessedTrajectory,
        include_failed: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Format full trajectory for SFT training with complete search flow.
        
        Output format is directly compatible with finetune.py SFT training:
        - input: The question
        - output: Full trajectory with reasoning, searches, results, and answer
        
        Includes all searches and the cumulative effort score as the target.
        Can include failed trajectories (score=0) for contrastive learning.
        """
        if not include_failed and not trajectory.gt_ever_found:
            return None
        
        # Build output with full trajectory
        output_parts = []
        for search in trajectory.searches:
            if search.reasoning:
                output_parts.append(f"<think>{search.reasoning}</think>")
            output_parts.append(f"<search>{search.query}</search>")
            
            if search.gt_found:
                output_parts.append(f"<result>Found relevant document at rank {search.gt_rank or '?'}</result>")
            else:
                output_parts.append("<result>No relevant results</result>")
        
        # Add final answer if available
        if trajectory.final_answer:
            answers = trajectory.final_answer
            if isinstance(answers, list):
                answers = ", ".join(str(a) for a in answers)
            output_parts.append(f"<answer>{answers}</answer>")
        
        return {
            "input": f"Question: {trajectory.question}",
            "output": "\n".join(output_parts),
            "trajectory_id": trajectory.id,
            "total_searches": trajectory.total_searches,
            "gt_ever_found": trajectory.gt_ever_found,
            "cumulative_effort_score": trajectory.cumulative_effort_score,
            "best_rank_score": trajectory.best_rank_score,
        }
    
    def process_file(
        self, 
        input_path: str, 
        output_path: str,
        output_format: OutputFormat = OutputFormat.DPO,
        include_failed: bool = False,
        stats_path: Optional[str] = None,
        train_ratio: float = 0.9,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Process all trajectories from input file.
        
        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file (or directory if train_ratio < 1.0)
            output_format: Format for output (dpo, sft-best, sft-trajectory)
            include_failed: Include trajectories where GT was never found
            stats_path: Optional path to save statistics JSON
            train_ratio: Ratio of data for training (rest goes to validation)
                        If 1.0, outputs single file. If < 1.0, outputs train/val files.
            seed: Random seed for train/val split
        """
        random.seed(seed)
        stats: Dict[str, Any] = {
            "output_format": output_format.value,
            "include_failed": include_failed,
            "total_trajectories": 0,
            "valid_trajectories": 0,
            "failed_trajectories": 0,
            "output_examples": 0,
            "total_searches": 0,
            "total_searches_all": 0,  # Including failed trajectories
            "successful_searches": 0,
            "total_query_pairs": 0,
            "rank_score_distribution": defaultdict(int),
            "iterations_to_first_success": defaultdict(int),
            "best_rank_distribution": defaultdict(int),
            "effort_score_distribution": defaultdict(int),
        }
        
        all_trajectories: List[ProcessedTrajectory] = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # First pass: process all trajectories
        for line in tqdm(lines, desc="Processing trajectories"):
            if not line.strip():
                continue
            
            raw_traj = json.loads(line)
            stats["total_trajectories"] += 1
            
            # Process with include_failed=True to get all trajectories
            processed = self.process_trajectory(raw_traj, include_failed=True)
            
            if processed is None:
                continue
            
            stats["total_searches_all"] += processed.total_searches
            
            if processed.gt_ever_found:
                stats["valid_trajectories"] += 1
                stats["total_searches"] += processed.total_searches
                stats["successful_searches"] += sum(1 for s in processed.searches if s.gt_found)
                stats["total_query_pairs"] += len(processed.query_pairs)
                
                if processed.best_rank_score > 0:
                    bucket = f"{processed.best_rank_score:.2f}"
                    stats["rank_score_distribution"][bucket] += 1
                
                if processed.first_success_step is not None:
                    stats["iterations_to_first_success"][processed.first_success_step] += 1
                
                if processed.best_search and processed.best_search.gt_rank:
                    stats["best_rank_distribution"][processed.best_search.gt_rank] += 1
                
                # Effort score distribution (bucket by 0.1)
                effort_bucket = f"{processed.cumulative_effort_score:.1f}"
                stats["effort_score_distribution"][effort_bucket] += 1
            else:
                stats["failed_trajectories"] += 1
                stats["effort_score_distribution"]["0.0"] += 1
            
            all_trajectories.append(processed)
        
        # Second pass: format examples based on output format
        def write_examples_with_split(
            examples: List[Dict[str, Any]], 
            base_path: Path, 
            name: str
        ) -> Dict[str, Any]:
            """Write examples with optional train/val split."""
            random.shuffle(examples)
            result = {"total": len(examples)}
            
            if train_ratio < 1.0:
                split_idx = int(len(examples) * train_ratio)
                train_examples = examples[:split_idx]
                val_examples = examples[split_idx:]
                
                train_path = base_path / f"{name}_train.jsonl"
                val_path = base_path / f"{name}_val.jsonl"
                
                with open(train_path, 'w', encoding='utf-8') as f:
                    for ex in train_examples:
                        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
                
                with open(val_path, 'w', encoding='utf-8') as f:
                    for ex in val_examples:
                        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
                
                result["train"] = len(train_examples)
                result["val"] = len(val_examples)
                result["train_path"] = str(train_path)
                result["val_path"] = str(val_path)
            else:
                out_path = base_path / f"{name}.jsonl"
                with open(out_path, 'w', encoding='utf-8') as f:
                    for ex in examples:
                        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
                result["path"] = str(out_path)
            
            return result
        
        # Handle ALL format - generate all formats into directory
        if output_format == OutputFormat.ALL:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect examples for each format
            dpo_examples: List[Dict[str, Any]] = []
            sft_best_examples: List[Dict[str, Any]] = []
            sft_traj_examples: List[Dict[str, Any]] = []
            
            for traj in all_trajectories:
                if not include_failed and not traj.gt_ever_found:
                    continue
                
                # DPO
                dpo_examples.extend(self.format_for_dpo(traj))
                
                # SFT Best
                sft_best = self.format_for_sft_best(traj)
                if sft_best:
                    sft_best_examples.append(sft_best)
                
                # SFT Trajectory
                sft_traj = self.format_for_sft_trajectory(traj, include_failed=include_failed)
                if sft_traj:
                    sft_traj_examples.append(sft_traj)
            
            # Write each format
            stats["formats"] = {}
            
            if dpo_examples:
                stats["formats"]["dpo"] = write_examples_with_split(dpo_examples, output_dir, "dpo")
                print(f"DPO: {len(dpo_examples)} examples")
            
            if sft_best_examples:
                stats["formats"]["sft_best"] = write_examples_with_split(sft_best_examples, output_dir, "sft_best")
                print(f"SFT-Best: {len(sft_best_examples)} examples")
            
            if sft_traj_examples:
                stats["formats"]["sft_trajectory"] = write_examples_with_split(sft_traj_examples, output_dir, "sft_trajectory")
                print(f"SFT-Trajectory: {len(sft_traj_examples)} examples")
            
            stats["output_examples"] = len(dpo_examples) + len(sft_best_examples) + len(sft_traj_examples)
            stats["output_dir"] = str(output_dir)
        
        else:
            # Single format output
            all_examples: List[Dict[str, Any]] = []
            for traj in all_trajectories:
                # Filter based on format requirements
                if not include_failed and not traj.gt_ever_found:
                    continue
                
                if output_format == OutputFormat.DPO:
                    examples = self.format_for_dpo(traj)
                    all_examples.extend(examples)
                
                elif output_format == OutputFormat.SFT_BEST:
                    example = self.format_for_sft_best(traj)
                    if example:
                        all_examples.append(example)
                
                elif output_format == OutputFormat.SFT_TRAJECTORY:
                    example = self.format_for_sft_trajectory(traj, include_failed=include_failed)
                    if example:
                        all_examples.append(example)
            
            # Shuffle and split into train/val
            random.shuffle(all_examples)
            
            if train_ratio < 1.0:
                split_idx = int(len(all_examples) * train_ratio)
                train_examples = all_examples[:split_idx]
                val_examples = all_examples[split_idx:]
                
                # Determine output paths
                output_path_obj = Path(output_path)
                if output_path_obj.suffix == '.jsonl':
                    # output_path is a file, create train/val variants
                    train_path = output_path_obj.with_stem(f"{output_path_obj.stem}_train")
                    val_path = output_path_obj.with_stem(f"{output_path_obj.stem}_val")
                else:
                    # output_path is a directory or base name
                    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                    train_path = output_path_obj.parent / f"{output_path_obj.stem}_train.jsonl"
                    val_path = output_path_obj.parent / f"{output_path_obj.stem}_val.jsonl"
                
                # Write train file
                with open(train_path, 'w', encoding='utf-8') as f:
                    for ex in train_examples:
                        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
                
                # Write val file
                with open(val_path, 'w', encoding='utf-8') as f:
                    for ex in val_examples:
                        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
                
                stats["train_examples"] = len(train_examples)
                stats["val_examples"] = len(val_examples)
                stats["train_path"] = str(train_path)
                stats["val_path"] = str(val_path)
                print(f"Train/val split: {len(train_examples)} train, {len(val_examples)} val")
            else:
                # Single output file
                with open(output_path, 'w', encoding='utf-8') as f:
                    for ex in all_examples:
                        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
                stats["train_path"] = output_path
            
            stats["output_examples"] = len(all_examples)
        
        # Convert defaultdicts to regular dicts
        stats["rank_score_distribution"] = dict(stats["rank_score_distribution"])
        stats["iterations_to_first_success"] = dict(stats["iterations_to_first_success"])
        stats["best_rank_distribution"] = dict(stats["best_rank_distribution"])
        stats["effort_score_distribution"] = dict(stats["effort_score_distribution"])
        
        if stats_path:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
        
        return stats


def print_stats(stats: Dict[str, Any]):
    """Pretty print statistics."""
    print("\n" + "=" * 60)
    print("TRAJECTORY PROCESSING STATISTICS")
    print("=" * 60)
    
    output_format = stats.get('output_format', 'dpo')
    include_failed = stats.get('include_failed', False)
    
    print(f"\nOutput Format: {output_format}")
    print(f"Include Failed: {include_failed}")
    
    total = stats['total_trajectories']
    valid = stats['valid_trajectories']
    failed = stats.get('failed_trajectories', stats.get('filtered_trajectories', 0))
    
    print(f"\nTrajectories:")
    print(f"  Total: {total}")
    print(f"  Valid (GT found): {valid} ({100*valid/max(1,total):.1f}%)")
    print(f"  Failed (GT not found): {failed} ({100*failed/max(1,total):.1f}%)")
    
    print(f"\nSearches:")
    total_searches_all = stats.get('total_searches_all', stats['total_searches'])
    print(f"  Total (all trajectories): {total_searches_all}")
    print(f"  Total (valid only): {stats['total_searches']}")
    print(f"  Successful: {stats['successful_searches']} "
          f"({100*stats['successful_searches']/max(1,stats['total_searches']):.1f}%)")
    
    print(f"\nTraining Data:")
    print(f"  Output examples: {stats.get('output_examples', 'N/A')}")
    if stats.get('train_examples') is not None:
        print(f"  Train examples: {stats['train_examples']}")
        print(f"  Val examples: {stats['val_examples']}")
    if output_format == 'all' and stats.get('formats'):
        print(f"  Formats generated:")
        for fmt_name, fmt_stats in stats['formats'].items():
            print(f"    {fmt_name}: {fmt_stats.get('total', 0)} examples")
    elif output_format == 'dpo':
        print(f"  Query pairs for DPO: {stats['total_query_pairs']}")
    
    print(f"\nIterations to First Success:")
    for step, count in sorted(stats['iterations_to_first_success'].items()):
        print(f"  Step {step}: {count}")
    
    print(f"\nBest Rank Distribution:")
    for rank, count in sorted(stats['best_rank_distribution'].items()):
        print(f"  Rank {rank}: {count}")
    
    if 'effort_score_distribution' in stats and stats['effort_score_distribution']:
        print(f"\nCumulative Effort Score Distribution:")
        for score, count in sorted(stats['effort_score_distribution'].items(), 
                                    key=lambda x: float(x[0])):
            print(f"  {score}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Process trajectories for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output Formats:
  dpo             Query pairs for Direct Preference Optimization.
                  Only includes trajectories with multiple searches of varying quality.
                  
  sft-best        Best query per trajectory for Supervised Fine-Tuning.
                  Uses the highest-scoring successful query as target.
                  Includes ALL valid trajectories (not just those with pairs).
                  
  sft-trajectory  Full trajectory with cumulative effort score.
                  Score = 1.0 for GT found at rank 1 in first search,
                  Score = 0.0 for GT never found.
                  Use --include-failed to add negative examples.
                  
  all             Generate ALL formats into a directory.
                  Creates: dpo.jsonl, sft_best.jsonl, sft_trajectory.jsonl
                  Use with --train-ratio for train/val splits.

Examples:
  # DPO training data (original behavior)
  python process_trajectories.py -i trajectories.jsonl -o dpo_train.jsonl --format dpo
  
  # SFT with best queries (all valid trajectories) with train/val split
  python process_trajectories.py -i trajectories.jsonl -o sft.jsonl --format sft-best --train-ratio 0.9
  
  # SFT with effort scores including failed trajectories
  python process_trajectories.py -i trajectories.jsonl -o sft_train.jsonl --format sft-trajectory --include-failed
  
  # Generate ALL formats into a directory with train/val splits
  python process_trajectories.py -i trajectories.jsonl -o training_data/ --format all --train-ratio 0.9
"""
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file (or directory if --format all)")
    parser.add_argument("--stats", "-s", help="Output JSON file for statistics")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k used during collection")
    parser.add_argument("--min-rank-score", type=float, default=0.0)
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["dpo", "sft-best", "sft-trajectory", "all"],
        default="sft-best",
        help="Output format: dpo, sft-best, sft-trajectory, or all (generates all into directory)"
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include trajectories where GT was never found (score=0.0). Useful for contrastive learning."
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=1.0,
        help="Train/val split ratio (default: 1.0 = no split). E.g., 0.9 creates 90%% train, 10%% val."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and train/val split (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Parse output format
    format_map = {
        "dpo": OutputFormat.DPO,
        "sft-best": OutputFormat.SFT_BEST,
        "sft-trajectory": OutputFormat.SFT_TRAJECTORY,
        "all": OutputFormat.ALL,
    }
    output_format = format_map[args.format]
    
    processor = TrajectoryProcessor(
        top_k=args.top_k,
        min_rank_score=args.min_rank_score
    )
    
    stats = processor.process_file(
        input_path=args.input,
        output_path=args.output,
        output_format=output_format,
        include_failed=args.include_failed,
        stats_path=args.stats,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    print_stats(stats)
    
    # Print output paths
    if output_format == OutputFormat.ALL:
        print(f"\nAll formats saved to: {stats.get('output_dir', args.output)}")
        for fmt_name, fmt_stats in stats.get('formats', {}).items():
            if args.train_ratio < 1.0:
                print(f"  {fmt_name}: {fmt_stats.get('train', 0)} train, {fmt_stats.get('val', 0)} val")
            else:
                print(f"  {fmt_name}: {fmt_stats.get('total', 0)} examples → {fmt_stats.get('path', 'N/A')}")
    elif args.train_ratio < 1.0:
        print(f"\nTraining data saved to:")
        print(f"  Train: {stats.get('train_path', 'N/A')} ({stats.get('train_examples', 0)} examples)")
        print(f"  Val:   {stats.get('val_path', 'N/A')} ({stats.get('val_examples', 0)} examples)")
    else:
        print(f"\nProcessed trajectories saved to: {args.output}")


if __name__ == "__main__":
    main()
