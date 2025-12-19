#!/usr/bin/env python3
"""
Trajectory Collector for Sparse Search Finetuning

Collects rich trajectory data from VLM search agent runs for finetuning.
Logs full conversations, reasoning tokens, search queries, and ground truth matching.

Supports parallel request processing to maximize vLLM throughput via continuous batching.
"""

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import openai
from PIL import Image
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm
from datasets import load_dataset

from search_engine import WhooshSearchEngine
from utils import get_pdf_page_as_png, image_to_base64, resize_image_if_needed


@dataclass
class SamplingConfig:
    """Configuration for sampling parameters to experiment with."""
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    max_tokens: int = 4096 #reasonable
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SamplingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class SearchResult:
    """A single search result with rank information."""
    file: str
    page_number: int
    total_pages: int
    rank: int  # 1-indexed rank in results
    
    def matches_ground_truth(self, gt_file: str, gt_page: int) -> bool:
        """Check if this result matches ground truth."""
        return self.file == gt_file and self.page_number == gt_page


@dataclass
class IterationLog:
    """Log of a single iteration in the search trajectory."""
    step: int
    reasoning: Optional[str]  # Model's reasoning/thinking tokens
    action: str  # "search" or "answer"
    query: Optional[str] = None  # Search query if action is "search"
    results: List[SearchResult] = field(default_factory=list)
    gt_found: bool = False
    gt_rank: Optional[int] = None  # Rank of ground truth in results (1-indexed)
    rank_score: Optional[float] = None  # Normalized rank score
    answer: Optional[List[str]] = None  # Answer if action is "answer"
    citations: Optional[List[Dict]] = None
    raw_response: Optional[str] = None  # Raw model response


@dataclass
class Trajectory:
    """Complete trajectory for a single question."""
    id: str
    question_id: str  # Original question ID from dataset
    question: str
    ground_truth: Dict[str, Any]  # file, page, answers
    iterations: List[IterationLog] = field(default_factory=list)
    final_answer: Optional[List[str]] = None
    final_citations: Optional[List[Dict]] = None
    success: bool = False  # Whether answer was found
    gt_ever_found: bool = False  # Whether GT was ever in search results
    first_gt_iteration: Optional[int] = None  # First iteration where GT appeared
    best_gt_rank: Optional[int] = None  # Best rank achieved for GT
    best_rank_score: Optional[float] = None  # Best normalized rank score
    total_iterations: int = 0
    total_searches: int = 0
    sampling_config: Optional[Dict] = None
    model: str = ""
    timestamp: str = ""
    error: Optional[str] = None
    
    def compute_metrics(self, top_k: int):
        """Compute aggregate metrics for this trajectory."""
        self.total_iterations = len(self.iterations)
        self.total_searches = sum(1 for it in self.iterations if it.action == "search")
        
        for it in self.iterations:
            if it.gt_found and it.gt_rank is not None:
                self.gt_ever_found = True
                if self.first_gt_iteration is None:
                    self.first_gt_iteration = it.step
                if self.best_gt_rank is None or it.gt_rank < self.best_gt_rank:
                    self.best_gt_rank = it.gt_rank
                    self.best_rank_score = it.rank_score


class TrajectoryCollector:
    """Collects rich trajectory data from VLM search agent runs."""
    
    def __init__(
        self,
        search_engine: WhooshSearchEngine,
        model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        sampling_config: Optional[SamplingConfig] = None
    ):
        self.search_engine = search_engine
        self.model = model
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8001/v1")
        api_key = api_key or os.environ.get("VLLM_API_KEY", "abc123")
        self.client = openai.OpenAI(base_url=self.base_url, api_key=api_key)
        self.sampling_config = sampling_config or SamplingConfig()
        print(f"Initialized TrajectoryCollector with model: {self.model}")
        print(f"Sampling config: {self.sampling_config}")
    
    def _load_page_image(self, file: str, page: int) -> Dict:
        """Load a single page as an image in OpenAI vision format."""
        image = get_pdf_page_as_png(file, page)
        _, base64_image = resize_image_if_needed(image)
        
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": "high"
            }
        }
    
    def _extract_thinking(self, text: str) -> Optional[str]:
        """Extract thinking/reasoning tokens from model response.
        
        Handles multiple formats:
        - <think>...</think> (standard format)
        - Everything before </think> (when opening tag is missing, common in Qwen-Thinking)
        - "Reasoning:" or "Let me think:" prefixed content
        """
        if not text:
            return None
        
        # Try to find <think>...</think> blocks (standard format)
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            return think_match.group(1).strip()
        
        # Handle case where model outputs thinking without opening <think> tag
        # but has closing </think> tag (common in Qwen-Thinking models)
        if '</think>' in text:
            think_content = text.split('</think>')[0]
            # Clean up any leading artifacts
            think_content = think_content.strip()
            if think_content:
                return think_content
        
        # Try to find reasoning in other formats
        reason_match = re.search(
            r'(?:Reasoning|Thinking|Let me think):\s*(.*?)(?:\n\n|$)', 
            text, re.DOTALL | re.IGNORECASE
        )
        if reason_match:
            return reason_match.group(1).strip()
        
        return None
    
    def _compute_rank_score(self, rank: int, top_k: int) -> float:
        """Compute normalized rank score: (top_k - rank + 1) / top_k.
        
        rank=1 -> score=1.0 (best)
        rank=top_k -> score=1/top_k (worst but still in results)
        """
        return (top_k - rank + 1) / top_k
    
    def _check_ground_truth_in_results(
        self, 
        results: List[SearchResult], 
        gt_file: str, 
        gt_page: int,
        top_k: int
    ) -> tuple[bool, Optional[int], Optional[float]]:
        """Check if ground truth is in results and return rank info."""
        for result in results:
            if result.matches_ground_truth(gt_file, gt_page):
                rank_score = self._compute_rank_score(result.rank, top_k)
                return True, result.rank, rank_score
        return False, None, None
    
    def collect_trajectory(
        self,
        question: str,
        ground_truth: Dict[str, Any],
        trajectory_id: str,
        question_id: str,
        max_iterations: int = 5,
        top_k: int = 3
    ) -> Trajectory:
        """Collect a single trajectory with rich logging."""
        
        trajectory = Trajectory(
            id=trajectory_id,
            question_id=question_id,
            question=question,
            ground_truth=ground_truth,
            sampling_config=self.sampling_config.to_dict(),
            model=self.model,
            timestamp=datetime.now().isoformat()
        )
        
        # Extract ground truth file and page
        gt_file = None
        gt_page = None
        if ground_truth.get("answer_locations"):
            loc = ground_truth["answer_locations"][0] if isinstance(
                ground_truth["answer_locations"], list
            ) else ground_truth["answer_locations"]
            if isinstance(loc, dict):
                gt_file = loc.get("document") or loc.get("file")
                gt_page = loc.get("page")
            elif isinstance(loc, str):
                parts = loc.split(":")
                if len(parts) == 2:
                    gt_file = parts[0]
                    gt_page = int(parts[1])
        
        # Define tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search document collection and return images of matching pages.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query using keywords, phrases, boolean operators"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "provide_answer",
                    "description": "Provide the final structured answer with citations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of answer values"
                            },
                            "citations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "file": {"type": "string"},
                                        "page": {"type": "integer"}
                                    },
                                    "required": ["file", "page"]
                                },
                                "description": "List of citations"
                            }
                        },
                        "required": ["answer", "citations"]
                    }
                }
            }
        ]
        
        system_prompt = """You are a document QA assistant with access to a search tool.
The search tool returns images of document pages.

IMPORTANT: The answer to the question is definitely in the documents. If your search returns no results or unhelpful pages, try different search terms.

Once you find relevant pages, analyze the images carefully. When you have the answer, use the provide_answer tool with:
- answer: list of answer values (use exact words from document when possible)
- citations: list of sources with file and page

Always use one of the available tools (search_documents or provide_answer)."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        try:
            for iteration in range(1, max_iterations + 1):
                iteration_log = IterationLog(step=iteration, reasoning=None, action="unknown")
                
                # Build API call with sampling parameters
                api_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    "temperature": self.sampling_config.temperature,
                    "top_p": self.sampling_config.top_p,
                    "max_tokens": self.sampling_config.max_tokens,
                }
                
                if self.sampling_config.repetition_penalty != 1.0:
                    api_kwargs["extra_body"] = {
                        "repetition_penalty": self.sampling_config.repetition_penalty
                    }
                
                # On last iteration, force answer
                if iteration == max_iterations:
                    force_message = {
                        "role": "user",
                        "content": "You must now provide your final answer using the provide_answer tool."
                    }
                    api_kwargs["messages"] = messages + [force_message]
                    api_kwargs["tool_choice"] = {
                        "type": "function", 
                        "function": {"name": "provide_answer"}
                    }
                else:
                    api_kwargs["tool_choice"] = "auto"
                
                response = self.client.chat.completions.create(**api_kwargs)
                message = response.choices[0].message
                
                iteration_log.raw_response = message.content
                iteration_log.reasoning = self._extract_thinking(message.content)
                
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        if tool_call.function.name == "provide_answer":
                            iteration_log.action = "answer"
                            try:
                                answer_data = json.loads(tool_call.function.arguments)
                                iteration_log.answer = answer_data.get("answer", [])
                                iteration_log.citations = answer_data.get("citations", [])
                                trajectory.final_answer = iteration_log.answer
                                trajectory.final_citations = iteration_log.citations
                                trajectory.success = True
                            except json.JSONDecodeError:
                                pass
                            
                            trajectory.iterations.append(iteration_log)
                            trajectory.compute_metrics(top_k)
                            return trajectory
                        
                        elif tool_call.function.name == "search_documents":
                            iteration_log.action = "search"
                            try:
                                query = json.loads(tool_call.function.arguments)["query"]
                                iteration_log.query = query
                            except (json.JSONDecodeError, KeyError):
                                continue
                            
                            # Execute search
                            raw_results = self.search_engine.search(query, top_k)
                            
                            # Convert to SearchResult with ranks
                            search_results = []
                            for rank, r in enumerate(raw_results, 1):
                                search_results.append(SearchResult(
                                    file=r["file"],
                                    page_number=r["page_number"],
                                    total_pages=r["total_pages"],
                                    rank=rank
                                ))
                            iteration_log.results = search_results
                            
                            # Check ground truth
                            if gt_file and gt_page:
                                gt_found, gt_rank, rank_score = self._check_ground_truth_in_results(
                                    search_results, gt_file, gt_page, top_k
                                )
                                iteration_log.gt_found = gt_found
                                iteration_log.gt_rank = gt_rank
                                iteration_log.rank_score = rank_score
                            
                            # Add to messages for next iteration
                            messages.append({
                                "role": "assistant",
                                "content": message.content,
                                "tool_calls": [{
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments
                                    }
                                }]
                            })
                            
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Found {len(raw_results)} matching pages."
                            })
                            
                            # Add images
                            if raw_results:
                                image_content: List[Dict[str, Any]] = [
                                    {"type": "text", "text": "Here are the matching pages:\n"}
                                ]
                                for result in raw_results:
                                    image_content.append({
                                        "type": "text",
                                        "text": f"\nFile: {result['file']}, Page: {result['page_number']}"
                                    })
                                    try:
                                        image_content.append(
                                            self._load_page_image(result['file'], result['page_number'])
                                        )
                                    except Exception as e:
                                        print(f"Warning: Could not load image: {e}")
                                
                                messages.append({"role": "user", "content": image_content})
                
                else:
                    iteration_log.action = "text_response"
                
                trajectory.iterations.append(iteration_log)
            
            trajectory.error = "Maximum iterations reached"
            
        except Exception as e:
            trajectory.error = str(e)
            print(f"Error collecting trajectory: {e}")
        
        trajectory.compute_metrics(top_k)
        return trajectory


class AsyncTrajectoryCollector:
    """Async version for parallel trajectory collection to maximize vLLM throughput."""
    
    def __init__(
        self,
        search_engine: WhooshSearchEngine,
        model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        sampling_config: Optional[SamplingConfig] = None
    ):
        self.search_engine = search_engine
        self.model = model
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8001/v1")
        api_key = api_key or os.environ.get("VLLM_API_KEY", "abc123")
        self.client = openai.AsyncOpenAI(base_url=self.base_url, api_key=api_key)
        self.sampling_config = sampling_config or SamplingConfig()
        print(f"Initialized AsyncTrajectoryCollector with model: {self.model}")
        print(f"Sampling config: {self.sampling_config}")
    
    def _load_page_image(self, file: str, page: int) -> Dict:
        """Load a single page as an image in OpenAI vision format."""
        image = get_pdf_page_as_png(file, page)
        _, base64_image = resize_image_if_needed(image)
        
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": "high"
            }
        }
    
    def _extract_thinking(self, text: str) -> Optional[str]:
        """Extract thinking/reasoning tokens from model response.
        
        Handles multiple formats:
        - <think>...</think> (standard format)
        - Everything before </think> (when opening tag is missing, common in Qwen-Thinking)
        - "Reasoning:" or "Let me think:" prefixed content
        """
        if not text:
            return None
        
        # Try to find <think>...</think> blocks (standard format)
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            return think_match.group(1).strip()
        
        # Handle case where model outputs thinking without opening <think> tag
        # but has closing </think> tag (common in Qwen-Thinking models)
        if '</think>' in text:
            think_content = text.split('</think>')[0]
            # Clean up any leading artifacts
            think_content = think_content.strip()
            if think_content:
                return think_content
        
        # Try to find reasoning in other formats
        reason_match = re.search(
            r'(?:Reasoning|Thinking|Let me think):\s*(.*?)(?:\n\n|$)', 
            text, re.DOTALL | re.IGNORECASE
        )
        if reason_match:
            return reason_match.group(1).strip()
        
        return None
    
    def _compute_rank_score(self, rank: int, top_k: int) -> float:
        """Compute normalized rank score."""
        return (top_k - rank + 1) / top_k
    
    def _check_ground_truth_in_results(
        self, 
        results: List[SearchResult], 
        gt_file: str, 
        gt_page: int,
        top_k: int
    ) -> tuple[bool, Optional[int], Optional[float]]:
        """Check if ground truth is in results and return rank info."""
        for result in results:
            if result.matches_ground_truth(gt_file, gt_page):
                rank_score = self._compute_rank_score(result.rank, top_k)
                return True, result.rank, rank_score
        return False, None, None
    
    async def collect_trajectory(
        self,
        question: str,
        ground_truth: Dict[str, Any],
        trajectory_id: str,
        question_id: str,
        max_iterations: int = 5,
        top_k: int = 3
    ) -> Trajectory:
        """Collect a single trajectory with rich logging (async version)."""
        
        trajectory = Trajectory(
            id=trajectory_id,
            question_id=question_id,
            question=question,
            ground_truth=ground_truth,
            sampling_config=self.sampling_config.to_dict(),
            model=self.model,
            timestamp=datetime.now().isoformat()
        )
        
        # Extract ground truth file and page
        gt_file = None
        gt_page = None
        if ground_truth.get("answer_locations"):
            loc = ground_truth["answer_locations"][0] if isinstance(
                ground_truth["answer_locations"], list
            ) else ground_truth["answer_locations"]
            if isinstance(loc, dict):
                gt_file = loc.get("document") or loc.get("file")
                gt_page = loc.get("page")
            elif isinstance(loc, str):
                parts = loc.split(":")
                if len(parts) == 2:
                    gt_file = parts[0]
                    gt_page = int(parts[1])
        
        # Define tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search document collection and return images of matching pages.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query using keywords, phrases, boolean operators"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "provide_answer",
                    "description": "Provide the final structured answer with citations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of answer values"
                            },
                            "citations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "file": {"type": "string"},
                                        "page": {"type": "integer"}
                                    },
                                    "required": ["file", "page"]
                                },
                                "description": "List of citations"
                            }
                        },
                        "required": ["answer", "citations"]
                    }
                }
            }
        ]
        
        system_prompt = """You are a document QA assistant with access to a search tool.
The search tool returns images of document pages.

IMPORTANT: The answer to the question is definitely in the documents. If your search returns no results or unhelpful pages, try different search terms.

Once you find relevant pages, analyze the images carefully. When you have the answer, use the provide_answer tool with:
- answer: list of answer values (use exact words from document when possible)
- citations: list of sources with file and page

Always use one of the available tools (search_documents or provide_answer)."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        try:
            for iteration in range(1, max_iterations + 1):
                iteration_log = IterationLog(step=iteration, reasoning=None, action="unknown")
                
                # Build API call with sampling parameters
                api_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    "temperature": self.sampling_config.temperature,
                    "top_p": self.sampling_config.top_p,
                    "max_tokens": self.sampling_config.max_tokens,
                }
                
                if self.sampling_config.repetition_penalty != 1.0:
                    api_kwargs["extra_body"] = {
                        "repetition_penalty": self.sampling_config.repetition_penalty
                    }
                
                # On last iteration, force answer
                if iteration == max_iterations:
                    force_message = {
                        "role": "user",
                        "content": "You must now provide your final answer using the provide_answer tool."
                    }
                    api_kwargs["messages"] = messages + [force_message]
                    api_kwargs["tool_choice"] = {
                        "type": "function", 
                        "function": {"name": "provide_answer"}
                    }
                else:
                    api_kwargs["tool_choice"] = "auto"
                
                response = await self.client.chat.completions.create(**api_kwargs)
                message = response.choices[0].message
                
                iteration_log.raw_response = message.content
                iteration_log.reasoning = self._extract_thinking(message.content)
                
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        if tool_call.function.name == "provide_answer":
                            iteration_log.action = "answer"
                            try:
                                answer_data = json.loads(tool_call.function.arguments)
                                iteration_log.answer = answer_data.get("answer", [])
                                iteration_log.citations = answer_data.get("citations", [])
                                trajectory.final_answer = iteration_log.answer
                                trajectory.final_citations = iteration_log.citations
                                trajectory.success = True
                            except json.JSONDecodeError:
                                pass
                            
                            trajectory.iterations.append(iteration_log)
                            trajectory.compute_metrics(top_k)
                            return trajectory
                        
                        elif tool_call.function.name == "search_documents":
                            iteration_log.action = "search"
                            try:
                                query = json.loads(tool_call.function.arguments)["query"]
                                iteration_log.query = query
                            except (json.JSONDecodeError, KeyError):
                                continue
                            
                            # Execute search (sync - it's fast)
                            raw_results = self.search_engine.search(query, top_k)
                            
                            # Convert to SearchResult with ranks
                            search_results = []
                            for rank, r in enumerate(raw_results, 1):
                                search_results.append(SearchResult(
                                    file=r["file"],
                                    page_number=r["page_number"],
                                    total_pages=r["total_pages"],
                                    rank=rank
                                ))
                            iteration_log.results = search_results
                            
                            # Check ground truth
                            if gt_file and gt_page:
                                gt_found, gt_rank, rank_score = self._check_ground_truth_in_results(
                                    search_results, gt_file, gt_page, top_k
                                )
                                iteration_log.gt_found = gt_found
                                iteration_log.gt_rank = gt_rank
                                iteration_log.rank_score = rank_score
                            
                            # Add to messages for next iteration
                            messages.append({
                                "role": "assistant",
                                "content": message.content,
                                "tool_calls": [{
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments
                                    }
                                }]
                            })
                            
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Found {len(raw_results)} matching pages."
                            })
                            
                            # Add images
                            if raw_results:
                                image_content: List[Dict[str, Any]] = [
                                    {"type": "text", "text": "Here are the matching pages:\n"}
                                ]
                                for result in raw_results:
                                    image_content.append({
                                        "type": "text",
                                        "text": f"\nFile: {result['file']}, Page: {result['page_number']}"
                                    })
                                    try:
                                        image_content.append(
                                            self._load_page_image(result['file'], result['page_number'])
                                        )
                                    except Exception as e:
                                        print(f"Warning: Could not load image: {e}")
                                
                                messages.append({"role": "user", "content": image_content})
                
                else:
                    iteration_log.action = "text_response"
                
                trajectory.iterations.append(iteration_log)
            
            trajectory.error = "Maximum iterations reached"
            
        except Exception as e:
            trajectory.error = str(e)
            # Don't print here to avoid cluttering async output
        
        trajectory.compute_metrics(top_k)
        return trajectory


def trajectory_to_dict(trajectory: Trajectory) -> Dict[str, Any]:
    """Convert trajectory to serializable dict."""
    result = {
        "id": trajectory.id,
        "question_id": trajectory.question_id,
        "question": trajectory.question,
        "ground_truth": trajectory.ground_truth,
        "iterations": [],
        "final_answer": trajectory.final_answer,
        "final_citations": trajectory.final_citations,
        "success": trajectory.success,
        "gt_ever_found": trajectory.gt_ever_found,
        "first_gt_iteration": trajectory.first_gt_iteration,
        "best_gt_rank": trajectory.best_gt_rank,
        "best_rank_score": trajectory.best_rank_score,
        "total_iterations": trajectory.total_iterations,
        "total_searches": trajectory.total_searches,
        "sampling_config": trajectory.sampling_config,
        "model": trajectory.model,
        "timestamp": trajectory.timestamp,
        "error": trajectory.error
    }
    
    for it in trajectory.iterations:
        it_dict = {
            "step": it.step,
            "reasoning": it.reasoning,
            "action": it.action,
            "query": it.query,
            "results": [
                {
                    "file": r.file,
                    "page_number": r.page_number,
                    "total_pages": r.total_pages,
                    "rank": r.rank
                }
                for r in it.results
            ],
            "gt_found": it.gt_found,
            "gt_rank": it.gt_rank,
            "rank_score": it.rank_score,
            "answer": it.answer,
            "citations": it.citations,
            "raw_response": it.raw_response
        }
        result["iterations"].append(it_dict)
    
    return result


# Predefined sampling configurations for experiments
SAMPLING_CONFIGS = {
    "default": SamplingConfig(temperature=0.7, top_p=0.95),
    "greedy": SamplingConfig(temperature=0.0, top_p=1.0),
    "low_temp": SamplingConfig(temperature=0.3, top_p=0.9),
    "high_temp": SamplingConfig(temperature=1.0, top_p=0.95),
    "creative": SamplingConfig(temperature=1.2, top_p=0.95),
    "diverse": SamplingConfig(temperature=0.9, top_p=0.98, repetition_penalty=1.1),
}


async def run_parallel_collection(
    collector: AsyncTrajectoryCollector,
    dataset,
    output_path: str,
    processed_ids: set,
    max_iterations: int,
    top_k: int,
    concurrency: int,
    resume: bool
):
    """Run parallel trajectory collection with controlled concurrency."""
    
    # Semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrency)
    # Lock for safe file writing
    file_lock = asyncio.Lock()
    
    # Stats tracking (thread-safe via lock)
    stats = {"total": 0, "success": 0, "gt_found": 0, "errors": 0}
    stats_lock = asyncio.Lock()
    
    # Open file in append mode for safe concurrent writing
    mode = 'a' if resume else 'w'
    output_file = open(output_path, mode, encoding='utf-8')
    
    async def process_item(idx: int, item: Dict) -> None:
        """Process a single item with semaphore control."""
        question_id = item.get("id", f"unknown_{idx}")
        traj_id = f"traj_{question_id}"
        
        # Skip if already processed (check both formats for backward compat)
        if traj_id in processed_ids or question_id in processed_ids:
            return
        
        async with semaphore:
            ground_truth = {
                "answers": item.get("answers"),
                "answer_locations": item.get("answer_locations"),
                "category": item.get("category")
            }
            
            trajectory = await collector.collect_trajectory(
                question=item["question"],
                ground_truth=ground_truth,
                trajectory_id=traj_id,
                question_id=question_id,
                max_iterations=max_iterations,
                top_k=top_k
            )
            
            # Update stats
            async with stats_lock:
                stats["total"] += 1
                if trajectory.success:
                    stats["success"] += 1
                if trajectory.gt_ever_found:
                    stats["gt_found"] += 1
                if trajectory.error:
                    stats["errors"] += 1
                current_total = stats["total"]
            
            # Write to file with lock to prevent corruption
            traj_dict = trajectory_to_dict(trajectory)
            json_line = json.dumps(traj_dict, ensure_ascii=False) + '\n'
            
            async with file_lock:
                output_file.write(json_line)
                output_file.flush()  # Ensure written to disk
            
            # Progress update every 10 completions
            if current_total % 10 == 0:
                async with stats_lock:
                    print(f"\nStats: {stats['success']}/{stats['total']} success, "
                          f"{stats['gt_found']} GT found, {stats['errors']} errors")
    
    # Create all tasks
    tasks = [
        process_item(idx, item) 
        for idx, item in enumerate(dataset)
    ]
    
    # Run with progress bar
    print(f"\nStarting parallel collection with concurrency={concurrency}")
    for coro in atqdm.as_completed(tasks, desc="Collecting trajectories", total=len(tasks)):
        await coro
    
    output_file.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Collect trajectories for finetuning")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--ocr-file", required=True, help="Path to OCR results JSONL")
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model name")
    parser.add_argument("--base-url", help="vLLM server URL")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max iterations")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k results per search")
    parser.add_argument("--dataset", default="agentic-document-ai/agentic-document-ai", 
                       help="Dataset name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--exclude-json", type=str, action="append",
                       help="Path to JSON file with questions to exclude (can be repeated)")
    parser.add_argument("--no-default-excludes", action="store_true",
                       help="Don't exclude default test/dev splits")
    parser.add_argument("--concurrency", type=int, default=8,
                       help="Number of concurrent requests (default: 8)")
    parser.add_argument("--sequential", action="store_true",
                       help="Use sequential (non-parallel) collection")
    
    # Sampling parameters
    parser.add_argument("--sampling-config", choices=list(SAMPLING_CONFIGS.keys()), 
                       default="default", help="Predefined sampling config")
    parser.add_argument("--temperature", type=float, help="Override temperature")
    parser.add_argument("--top-p", type=float, help="Override top_p")
    parser.add_argument("--repetition-penalty", type=float, help="Override repetition penalty")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset: {args.dataset} ({args.split})")
    dataset = load_dataset(args.dataset, split=args.split)
    print(f"Loaded {len(dataset)} questions")
    
    # Build list of exclude files
    exclude_files = args.exclude_json or []
    if not args.no_default_excludes:
        # Default: exclude test and dev splits
        script_dir = Path(__file__).parent
        default_excludes = [
            script_dir / "../../splits/ctt_subset_test.json",
            script_dir / "../../splits/ctt_subset_dev.json",
        ]
        exclude_files = [str(p) for p in default_excludes if p.exists()] + exclude_files
    
    # Exclude questions from specified files
    if exclude_files:
        exclude_ids = set()
        for exclude_file in exclude_files:
            exclude_path = Path(exclude_file)
            if exclude_path.exists():
                with open(exclude_path, 'r') as f:
                    exclude_data = json.load(f)
                # Extract question IDs to exclude
                if "items" in exclude_data:
                    file_ids = {item["question_id"] for item in exclude_data["items"]}
                elif isinstance(exclude_data, list):
                    file_ids = {item.get("question_id") or item.get("id") for item in exclude_data}
                else:
                    file_ids = set()
                exclude_ids.update(file_ids)
                print(f"Loaded {len(file_ids)} question IDs to exclude from {exclude_path.name}")
            else:
                print(f"Warning: exclude file not found: {exclude_file}")
        
        if exclude_ids:
            original_len = len(dataset)
            dataset = dataset.filter(lambda x: x["id"] not in exclude_ids)
            print(f"Excluded {original_len - len(dataset)} questions total")
            print(f"Remaining: {len(dataset)} questions")
    
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limited to {len(dataset)} questions")
    
    # Setup sampling config
    sampling_config = SAMPLING_CONFIGS[args.sampling_config]
    if args.temperature is not None:
        sampling_config.temperature = args.temperature
    if args.top_p is not None:
        sampling_config.top_p = args.top_p
    if args.repetition_penalty is not None:
        sampling_config.repetition_penalty = args.repetition_penalty
    
    # Initialize search engine
    search_engine = WhooshSearchEngine(args.ocr_file)
    
    # Check resume - load processed IDs
    processed_ids = set()
    if args.resume and os.path.exists(args.output):
        print(f"Resuming from {args.output}")
        with open(args.output, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        # Track both id and question_id for backward compat
                        processed_ids.add(data.get('id'))
                        if data.get('question_id'):
                            processed_ids.add(data.get('question_id'))
                    except json.JSONDecodeError:
                        continue
        print(f"Found {len(processed_ids)} already processed")
    
    if args.sequential:
        # Use original sequential collector
        collector = TrajectoryCollector(
            search_engine=search_engine,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            sampling_config=sampling_config
        )
        
        mode = 'a' if args.resume else 'w'
        stats = {"total": 0, "success": 0, "gt_found": 0, "errors": 0}
        
        with open(args.output, mode, encoding='utf-8') as f:
            for idx, item in enumerate(tqdm(dataset, desc="Collecting trajectories")):
                question_id = item.get("id", f"unknown_{idx}")
                traj_id = f"traj_{question_id}"
                
                if traj_id in processed_ids or question_id in processed_ids:
                    continue
                
                ground_truth = {
                    "answers": item.get("answers"),
                    "answer_locations": item.get("answer_locations"),
                    "category": item.get("category")
                }
                
                trajectory = collector.collect_trajectory(
                    question=item["question"],
                    ground_truth=ground_truth,
                    trajectory_id=traj_id,
                    question_id=question_id,
                    max_iterations=args.max_iterations,
                    top_k=args.top_k
                )
                
                stats["total"] += 1
                if trajectory.success:
                    stats["success"] += 1
                if trajectory.gt_ever_found:
                    stats["gt_found"] += 1
                if trajectory.error:
                    stats["errors"] += 1
                
                traj_dict = trajectory_to_dict(trajectory)
                f.write(json.dumps(traj_dict, ensure_ascii=False) + '\n')
                f.flush()
                
                if stats["total"] % 10 == 0:
                    print(f"\nStats: {stats['success']}/{stats['total']} success, "
                          f"{stats['gt_found']} GT found, {stats['errors']} errors")
    else:
        # Use parallel async collector
        collector = AsyncTrajectoryCollector(
            search_engine=search_engine,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            sampling_config=sampling_config
        )
        
        stats = asyncio.run(run_parallel_collection(
            collector=collector,
            dataset=dataset,
            output_path=args.output,
            processed_ids=processed_ids,
            max_iterations=args.max_iterations,
            top_k=args.top_k,
            concurrency=args.concurrency,
            resume=args.resume
        ))
    
    print(f"\n{'='*60}")
    print(f"Collection complete!")
    print(f"Total: {stats['total']}")
    print(f"Success: {stats['success']} ({100*stats['success']/max(1,stats['total']):.1f}%)")
    print(f"GT Found: {stats['gt_found']} ({100*stats['gt_found']/max(1,stats['total']):.1f}%)")
    print(f"Errors: {stats['errors']}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
