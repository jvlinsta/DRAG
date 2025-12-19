#!/usr/bin/env python3
"""
Search Agent Evaluation and Inference

Unified script that supports:
1. Single question answering (interactive mode via --question)
2. Dataset evaluation with metrics (batch mode)
3. Model comparison (via --compare)

Metrics computed:
- Iterations to success
- First-hit rank (GT doc rank on first search)
- MRR (Mean Reciprocal Rank)
- Success rate
- Hit rate at different ranks (Hit@1, Hit@3, Hit@5)

This script subsumes the functionality of vllm_search_agent.py.
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import openai
from PIL import Image
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from datasets import load_dataset
from anls_star import anls_score

from search_engine import WhooshSearchEngine
from utils import get_pdf_page_as_png, image_to_base64, resize_image_if_needed


def compute_anls_star(predicted: Any, ground_truths: List[List[str]]) -> float:
    """
    Calculate ANLS* score using the official anls_star library.
    Supports nested ground truths (list of lists) and case-insensitive matching.
    
    Args:
        predicted: Predicted answer (can be string or list of strings)
        ground_truths: List of alternative ground truth answers, where each alternative is a list of strings
    
    Returns:
        Maximum ANLS* score across all ground truth alternatives
    """
    if not ground_truths:
        return 0.0

    # Ensure predicted is a list
    if isinstance(predicted, str):
        predicted = [predicted]
    
    # Convert predicted to lowercase
    predicted_lower = [p.lower() if isinstance(p, str) else str(p).lower() for p in predicted]
    
    # Calculate score for each gold variant and take the maximum
    max_score = 0.0
    for gold_variant in ground_truths:
        # Ensure gold_variant is a list
        if isinstance(gold_variant, str):
            gold_variant = [gold_variant]
        
        # Convert gold_variant to lowercase
        gold_variant_lower = [g.lower() if isinstance(g, str) else str(g).lower() for g in gold_variant]
        
        # Calculate ANLS score for this variant
        score = anls_score(predicted_lower, gold_variant_lower)
        max_score = max(max_score, score)
    
    return max_score


def compute_citation_f1(
    predicted_citations: List[Dict[str, Any]],
    ground_truth_locations: List[Dict[str, Any]],
    level: str = 'page'
) -> Dict[str, float]:
    """
    Calculate Citation F1 score at document or page level.
    
    Args:
        predicted_citations: List of citations with 'file'/'document' and 'page'
        ground_truth_locations: List of ground truth locations with 'document' and 'page'
        level: 'document' or 'page' - granularity of citation matching
    
    Returns:
        Dict with 'precision', 'recall', 'f1', and 'support'
    """
    if not ground_truth_locations:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}
    
    # Extract ground truth citations (use 'document' field)
    if level == 'document':
        gt_citations = {loc.get('document') for loc in ground_truth_locations if loc.get('document')}
    else:  # page level
        gt_citations = {
            (loc.get('document'), loc.get('page')) 
            for loc in ground_truth_locations 
            if loc.get('document') is not None
        }
    
    # Extract predicted citations (use 'file' or 'document' field)
    if not predicted_citations:
        pred_citations = set()
    else:
        if level == 'document':
            pred_citations = {
                cite.get('file') or cite.get('document') 
                for cite in predicted_citations 
                if (cite.get('file') or cite.get('document'))
            }
        else:  # page level
            pred_citations = {
                (cite.get('file') or cite.get('document'), cite.get('page')) 
                for cite in predicted_citations 
                if (cite.get('file') or cite.get('document')) is not None
            }
    
    # Remove None values
    gt_citations = {c for c in gt_citations if c is not None and (not isinstance(c, tuple) or None not in c)}
    pred_citations = {c for c in pred_citations if c is not None and (not isinstance(c, tuple) or None not in c)}
    
    if not gt_citations:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}
    
    # Calculate metrics
    true_positives = len(gt_citations & pred_citations)
    
    precision = true_positives / len(pred_citations) if pred_citations else 0.0
    recall = true_positives / len(gt_citations) if gt_citations else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': len(gt_citations)
    }


def parse_tag_response(response: str) -> dict:
    """Parse model response into think and search components.
    
    Expected format (from fine-tuned model):
        <think>reasoning here</think>
        <search>query here</search>
    
    Returns:
        dict with 'think', 'search', and 'raw' keys
    """
    result = {"raw": response, "think": None, "search": None}
    
    # Extract <think> content
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if think_match:
        result["think"] = think_match.group(1).strip()
    
    # Extract <search> content
    search_match = re.search(r"<search>(.*?)</search>", response, re.DOTALL)
    if search_match:
        result["search"] = search_match.group(1).strip()
    
    return result


@dataclass
class SearchMetrics:
    """Metrics for a single search."""
    query: str
    results: List[Dict[str, Any]]
    gt_file: Optional[str]
    gt_page: Optional[int]
    gt_found: bool = False
    gt_rank: Optional[int] = None
    reciprocal_rank: float = 0.0


@dataclass
class QuestionMetrics:
    """Metrics for a single question."""
    question_id: str
    question: str
    ground_truth: Dict[str, Any]
    searches: List[SearchMetrics] = field(default_factory=list)
    
    success: bool = False
    iterations_to_success: Optional[int] = None
    first_hit_rank: Optional[int] = None
    first_search_mrr: float = 0.0
    best_rank: Optional[int] = None
    best_mrr: float = 0.0
    total_iterations: int = 0
    
    final_answer: Optional[List[str]] = None
    final_citations: Optional[List[Dict]] = None
    answer_correct: Optional[bool] = None
    
    # ANLS* and Citation F1 metrics
    anls_score: float = 0.0
    page_f1: float = 0.0
    page_precision: float = 0.0
    page_recall: float = 0.0
    doc_f1: float = 0.0
    doc_precision: float = 0.0
    doc_recall: float = 0.0


@dataclass
class EvaluationResults:
    """Aggregate evaluation results."""
    model: str
    total_questions: int = 0
    
    success_count: int = 0
    success_rate: float = 0.0
    
    avg_iterations: float = 0.0
    avg_iterations_to_success: float = 0.0
    
    first_hit_rate: float = 0.0
    avg_first_hit_rank: float = 0.0
    first_search_mrr: float = 0.0
    
    avg_best_rank: float = 0.0
    best_search_mrr: float = 0.0
    
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    
    # ANLS* and Citation F1 aggregate metrics
    avg_anls: float = 0.0
    anls_accuracy: float = 0.0  # % with ANLS >= 0.5
    avg_page_f1: float = 0.0
    avg_page_precision: float = 0.0
    avg_page_recall: float = 0.0
    avg_doc_f1: float = 0.0
    avg_doc_precision: float = 0.0
    avg_doc_recall: float = 0.0
    
    question_metrics: List[QuestionMetrics] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "total_questions": self.total_questions,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "avg_iterations": self.avg_iterations,
            "avg_iterations_to_success": self.avg_iterations_to_success,
            "first_hit_rate": self.first_hit_rate,
            "avg_first_hit_rank": self.avg_first_hit_rank,
            "first_search_mrr": self.first_search_mrr,
            "avg_best_rank": self.avg_best_rank,
            "best_search_mrr": self.best_search_mrr,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "avg_anls": self.avg_anls,
            "anls_accuracy": self.anls_accuracy,
            "avg_page_f1": self.avg_page_f1,
            "avg_page_precision": self.avg_page_precision,
            "avg_page_recall": self.avg_page_recall,
            "avg_doc_f1": self.avg_doc_f1,
            "avg_doc_precision": self.avg_doc_precision,
            "avg_doc_recall": self.avg_doc_recall,
        }


class SearchEvaluator:
    """Evaluates search agent performance."""
    
    def __init__(
        self,
        search_engine: WhooshSearchEngine,
        model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tag_mode: bool = False
    ):
        self.search_engine = search_engine
        self.model = model
        self.tag_mode = tag_mode
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8001/v1")
        api_key = api_key or os.environ.get("VLLM_API_KEY", "abc123")
        self.client = openai.OpenAI(base_url=self.base_url, api_key=api_key)
    
    def _load_page_image(self, file: str, page: int) -> Dict:
        """Load a page as an image."""
        image = get_pdf_page_as_png(file, page)
        _, base64_image = resize_image_if_needed(image)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": "high"
            }
        }
    
    def _parse_answer_from_text(self, text: str) -> Dict[str, Any]:
        """Try to parse structured answer from text response.
        
        Falls back to returning the raw text if parsing fails.
        """
        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*"answer"[\s\S]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                answer = data.get("answer", [])
                if isinstance(answer, str):
                    answer = [answer]
                citations = data.get("citations", [])
                return {"answer": answer, "citations": citations}
            except json.JSONDecodeError:
                pass
        
        # Look for answer patterns
        answer_match = re.search(r'(?:answer|Answer|ANSWER)[:\s]+([^\n]+)', text)
        if answer_match:
            return {"answer": [answer_match.group(1).strip()], "citations": []}
        
        # Return raw text as answer
        return {"answer": [text.strip()], "citations": []}
    
    def _extract_ground_truth(self, item: Dict[str, Any]) -> tuple[Optional[str], Optional[int]]:
        """Extract ground truth file and page."""
        gt_file = None
        gt_page = None
        
        if "answer_locations" in item and item["answer_locations"]:
            loc = item["answer_locations"]
            if isinstance(loc, list) and len(loc) > 0:
                loc = loc[0]
            
            if isinstance(loc, dict):
                gt_file = loc.get("document") or loc.get("file")
                gt_page = loc.get("page")
            elif isinstance(loc, str):
                parts = loc.split(":")
                if len(parts) >= 2:
                    gt_file = parts[0]
                    try:
                        gt_page = int(parts[1])
                    except ValueError:
                        pass
        
        return gt_file, gt_page
    
    def _check_result_matches_gt(
        self,
        result: Dict[str, Any],
        gt_file: str,
        gt_page: int
    ) -> bool:
        """Check if result matches ground truth."""
        return (
            result.get("file") == gt_file and 
            result.get("page_number") == gt_page
        )
    
    def _compute_search_metrics(
        self,
        query: str,
        results: List[Dict[str, Any]],
        gt_file: Optional[str],
        gt_page: Optional[int]
    ) -> SearchMetrics:
        """Compute metrics for a single search."""
        metrics = SearchMetrics(
            query=query,
            results=results,
            gt_file=gt_file,
            gt_page=gt_page
        )
        
        if gt_file is None or gt_page is None:
            return metrics
        
        for rank, result in enumerate(results, 1):
            if self._check_result_matches_gt(result, gt_file, gt_page):
                metrics.gt_found = True
                metrics.gt_rank = rank
                metrics.reciprocal_rank = 1.0 / rank
                break
        
        return metrics
    
    def _run_agent(
        self,
        question: str,
        max_iterations: int = 5,
        top_k: int = 3,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Run the search agent on a question."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search document collection and return images of matching pages. Supports: terms and phrases (use quotes for exact match), boolean operators (AND, OR, NOT - AND is default), wildcards (* for multiple chars, ? for single char). Examples: 'engine specifications', '\"Bell 407\" AND accessories', 'Bell*', 'incorporation NOT date'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query using keywords, phrases in quotes, and boolean operators"
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
                    "description": "Provide the final structured answer with citations after analyzing the document pages.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of answer values (one or more items)"
                            },
                            "citations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "description": "The exact PDF filename (e.g., '1007969.pdf')"
                                        },
                                        "page": {
                                            "type": "integer",
                                            "description": "The page number"
                                        }
                                    },
                                    "required": ["file", "page"]
                                },
                                "description": "List of citations with file and page information"
                            }
                        },
                        "required": ["answer", "citations"]
                    }
                }
            }
        ]
        
        system_prompt = """You are a document QA assistant with access to a search tool.
The search tool returns images of document pages.

IMPORTANT: The answer to the question is definitely in the documents. If your search returns no results or unhelpful pages, try different search terms. Be creative with queries - use synonyms, abbreviations, or different phrasings.

Once you find relevant pages, analyze the images carefully. When you have the answer, use the provide_answer tool with:
- answer: list of answer values (one or more items)
  * if there is a single answer, the output should be a one-element list
  * if the answer refers to multiple items or entities, the list will have several elements
  * do not write a full sentence there, use as few words as possible
  * if possible, use the exact words from the document
- citations: list of sources where EACH citation must have:
  * file: the exact PDF filename shown in the image (e.g., "1007969.pdf", "doc_name.pdf")
  * page: the page number (integer)

Always use one of the available tools (search_documents or provide_answer)."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        search_history = []
        
        for iteration in range(1, max_iterations + 1):
            if verbose:
                print(f"\n=== Iteration {iteration} ===")
            
            try:
                if iteration == max_iterations:
                    # Force answer on last iteration
                    force_message = {
                        "role": "user", 
                        "content": "You must now provide your final answer using the provide_answer tool based on what you've found."
                    }
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages + [force_message],
                        tools=tools,
                        tool_choice={"type": "function", "function": {"name": "provide_answer"}}
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto"
                    )
            except Exception as e:
                print(f"API Error on iteration {iteration}: {e}")
                # Retry without tool_choice if it fails
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools
                    )
                except Exception as e2:
                    print(f"Retry also failed: {e2}")
                    break
            
            message = response.choices[0].message
            
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "provide_answer":
                        try:
                            answer_data = json.loads(tool_call.function.arguments)
                            return {
                                "question": question,
                                "answer": answer_data.get("answer", []),
                                "citations": answer_data.get("citations", []),
                                "iterations": iteration,
                                "search_history": search_history,
                                "model": self.model
                            }
                        except json.JSONDecodeError:
                            pass
                    
                    elif tool_call.function.name == "search_documents":
                        try:
                            query = json.loads(tool_call.function.arguments)["query"]
                        except (json.JSONDecodeError, KeyError):
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": "Error: Invalid search query format"
                            })
                            continue
                        
                        if verbose:
                            print(f"Searching: {query}")
                        
                        results = self.search_engine.search(query, top_k)
                        
                        if verbose:
                            print(f"Returning {len(results)} results to model")
                        
                        search_history.append({
                            "iteration": iteration,
                            "query": query,
                            "results": results,
                            "num_results": len(results)
                        })
                        
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
                            "content": f"Found {len(results)} matching pages."
                        })
                        
                        if results:
                            image_content: List[Dict[str, Any]] = [
                                {"type": "text", "text": "Here are the matching pages:\n"}
                            ]
                            for result in results:
                                image_content.append({
                                    "type": "text",
                                    "text": f"\nFile: {result['file']}, Page: {result['page_number']}"
                                })
                                try:
                                    image_content.append(
                                        self._load_page_image(result['file'], result['page_number'])
                                    )
                                except Exception as e:
                                    print(f"Warning: Could not load {result['file']} page {result['page_number']}: {e}")
                            
                            messages.append({"role": "user", "content": image_content})
            else:
                # No tool calls - try to parse answer from text
                text_content = message.content or ""
                if verbose:
                    print(f"No tool call, got text response: {text_content[:200]}...")
                
                parsed = self._parse_answer_from_text(text_content)
                return {
                    "question": question,
                    "answer": parsed["answer"],
                    "citations": parsed["citations"],
                    "iterations": iteration,
                    "search_history": search_history,
                    "model": self.model,
                    "raw_response": text_content
                }
        
        return {
            "question": question,
            "answer": [],
            "citations": [],
            "iterations": max_iterations,
            "search_history": search_history,
            "model": self.model,
            "error": "Max iterations reached"
        }
    
    def _run_agent_tag_mode(
        self,
        question: str,
        max_iterations: int = 5,
        top_k: int = 3,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Run agent using <question>/<search> tag format (for fine-tuned models).
        
        This mode is for models fine-tuned with:
        - Input: <question>...</question>
        - Output: <think>...</think><search>...</search>
        
        Does NOT use tool calling - parses tags directly from model output.
        """
        # Format question with tags (matching training format)
        formatted_question = f"<question>{question}</question>"
        
        system_prompt = """You are a document search assistant. Given a question, think about what to search for and output a search query.

Output format:
<think>your reasoning about what to search</think>
<search>your search query</search>"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_question}
        ]
        
        search_history = []
        
        for iteration in range(1, max_iterations + 1):
            if verbose:
                print(f"\n=== Iteration {iteration} ===")
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            except Exception as e:
                print(f"API Error on iteration {iteration}: {e}")
                break
            
            message = response.choices[0].message
            text_content = message.content or ""
            
            if verbose:
                print(f"Response: {text_content[:300]}...")
            
            # Parse the tag-based response
            parsed = parse_tag_response(text_content)
            
            if verbose:
                if parsed["think"]:
                    print(f"Think: {parsed['think'][:200]}...")
                if parsed["search"]:
                    print(f"Search: {parsed['search']}")
            
            # If we got a search query, execute it
            if parsed["search"]:
                query = parsed["search"]
                
                if verbose:
                    print(f"Executing search: {query}")
                
                results = self.search_engine.search(query, top_k)
                
                if verbose:
                    print(f"Found {len(results)} results")
                
                search_history.append({
                    "iteration": iteration,
                    "query": query,
                    "results": results,
                    "num_results": len(results),
                    "think": parsed["think"]
                })
                
                # Add assistant response to conversation
                messages.append({
                    "role": "assistant",
                    "content": text_content
                })
                
                # Add search results as user message (simplified - no images in tag mode)
                if results:
                    result_text = f"Search results for '{query}':\n"
                    for i, r in enumerate(results, 1):
                        result_text += f"  {i}. {r['file']} page {r['page_number']}\n"
                    
                    # For tag mode, we just report what was found
                    # The model was trained to generate queries, not full agent loops
                    # So we return after first search for evaluation
                    return {
                        "question": question,
                        "answer": [],
                        "citations": [],
                        "iterations": iteration,
                        "search_history": search_history,
                        "model": self.model,
                        "tag_mode": True
                    }
                else:
                    # No results, ask for another query
                    messages.append({
                        "role": "user",
                        "content": f"No results found for '{query}'. Please try a different search query.\n\n{formatted_question}"
                    })
            else:
                # No search tag found - model didn't produce expected format
                if verbose:
                    print("No <search> tag found in response")
                
                # Try to continue
                messages.append({
                    "role": "assistant",
                    "content": text_content
                })
                messages.append({
                    "role": "user", 
                    "content": f"Please output a search query in this format: <search>your query</search>\n\n{formatted_question}"
                })
        
        return {
            "question": question,
            "answer": [],
            "citations": [],
            "iterations": max_iterations,
            "search_history": search_history,
            "model": self.model,
            "tag_mode": True,
            "error": "Max iterations reached"
        }
    
    def evaluate_question(
        self,
        question: str,
        ground_truth: Dict[str, Any],
        question_id: str,
        max_iterations: int = 5,
        top_k: int = 3
    ) -> QuestionMetrics:
        """Evaluate agent on a single question."""
        gt_file, gt_page = self._extract_ground_truth(ground_truth)
        
        metrics = QuestionMetrics(
            question_id=question_id,
            question=question,
            ground_truth=ground_truth
        )
        
        # Choose method based on tag_mode
        if self.tag_mode:
            result = self._run_agent_tag_mode(question, max_iterations, top_k)
        else:
            result = self._run_agent(question, max_iterations, top_k)
        
        metrics.final_answer = result.get("answer")
        metrics.final_citations = result.get("citations")
        metrics.total_iterations = result.get("iterations", 0)
        
        for search in result.get("search_history", []):
            search_metrics = self._compute_search_metrics(
                query=search["query"],
                results=search["results"],
                gt_file=gt_file,
                gt_page=gt_page
            )
            metrics.searches.append(search_metrics)
            
            if search_metrics.gt_found and metrics.iterations_to_success is None:
                metrics.iterations_to_success = search["iteration"]
                metrics.success = True
        
        if metrics.searches:
            first_search = metrics.searches[0]
            metrics.first_hit_rank = first_search.gt_rank
            metrics.first_search_mrr = first_search.reciprocal_rank
            
            best_search = max(metrics.searches, key=lambda s: s.reciprocal_rank)
            metrics.best_rank = best_search.gt_rank
            metrics.best_mrr = best_search.reciprocal_rank
        
        # Compute ANLS* score
        gold_answers = ground_truth.get("answers", [])
        if gold_answers and metrics.final_answer:
            metrics.anls_score = compute_anls_star(metrics.final_answer, gold_answers)
        
        # Compute Citation F1 metrics
        gt_locations = ground_truth.get("answer_locations", [])
        if gt_locations and metrics.final_citations:
            page_f1_metrics = compute_citation_f1(metrics.final_citations, gt_locations, level='page')
            metrics.page_f1 = page_f1_metrics['f1']
            metrics.page_precision = page_f1_metrics['precision']
            metrics.page_recall = page_f1_metrics['recall']
            
            doc_f1_metrics = compute_citation_f1(metrics.final_citations, gt_locations, level='document')
            metrics.doc_f1 = doc_f1_metrics['f1']
            metrics.doc_precision = doc_f1_metrics['precision']
            metrics.doc_recall = doc_f1_metrics['recall']
        
        return metrics
    
    def evaluate_dataset(
        self,
        dataset,
        max_iterations: int = 5,
        top_k: int = 3,
        limit: Optional[int] = None
    ) -> EvaluationResults:
        """Evaluate agent on full dataset."""
        results = EvaluationResults(model=self.model)
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        all_iterations = []
        iterations_to_success = []
        first_hit_ranks = []
        first_mrrs = []
        best_ranks = []
        best_mrrs = []
        hit_at_1_count = 0
        hit_at_3_count = 0
        hit_at_5_count = 0
        
        # ANLS* and Citation F1 trackers
        all_anls = []
        all_page_f1 = []
        all_page_precision = []
        all_page_recall = []
        all_doc_f1 = []
        all_doc_precision = []
        all_doc_recall = []
        
        for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
            ground_truth = {
                "answers": item.get("answers"),
                "answer_locations": item.get("answer_locations"),
                "category": item.get("category")
            }
            
            q_metrics = self.evaluate_question(
                question=item["question"],
                ground_truth=ground_truth,
                question_id=item.get("id", f"q_{idx}"),
                max_iterations=max_iterations,
                top_k=top_k
            )
            
            results.question_metrics.append(q_metrics)
            results.total_questions += 1
            
            all_iterations.append(q_metrics.total_iterations)
            
            if q_metrics.success:
                results.success_count += 1
                if q_metrics.iterations_to_success is not None:
                    iterations_to_success.append(q_metrics.iterations_to_success)
            
            if q_metrics.first_hit_rank is not None:
                first_hit_ranks.append(q_metrics.first_hit_rank)
                first_mrrs.append(q_metrics.first_search_mrr)
                
                if q_metrics.first_hit_rank == 1:
                    hit_at_1_count += 1
                if q_metrics.first_hit_rank <= 3:
                    hit_at_3_count += 1
                if q_metrics.first_hit_rank <= 5:
                    hit_at_5_count += 1
            
            if q_metrics.best_rank is not None:
                best_ranks.append(q_metrics.best_rank)
                best_mrrs.append(q_metrics.best_mrr)
            
            # Track ANLS* and Citation F1
            all_anls.append(q_metrics.anls_score)
            all_page_f1.append(q_metrics.page_f1)
            all_page_precision.append(q_metrics.page_precision)
            all_page_recall.append(q_metrics.page_recall)
            all_doc_f1.append(q_metrics.doc_f1)
            all_doc_precision.append(q_metrics.doc_precision)
            all_doc_recall.append(q_metrics.doc_recall)
        
        n = results.total_questions
        
        results.success_rate = results.success_count / n if n > 0 else 0
        results.avg_iterations = sum(all_iterations) / n if n > 0 else 0
        
        if iterations_to_success:
            results.avg_iterations_to_success = sum(iterations_to_success) / len(iterations_to_success)
        
        if first_hit_ranks:
            results.first_hit_rate = len(first_hit_ranks) / n
            results.avg_first_hit_rank = sum(first_hit_ranks) / len(first_hit_ranks)
            results.first_search_mrr = sum(first_mrrs) / len(first_mrrs)
        
        if best_ranks:
            results.avg_best_rank = sum(best_ranks) / len(best_ranks)
            results.best_search_mrr = sum(best_mrrs) / len(best_mrrs)
        
        results.hit_at_1 = hit_at_1_count / n if n > 0 else 0
        results.hit_at_3 = hit_at_3_count / n if n > 0 else 0
        results.hit_at_5 = hit_at_5_count / n if n > 0 else 0
        
        # Compute ANLS* and Citation F1 aggregates
        if all_anls:
            results.avg_anls = sum(all_anls) / len(all_anls)
            results.anls_accuracy = sum(1 for s in all_anls if s >= 0.5) / len(all_anls)
        if all_page_f1:
            results.avg_page_f1 = sum(all_page_f1) / len(all_page_f1)
            results.avg_page_precision = sum(all_page_precision) / len(all_page_precision)
            results.avg_page_recall = sum(all_page_recall) / len(all_page_recall)
        if all_doc_f1:
            results.avg_doc_f1 = sum(all_doc_f1) / len(all_doc_f1)
            results.avg_doc_precision = sum(all_doc_precision) / len(all_doc_precision)
            results.avg_doc_recall = sum(all_doc_recall) / len(all_doc_recall)
        
        return results


class AsyncSearchEvaluator:
    """Async version of SearchEvaluator for parallel evaluation to maximize vLLM throughput."""
    
    def __init__(
        self,
        search_engine: WhooshSearchEngine,
        model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tag_mode: bool = False
    ):
        self.search_engine = search_engine
        self.model = model
        self.tag_mode = tag_mode
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8001/v1")
        api_key = api_key or os.environ.get("VLLM_API_KEY", "abc123")
        self.client = openai.AsyncOpenAI(base_url=self.base_url, api_key=api_key)
    
    def _load_page_image(self, file: str, page: int) -> Dict:
        """Load a page as an image."""
        image = get_pdf_page_as_png(file, page)
        _, base64_image = resize_image_if_needed(image)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": "high"
            }
        }
    
    def _parse_answer_from_text(self, text: str) -> Dict[str, Any]:
        """Try to parse structured answer from text response."""
        json_match = re.search(r'\{[\s\S]*"answer"[\s\S]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                answer = data.get("answer", [])
                if isinstance(answer, str):
                    answer = [answer]
                citations = data.get("citations", [])
                return {"answer": answer, "citations": citations}
            except json.JSONDecodeError:
                pass
        
        answer_match = re.search(r'(?:answer|Answer|ANSWER)[:\s]+([^\n]+)', text)
        if answer_match:
            return {"answer": [answer_match.group(1).strip()], "citations": []}
        
        return {"answer": [text.strip()], "citations": []}
    
    def _extract_ground_truth(self, item: Dict[str, Any]) -> tuple[Optional[str], Optional[int]]:
        """Extract ground truth file and page."""
        gt_file = None
        gt_page = None
        
        if "answer_locations" in item and item["answer_locations"]:
            loc = item["answer_locations"]
            if isinstance(loc, list) and len(loc) > 0:
                loc = loc[0]
            
            if isinstance(loc, dict):
                gt_file = loc.get("document") or loc.get("file")
                gt_page = loc.get("page")
            elif isinstance(loc, str):
                parts = loc.split(":")
                if len(parts) >= 2:
                    gt_file = parts[0]
                    try:
                        gt_page = int(parts[1])
                    except ValueError:
                        pass
        
        return gt_file, gt_page
    
    def _check_result_matches_gt(
        self,
        result: Dict[str, Any],
        gt_file: str,
        gt_page: int
    ) -> bool:
        """Check if result matches ground truth."""
        return (
            result.get("file") == gt_file and 
            result.get("page_number") == gt_page
        )
    
    def _compute_search_metrics(
        self,
        query: str,
        results: List[Dict[str, Any]],
        gt_file: Optional[str],
        gt_page: Optional[int]
    ) -> SearchMetrics:
        """Compute metrics for a single search."""
        metrics = SearchMetrics(
            query=query,
            results=results,
            gt_file=gt_file,
            gt_page=gt_page
        )
        
        if gt_file is None or gt_page is None:
            return metrics
        
        for rank, result in enumerate(results, 1):
            if self._check_result_matches_gt(result, gt_file, gt_page):
                metrics.gt_found = True
                metrics.gt_rank = rank
                metrics.reciprocal_rank = 1.0 / rank
                break
        
        return metrics
    
    async def _run_agent(
        self,
        question: str,
        max_iterations: int = 5,
        top_k: int = 3,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Run the search agent on a question (async version)."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search document collection and return images of matching pages. Supports: terms and phrases (use quotes for exact match), boolean operators (AND, OR, NOT - AND is default), wildcards (* for multiple chars, ? for single char). Examples: 'engine specifications', '\"Bell 407\" AND accessories', 'Bell*', 'incorporation NOT date'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query using keywords, phrases in quotes, and boolean operators"
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
                    "description": "Provide the final structured answer with citations after analyzing the document pages.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of answer values (one or more items)"
                            },
                            "citations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "description": "The exact PDF filename (e.g., '1007969.pdf')"
                                        },
                                        "page": {
                                            "type": "integer",
                                            "description": "The page number"
                                        }
                                    },
                                    "required": ["file", "page"]
                                },
                                "description": "List of citations with file and page information"
                            }
                        },
                        "required": ["answer", "citations"]
                    }
                }
            }
        ]
        
        system_prompt = """You are a document QA assistant with access to a search tool.
The search tool returns images of document pages.

IMPORTANT: The answer to the question is definitely in the documents. If your search returns no results or unhelpful pages, try different search terms. Be creative with queries - use synonyms, abbreviations, or different phrasings.

Once you find relevant pages, analyze the images carefully. When you have the answer, use the provide_answer tool with:
- answer: list of answer values (one or more items)
  * if there is a single answer, the output should be a one-element list
  * if the answer refers to multiple items or entities, the list will have several elements
  * do not write a full sentence there, use as few words as possible
  * if possible, use the exact words from the document
- citations: list of sources where EACH citation must have:
  * file: the exact PDF filename shown in the image (e.g., "1007969.pdf", "doc_name.pdf")
  * page: the page number (integer)

Always use one of the available tools (search_documents or provide_answer)."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        search_history = []
        
        for iteration in range(1, max_iterations + 1):
            if verbose:
                print(f"\n=== Iteration {iteration} ===")
            
            try:
                if iteration == max_iterations:
                    force_message = {
                        "role": "user", 
                        "content": "You must now provide your final answer using the provide_answer tool based on what you've found."
                    }
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages + [force_message],
                        tools=tools,
                        tool_choice={"type": "function", "function": {"name": "provide_answer"}}
                    )
                else:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto"
                    )
            except Exception as e:
                if verbose:
                    print(f"API Error on iteration {iteration}: {e}")
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools
                    )
                except Exception as e2:
                    if verbose:
                        print(f"Retry also failed: {e2}")
                    break
            
            message = response.choices[0].message
            
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "provide_answer":
                        try:
                            answer_data = json.loads(tool_call.function.arguments)
                            return {
                                "question": question,
                                "answer": answer_data.get("answer", []),
                                "citations": answer_data.get("citations", []),
                                "iterations": iteration,
                                "search_history": search_history,
                                "model": self.model
                            }
                        except json.JSONDecodeError:
                            pass
                    
                    elif tool_call.function.name == "search_documents":
                        try:
                            query = json.loads(tool_call.function.arguments)["query"]
                        except (json.JSONDecodeError, KeyError):
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": "Error: Invalid search query format"
                            })
                            continue
                        
                        if verbose:
                            print(f"Searching: {query}")
                        
                        results = self.search_engine.search(query, top_k)
                        
                        if verbose:
                            print(f"Returning {len(results)} results to model")
                        
                        search_history.append({
                            "iteration": iteration,
                            "query": query,
                            "results": results,
                            "num_results": len(results)
                        })
                        
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
                            "content": f"Found {len(results)} matching pages."
                        })
                        
                        if results:
                            image_content: List[Dict[str, Any]] = [
                                {"type": "text", "text": "Here are the matching pages:\n"}
                            ]
                            for result in results:
                                image_content.append({
                                    "type": "text",
                                    "text": f"\nFile: {result['file']}, Page: {result['page_number']}"
                                })
                                try:
                                    image_content.append(
                                        self._load_page_image(result['file'], result['page_number'])
                                    )
                                except Exception as e:
                                    pass  # Silent in async mode
                            
                            messages.append({"role": "user", "content": image_content})
            else:
                text_content = message.content or ""
                if verbose:
                    print(f"No tool call, got text response: {text_content[:200]}...")
                
                parsed = self._parse_answer_from_text(text_content)
                return {
                    "question": question,
                    "answer": parsed["answer"],
                    "citations": parsed["citations"],
                    "iterations": iteration,
                    "search_history": search_history,
                    "model": self.model,
                    "raw_response": text_content
                }
        
        return {
            "question": question,
            "answer": [],
            "citations": [],
            "iterations": max_iterations,
            "search_history": search_history,
            "model": self.model,
            "error": "Max iterations reached"
        }
    
    async def _run_agent_tag_mode(
        self,
        question: str,
        max_iterations: int = 5,
        top_k: int = 3,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Run agent using <question>/<search> tag format (async version)."""
        formatted_question = f"<question>{question}</question>"
        
        system_prompt = """You are a document search assistant. Given a question, think about what to search for and output a search query.

Output format:
<think>your reasoning about what to search</think>
<search>your search query</search>"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_question}
        ]
        
        search_history = []
        
        for iteration in range(1, max_iterations + 1):
            if verbose:
                print(f"\n=== Iteration {iteration} ===")
            
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            except Exception as e:
                if verbose:
                    print(f"API Error on iteration {iteration}: {e}")
                break
            
            message = response.choices[0].message
            text_content = message.content or ""
            
            if verbose:
                print(f"Response: {text_content[:300]}...")
            
            parsed = parse_tag_response(text_content)
            
            if verbose:
                if parsed["think"]:
                    print(f"Think: {parsed['think'][:200]}...")
                if parsed["search"]:
                    print(f"Search: {parsed['search']}")
            
            if parsed["search"]:
                query = parsed["search"]
                
                if verbose:
                    print(f"Executing search: {query}")
                
                results = self.search_engine.search(query, top_k)
                
                if verbose:
                    print(f"Found {len(results)} results")
                
                search_history.append({
                    "iteration": iteration,
                    "query": query,
                    "results": results,
                    "num_results": len(results),
                    "think": parsed["think"]
                })
                
                messages.append({
                    "role": "assistant",
                    "content": text_content
                })
                
                if results:
                    result_text = f"Search results for '{query}':\n"
                    for i, r in enumerate(results, 1):
                        result_text += f"  {i}. {r['file']} page {r['page_number']}\n"
                    
                    return {
                        "question": question,
                        "answer": [],
                        "citations": [],
                        "iterations": iteration,
                        "search_history": search_history,
                        "model": self.model,
                        "tag_mode": True
                    }
                else:
                    messages.append({
                        "role": "user",
                        "content": f"No results found for '{query}'. Please try a different search query.\n\n{formatted_question}"
                    })
            else:
                if verbose:
                    print("No <search> tag found in response")
                
                messages.append({
                    "role": "assistant",
                    "content": text_content
                })
                messages.append({
                    "role": "user", 
                    "content": f"Please output a search query in this format: <search>your query</search>\n\n{formatted_question}"
                })
        
        return {
            "question": question,
            "answer": [],
            "citations": [],
            "iterations": max_iterations,
            "search_history": search_history,
            "model": self.model,
            "tag_mode": True,
            "error": "Max iterations reached"
        }
    
    async def evaluate_question(
        self,
        question: str,
        ground_truth: Dict[str, Any],
        question_id: str,
        max_iterations: int = 5,
        top_k: int = 3
    ) -> QuestionMetrics:
        """Evaluate agent on a single question (async version)."""
        gt_file, gt_page = self._extract_ground_truth(ground_truth)

        metrics = QuestionMetrics(
            question_id=question_id,
            question=question,
            ground_truth=ground_truth
        )

        if self.tag_mode:
            result = await self._run_agent_tag_mode(question, max_iterations, top_k)
        else:
            result = await self._run_agent(question, max_iterations, top_k)

        metrics.final_answer = result.get("answer")
        metrics.final_citations = result.get("citations")
        metrics.total_iterations = result.get("iterations", 0)

        for search in result.get("search_history", []):
            search_metrics = self._compute_search_metrics(
                query=search["query"],
                results=search["results"],
                gt_file=gt_file,
                gt_page=gt_page
            )
            metrics.searches.append(search_metrics)

            if search_metrics.gt_found and metrics.iterations_to_success is None:
                metrics.iterations_to_success = search["iteration"]
                metrics.success = True

        if metrics.searches:
            first_search = metrics.searches[0]
            metrics.first_hit_rank = first_search.gt_rank
            metrics.first_search_mrr = first_search.reciprocal_rank

            best_search = max(metrics.searches, key=lambda s: s.reciprocal_rank)
            metrics.best_rank = best_search.gt_rank
            metrics.best_mrr = best_search.reciprocal_rank

        # Compute ANLS* score
        gold_answers = ground_truth.get("answers", [])
        if gold_answers and metrics.final_answer:
            metrics.anls_score = compute_anls_star(metrics.final_answer, gold_answers)
        
        # Compute Citation F1 metrics
        gt_locations = ground_truth.get("answer_locations", [])
        if gt_locations and metrics.final_citations:
            page_f1_metrics = compute_citation_f1(metrics.final_citations, gt_locations, level='page')
            metrics.page_f1 = page_f1_metrics['f1']
            metrics.page_precision = page_f1_metrics['precision']
            metrics.page_recall = page_f1_metrics['recall']
            
            doc_f1_metrics = compute_citation_f1(metrics.final_citations, gt_locations, level='document')
            metrics.doc_f1 = doc_f1_metrics['f1']
            metrics.doc_precision = doc_f1_metrics['precision']
            metrics.doc_recall = doc_f1_metrics['recall']

        return metrics


async def run_parallel_evaluation(
    evaluator: AsyncSearchEvaluator,
    dataset,
    output_path: Optional[str],
    max_iterations: int,
    top_k: int,
    concurrency: int,
    timeout_per_question: int = 300  # 5 minutes per question
) -> EvaluationResults:
    """Run parallel evaluation with controlled concurrency."""
    
    semaphore = asyncio.Semaphore(concurrency)
    results_lock = asyncio.Lock()
    
    # Collect all metrics
    all_question_metrics: List[QuestionMetrics] = []
    
    async def process_item(idx: int, item: Dict) -> QuestionMetrics:
        """Process a single item with semaphore control and timeout."""
        question_id = item.get("id", f"q_{idx}")
        ground_truth = {
            "answers": item.get("answers"),
            "answer_locations": item.get("answer_locations"),
            "category": item.get("category")
        }
        
        async with semaphore:
            try:
                q_metrics = await asyncio.wait_for(
                    evaluator.evaluate_question(
                        question=item["question"],
                        ground_truth=ground_truth,
                        question_id=question_id,
                        max_iterations=max_iterations,
                        top_k=top_k
                    ),
                    timeout=timeout_per_question
                )
                return q_metrics
            except asyncio.TimeoutError:
                print(f"\nTimeout for question {question_id} after {timeout_per_question}s")
                # Return empty metrics for timed-out question
                return QuestionMetrics(
                    question_id=question_id,
                    question=item["question"],
                    ground_truth=ground_truth
                )
    
    # Create all tasks
    tasks = [
        process_item(idx, item) 
        for idx, item in enumerate(dataset)
    ]
    
    print(f"\nStarting parallel evaluation with concurrency={concurrency}, timeout={timeout_per_question}s per question")
    
    # Run with progress bar and collect results
    for coro in atqdm.as_completed(tasks, desc="Evaluating", total=len(tasks)):
        q_metrics = await coro
        async with results_lock:
            all_question_metrics.append(q_metrics)
    
    # Compute aggregate results
    results = EvaluationResults(model=evaluator.model)
    results.question_metrics = all_question_metrics
    results.total_questions = len(all_question_metrics)
    
    all_iterations = []
    iterations_to_success = []
    first_hit_ranks = []
    first_mrrs = []
    best_ranks = []
    best_mrrs = []
    hit_at_1_count = 0
    hit_at_3_count = 0
    hit_at_5_count = 0
    
    # ANLS* and Citation F1 trackers
    all_anls = []
    all_page_f1 = []
    all_page_precision = []
    all_page_recall = []
    all_doc_f1 = []
    all_doc_precision = []
    all_doc_recall = []
    
    for q_metrics in all_question_metrics:
        all_iterations.append(q_metrics.total_iterations)
        
        if q_metrics.success:
            results.success_count += 1
            if q_metrics.iterations_to_success is not None:
                iterations_to_success.append(q_metrics.iterations_to_success)
        
        if q_metrics.first_hit_rank is not None:
            first_hit_ranks.append(q_metrics.first_hit_rank)
            first_mrrs.append(q_metrics.first_search_mrr)
            
            if q_metrics.first_hit_rank == 1:
                hit_at_1_count += 1
            if q_metrics.first_hit_rank <= 3:
                hit_at_3_count += 1
            if q_metrics.first_hit_rank <= 5:
                hit_at_5_count += 1
        
        if q_metrics.best_rank is not None:
            best_ranks.append(q_metrics.best_rank)
            best_mrrs.append(q_metrics.best_mrr)
        
        # Track ANLS* and Citation F1
        all_anls.append(q_metrics.anls_score)
        all_page_f1.append(q_metrics.page_f1)
        all_page_precision.append(q_metrics.page_precision)
        all_page_recall.append(q_metrics.page_recall)
        all_doc_f1.append(q_metrics.doc_f1)
        all_doc_precision.append(q_metrics.doc_precision)
        all_doc_recall.append(q_metrics.doc_recall)
    
    n = results.total_questions
    
    results.success_rate = results.success_count / n if n > 0 else 0
    results.avg_iterations = sum(all_iterations) / n if n > 0 else 0
    
    if iterations_to_success:
        results.avg_iterations_to_success = sum(iterations_to_success) / len(iterations_to_success)
    
    if first_hit_ranks:
        results.first_hit_rate = len(first_hit_ranks) / n
        results.avg_first_hit_rank = sum(first_hit_ranks) / len(first_hit_ranks)
        results.first_search_mrr = sum(first_mrrs) / len(first_mrrs)
    
    if best_ranks:
        results.avg_best_rank = sum(best_ranks) / len(best_ranks)
        results.best_search_mrr = sum(best_mrrs) / len(best_mrrs)
    
    results.hit_at_1 = hit_at_1_count / n if n > 0 else 0
    results.hit_at_3 = hit_at_3_count / n if n > 0 else 0
    results.hit_at_5 = hit_at_5_count / n if n > 0 else 0
    
    # Compute ANLS* and Citation F1 aggregates
    if all_anls:
        results.avg_anls = sum(all_anls) / len(all_anls)
        results.anls_accuracy = sum(1 for s in all_anls if s >= 0.5) / len(all_anls)
    if all_page_f1:
        results.avg_page_f1 = sum(all_page_f1) / len(all_page_f1)
        results.avg_page_precision = sum(all_page_precision) / len(all_page_precision)
        results.avg_page_recall = sum(all_page_recall) / len(all_page_recall)
    if all_doc_f1:
        results.avg_doc_f1 = sum(all_doc_f1) / len(all_doc_f1)
        results.avg_doc_precision = sum(all_doc_precision) / len(all_doc_precision)
        results.avg_doc_recall = sum(all_doc_recall) / len(all_doc_recall)
    
    return results


def print_single_result(result: Dict[str, Any]):
    """Print result for a single question."""
    print("\n" + "=" * 80)
    print("QUESTION:", result.get("question", "N/A"))
    print("\nANSWER:", json.dumps(result.get("answer", []), indent=2))
    print("\nCITATIONS:", json.dumps(result.get("citations", []), indent=2))
    print("\nMETADATA:")
    print(f"  Model: {result.get('model', 'N/A')}")
    print(f"  Iterations: {result.get('iterations', 0)}")
    print(f"  Searches:")
    for search in result.get("search_history", []):
        print(f"    [{search.get('iteration', '?')}] '{search.get('query', '')}' -> {search.get('num_results', len(search.get('results', [])))} results")
    if "error" in result:
        print(f"  Error: {result['error']}")
    if "raw_response" in result:
        print(f"  (Parsed from text response)")
    print("=" * 80)


def print_results(results: EvaluationResults):
    """Pretty print evaluation results."""
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS: {results.model}")
    print("=" * 70)
    
    print(f"\nTotal Questions: {results.total_questions}")
    
    print(f"\n--- Answer Quality Metrics ---")
    print(f"ANLS* (avg): {results.avg_anls:.4f}")
    print(f"ANLS* Accuracy (≥0.5): {results.anls_accuracy:.1%}")
    
    print(f"\n--- Citation Metrics (Page-level) ---")
    print(f"Page F1: {results.avg_page_f1:.4f}")
    print(f"Page Precision: {results.avg_page_precision:.4f}")
    print(f"Page Recall: {results.avg_page_recall:.4f}")
    
    print(f"\n--- Citation Metrics (Document-level) ---")
    print(f"Doc F1: {results.avg_doc_f1:.4f}")
    print(f"Doc Precision: {results.avg_doc_precision:.4f}")
    print(f"Doc Recall: {results.avg_doc_recall:.4f}")
    
    print(f"\n--- Retrieval Success Metrics ---")
    print(f"Success Rate: {results.success_rate:.1%}")
    print(f"Success Count: {results.success_count}/{results.total_questions}")
    
    print(f"\n--- Iteration Metrics ---")
    print(f"Avg Iterations: {results.avg_iterations:.2f}")
    print(f"Avg Iterations to Success: {results.avg_iterations_to_success:.2f}")
    
    print(f"\n--- First Search Metrics ---")
    print(f"First Hit Rate: {results.first_hit_rate:.1%}")
    print(f"Avg First Hit Rank: {results.avg_first_hit_rank:.2f}")
    print(f"First Search MRR: {results.first_search_mrr:.4f}")
    
    print(f"\n--- Best Search Metrics ---")
    print(f"Avg Best Rank: {results.avg_best_rank:.2f}")
    print(f"Best Search MRR: {results.best_search_mrr:.4f}")
    
    print(f"\n--- Hit Rate @ K (First Search) ---")
    print(f"Hit@1: {results.hit_at_1:.1%}")
    print(f"Hit@3: {results.hit_at_3:.1%}")
    print(f"Hit@5: {results.hit_at_5:.1%}")
    
    print("=" * 70)


def compare_models(results_list: List[EvaluationResults]):
    """Compare multiple model results."""
    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)
    
    headers = ["Metric"] + [r.model[:30] for r in results_list]
    header_line = " | ".join(f"{h:^20}" for h in headers)
    print(header_line)
    print("-" * len(header_line))
    
    metrics = [
        ("Success Rate", lambda r: f"{r.success_rate:.1%}"),
        ("Avg Iterations", lambda r: f"{r.avg_iterations:.2f}"),
        ("Iterations to Success", lambda r: f"{r.avg_iterations_to_success:.2f}"),
        ("First Hit Rate", lambda r: f"{r.first_hit_rate:.1%}"),
        ("First Search MRR", lambda r: f"{r.first_search_mrr:.4f}"),
        ("Best Search MRR", lambda r: f"{r.best_search_mrr:.4f}"),
        ("Hit@1", lambda r: f"{r.hit_at_1:.1%}"),
        ("Hit@3", lambda r: f"{r.hit_at_3:.1%}"),
    ]
    
    for metric_name, metric_fn in metrics:
        row = [metric_name] + [metric_fn(r) for r in results_list]
        row_line = " | ".join(f"{v:^20}" for v in row)
        print(row_line)
    
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate search agent performance or answer single questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Answer a single question (interactive mode)
  python evaluate.py --ocr-file data/ocr_output.jsonl --question "What is the total revenue?"
  
  # Evaluate on full dataset
  python evaluate.py --ocr-file data/ocr_output.jsonl --output results.json
  
  # Evaluate only on test split
  python evaluate.py --ocr-file data/ocr_output.jsonl --include-json ../../splits/ctt_subset_test.json
  
  # Compare multiple model results
  python evaluate.py --compare results_base.json results_finetuned.json
  
  # Evaluate fine-tuned model with tag mode (uses <question>/<search> format)
  python evaluate.py --ocr-file data/ocr_output.jsonl --model my-adapter --tag-mode
"""
    )
    
    # Single question mode
    parser.add_argument("--question", "-q", help="Single question to answer (interactive mode)")
    
    # Model and server settings
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model to evaluate")
    parser.add_argument("--ocr-file", help="Path to OCR JSONL file (required except for --compare)")
    parser.add_argument("--base-url", help="vLLM server URL")
    parser.add_argument("--api-key", help="API key")
    
    # Dataset settings
    parser.add_argument("--dataset", default="agentic-document-ai/agentic-document-ai")
    parser.add_argument("--split", default="train")
    parser.add_argument("--include-json", help="JSON file with question IDs to include (e.g., splits/ctt_subset_test.json)")
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    
    # Agent settings
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output (show search queries)")
    parser.add_argument(
        "--tag-mode", 
        action="store_true",
        help="Use <question>/<search> tag format for fine-tuned models (instead of tool calling)"
    )
    
    # Concurrency settings
    parser.add_argument("--concurrency", type=int, default=8,
                       help="Number of concurrent requests (default: 8)")
    parser.add_argument("--sequential", action="store_true",
                       help="Use sequential (non-parallel) evaluation")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout per question in seconds (default: 300s = 5min)")
    
    # Output settings
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--compare", nargs="+", help="JSON files to compare")
    
    args = parser.parse_args()
    
    # Compare mode - doesn't need ocr-file
    if args.compare:
        results_list = []
        for path in args.compare:
            with open(path, 'r') as f:
                data = json.load(f)
                results = EvaluationResults(
                    model=data.get("model", path),
                    total_questions=data.get("total_questions", 0),
                    success_count=data.get("success_count", 0),
                    success_rate=data.get("success_rate", 0),
                    avg_iterations=data.get("avg_iterations", 0),
                    avg_iterations_to_success=data.get("avg_iterations_to_success", 0),
                    first_hit_rate=data.get("first_hit_rate", 0),
                    avg_first_hit_rank=data.get("avg_first_hit_rank", 0),
                    first_search_mrr=data.get("first_search_mrr", 0),
                    avg_best_rank=data.get("avg_best_rank", 0),
                    best_search_mrr=data.get("best_search_mrr", 0),
                    hit_at_1=data.get("hit_at_1", 0),
                    hit_at_3=data.get("hit_at_3", 0),
                    hit_at_5=data.get("hit_at_5", 0),
                )
                results_list.append(results)
        
        compare_models(results_list)
        return
    
    # All other modes require ocr-file
    if not args.ocr_file:
        parser.error("--ocr-file is required (except for --compare mode)")
    
    # Initialize search engine and evaluator
    search_engine = WhooshSearchEngine(args.ocr_file)
    
    evaluator = SearchEvaluator(
        search_engine=search_engine,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        tag_mode=args.tag_mode
    )
    
    # Single question mode
    if args.question:
        print(f"Model: {args.model}")
        if args.tag_mode:
            print("Mode: tag-mode (<question>/<search> format)")
            result = evaluator._run_agent_tag_mode(
                question=args.question,
                max_iterations=args.max_iterations,
                top_k=args.top_k,
                verbose=args.verbose
            )
        else:
            result = evaluator._run_agent(
                question=args.question,
                max_iterations=args.max_iterations,
                top_k=args.top_k,
                verbose=args.verbose
            )
        
        print_single_result(result)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                # Convert results for JSON serialization
                output_result = dict(result)
                json.dump(output_result, f, indent=2, ensure_ascii=False)
            print(f"\nSaved to {args.output}")
        return
    
    # Dataset evaluation mode
    if args.tag_mode:
        print("Mode: tag-mode (<question>/<search> format for fine-tuned models)")
    print(f"Loading dataset: {args.dataset} ({args.split})")
    dataset = load_dataset(args.dataset, split=args.split)
    print(f"Loaded {len(dataset)} questions")
    
    # Filter to only include questions from specified split file
    if args.include_json:
        include_path = Path(args.include_json)
        if include_path.exists():
            with open(include_path, 'r') as f:
                include_data = json.load(f)
            
            # Extract question IDs to include (same format as trajectory_collector.py)
            if "items" in include_data:
                include_ids = {item["question_id"] for item in include_data["items"]}
            elif isinstance(include_data, list):
                include_ids = {item.get("question_id") or item.get("id") for item in include_data}
            else:
                include_ids = set()
            
            print(f"Loaded {len(include_ids)} question IDs to include from {include_path.name}")
            original_len = len(dataset)
            dataset = dataset.filter(lambda x: x["id"] in include_ids)
            print(f"Filtered to {len(dataset)} questions (from {original_len})")
        else:
            print(f"Warning: include file not found: {args.include_json}")
    
    # Apply limit if specified
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limited to {len(dataset)} questions")
    
    if args.sequential:
        # Use original sequential evaluator
        results = evaluator.evaluate_dataset(
            dataset=dataset,
            max_iterations=args.max_iterations,
            top_k=args.top_k,
            limit=None  # Already applied above
        )
    else:
        # Use parallel async evaluator
        async_evaluator = AsyncSearchEvaluator(
            search_engine=search_engine,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            tag_mode=args.tag_mode
        )
        
        results = asyncio.run(run_parallel_evaluation(
            evaluator=async_evaluator,
            dataset=dataset,
            output_path=args.output,
            max_iterations=args.max_iterations,
            top_k=args.top_k,
            concurrency=args.concurrency,
            timeout_per_question=args.timeout
        ))
    
    print_results(results)
    
    if args.output:
        # Write JSONL format (one JSON object per line for each question)
        # Compatible with analyse_results.py
        with open(args.output, 'w', encoding='utf-8') as f:
            for qm in results.question_metrics:
                entry = {
                    "id": qm.question_id,
                    "question": qm.question,
                    "answer": qm.final_answer or [],
                    "citations": qm.final_citations or [],
                    "iterations": qm.total_iterations,
                    "ground_truth_answers": qm.ground_truth.get("answers", []),
                    "ground_truth_locations": qm.ground_truth.get("answer_locations", []),
                    "category": qm.ground_truth.get("category", "Unknown"),
                    # Retrieval metrics
                    "success": qm.success,
                    "iterations_to_success": qm.iterations_to_success,
                    "first_hit_rank": qm.first_hit_rank,
                    "best_rank": qm.best_rank,
                    # Answer quality metrics
                    "anls_score": qm.anls_score,
                    # Citation metrics
                    "page_f1": qm.page_f1,
                    "page_precision": qm.page_precision,
                    "page_recall": qm.page_recall,
                    "doc_f1": qm.doc_f1,
                    "doc_precision": qm.doc_precision,
                    "doc_recall": qm.doc_recall,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"\nResults saved to: {args.output} (JSONL format, {len(results.question_metrics)} entries)")
        
        # Also save aggregate summary as separate JSON file
        summary_path = args.output.replace('.jsonl', '_summary.json')
        if summary_path == args.output:
            summary_path = args.output + '_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
