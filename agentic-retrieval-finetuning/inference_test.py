#!/usr/bin/env python3
"""
Quick Inference Test with LoRA Adapter

Test the finetuned Qwen3-VL model with adapter weights.
"""

import argparse
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


def load_model_with_adapter(
    base_model: str = "Qwen/Qwen3-VL-8B-Thinking",
    adapter_path: str = "checkpoints/sft/final",
    use_4bit: bool = False
):
    """Load base model with LoRA adapter.
    
    Args:
        base_model: HuggingFace model ID or path
        adapter_path: Path to LoRA adapter weights
        use_4bit: Use 4-bit quantization for lower memory
    
    Returns:
        model, processor
    """
    print(f"Loading base model: {base_model}")
    
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    
    return model, processor


def format_prompt(user_input: str, use_question_tags: bool = True) -> str:
    """Format input as training prompt.
    
    The SFT training data uses:
    - Input: <question>...</question>
    - Output: <think>...</think>\n<search>...</search>
    
    Args:
        user_input: The question or input text
        use_question_tags: Wrap input in <question> tags (matches training format)
    
    Returns:
        Formatted prompt string
    """
    if use_question_tags:
        content = f"<question>{user_input}</question>"
    else:
        content = user_input
    
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def generate_response(
    model,
    processor,
    user_input: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
    use_question_tags: bool = True
) -> str:
    """Generate a response from the model.
    
    Args:
        model: The loaded model
        processor: The processor/tokenizer
        user_input: User's input text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        use_question_tags: Wrap input in <question> tags (matches training format)
    
    Returns:
        Generated response text
    """
    prompt = format_prompt(user_input, use_question_tags)
    
    inputs = processor(text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        )
    
    # Decode only the new tokens
    input_length = inputs["input_ids"].shape[1]
    response = processor.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response


def parse_response(response: str) -> dict:
    """Parse model response into think and search components.
    
    Expected format:
        <think>reasoning here</think>
        <search>query here</search>
    
    Returns:
        dict with 'think', 'search', and 'raw' keys
    """
    import re
    
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


def interactive_mode(model, processor, use_question_tags: bool = True):
    """Run interactive chat session."""
    print("\n" + "=" * 60)
    print("Interactive Mode - Type 'quit' or 'exit' to stop")
    print("Enter questions to get <think> reasoning and <search> queries")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("Question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            response = generate_response(
                model, processor, user_input, 
                use_question_tags=use_question_tags
            )
            parsed = parse_response(response)
            
            print("\n" + "-" * 40)
            if parsed["think"]:
                print(f"Think: {parsed['think'][:500]}..." if len(parsed['think'] or '') > 500 else f"Think: {parsed['think']}")
            if parsed["search"]:
                print(f"\nSearch Query: {parsed['search']}")
            if not parsed["think"] and not parsed["search"]:
                print(f"Raw Output: {response}")
            print("-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Test inference with LoRA adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with a question (uses <question> tags like training data)
  python inference_test.py --adapter checkpoints/sft/final -i "What is the maximum speed of the Bell 407?"
  
  # Interactive mode
  python inference_test.py --adapter checkpoints/sft/final
  
  # Without question tags (raw input)
  python inference_test.py --adapter checkpoints/sft/final --no-question-tags -i "Search for helicopter specs"
"""
    )
    parser.add_argument(
        "--model", 
        default="Qwen/Qwen3-VL-8B-Thinking",
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter", 
        default="checkpoints/sft/final",
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--input", "-i",
        help="Single input to test (otherwise enters interactive mode)"
    )
    parser.add_argument(
        "--4bit", 
        dest="use_4bit",
        action="store_true",
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling (greedy decoding)"
    )
    parser.add_argument(
        "--no-question-tags",
        action="store_true",
        help="Don't wrap input in <question> tags (default: uses tags to match training format)"
    )
    
    args = parser.parse_args()
    
    use_question_tags = not args.no_question_tags
    
    # Load model
    model, processor = load_model_with_adapter(
        base_model=args.model,
        adapter_path=args.adapter,
        use_4bit=args.use_4bit
    )
    
    print(f"\nModel loaded successfully!")
    print(f"  Base: {args.model}")
    print(f"  Adapter: {args.adapter}")
    print(f"  Device: {model.device}")
    print(f"  Question tags: {use_question_tags}")
    
    if args.input:
        # Single inference
        print(f"\n{'=' * 60}")
        print(f"Question: {args.input}")
        print(f"{'=' * 60}")
        
        response = generate_response(
            model, processor, args.input,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=not args.no_sample,
            use_question_tags=use_question_tags
        )
        
        parsed = parse_response(response)
        
        print(f"\n--- Parsed Response ---")
        if parsed["think"]:
            print(f"\n[Think]\n{parsed['think']}")
        if parsed["search"]:
            print(f"\n[Search Query]\n{parsed['search']}")
        if not parsed["think"] and not parsed["search"]:
            print(f"\n[Raw Output]\n{response}")
        print(f"\n{'=' * 60}")
    else:
        # Interactive mode
        interactive_mode(model, processor, use_question_tags)


if __name__ == "__main__":
    main()
