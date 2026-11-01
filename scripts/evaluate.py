"""
Evaluation Script

Runs the fine-tuned model on a set of test prompts and prints the results.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model-path ./output/final
"""

import argparse
import logging

from src.config import settings
from src.inference.generate import LLMGenerator

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Test prompts covering different capabilities
TEST_PROMPTS = [
    "Explain the concept of recursion in programming with a simple example.",
    "What is the difference between machine learning and deep learning?",
    "Solve: If a train travels at 60 km/h for 2.5 hours, how far does it go?",
    "What are the main causes of climate change?",
    "Write a Python function that checks if a number is prime.",
    "Explain the theory of relativity in simple terms.",
    "What is the time complexity of binary search and why?",
    "Describe the process of photosynthesis.",
]


def evaluate(model_path: str, base_model: str):
    """Run evaluation on test prompts."""
    print("=" * 70)
    print("📊 Personal LLM — Evaluation")
    print("=" * 70)

    generator = LLMGenerator(base_model, model_path)

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'─' * 70}")
        print(f"📝 [{i}/{len(TEST_PROMPTS)}] {prompt}")
        print(f"{'─' * 70}")

        response = generator.generate(
            prompt,
            max_new_tokens=256,
            temperature=0.7,
        )
        print(f"🤖 {response}")

    print(f"\n{'=' * 70}")
    print("✅ Evaluation complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned LLM")
    parser.add_argument(
        "--model-path",
        type=str,
        default=settings.ADAPTER_PATH,
        help="Path to fine-tuned model adapters",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=settings.BASE_MODEL,
        help="Base model name",
    )

    args = parser.parse_args()
    evaluate(args.model_path, args.base_model)
