"""
Dataset Loader — Open-Platypus
Handles downloading, preprocessing, and formatting the Open-Platypus dataset
into instruction-following format suitable for fine-tuning.
"""

import logging

from datasets import Dataset, load_dataset

from src.constants import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# Chat template for TinyLlama (ChatML format)
# Uses SYSTEM_PROMPT from constants.py to stay in sync with inference
CHAT_TEMPLATE = """<|system|>
{system_prompt}</s>
<|user|>
{instruction}{input_text}</s>
<|assistant|>
{output}</s>"""


def format_instruction(sample: dict) -> str:
    """
    Format a single Open-Platypus sample into the chat template.

    The Open-Platypus dataset has these fields:
    - instruction: The question or task
    - input: Additional context (often empty)
    - output: The expected answer
    """
    input_text = ""
    if sample.get("input") and sample["input"].strip():
        input_text = f"\n\nContext: {sample['input']}"

    return CHAT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        instruction=sample["instruction"],
        input_text=input_text,
        output=sample["output"],
    )


def load_platypus_dataset(
    max_samples: int | None = None,
    seed: int = 42,
) -> "Dataset":
    """
    Load and preprocess the Open-Platypus dataset.

    Args:
        max_samples: If set, randomly sample this many examples.
        seed: Random seed for reproducibility.

    Returns:
        HuggingFace Dataset with a 'text' column ready for training.
    """
    logger.info("Loading Open-Platypus dataset from HuggingFace...")
    dataset = load_dataset("garage-bAInd/Open-Platypus", split="train")
    logger.info("Loaded %d samples", len(dataset))

    # Optionally limit dataset size (useful for testing)
    if max_samples and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
        logger.info("Sampled %d examples for training", max_samples)

    # Format each sample into the chat template
    dataset = dataset.map(
        lambda sample: {"text": format_instruction(sample)},
        remove_columns=dataset.column_names,
        desc="Formatting dataset",
    )

    logger.info("Dataset ready: %d formatted samples", len(dataset))
    return dataset


def get_dataset_info() -> dict:
    """Return metadata about the dataset for API responses."""
    return {
        "name": "Open-Platypus",
        "source": "garage-bAInd/Open-Platypus",
        "description": (
            "Curated dataset of ~25K STEM and logic questions "
            "used to train the Platypus family of models."
        ),
        "url": "https://huggingface.co/datasets/garage-bAInd/Open-Platypus",
        "paper": "https://arxiv.org/abs/2308.07317",
    }
