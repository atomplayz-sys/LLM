import logging

from datasets import Dataset, load_dataset

from src.constants import SYSTEM_PROMPT

logger = logging.getLogger(__name__)
CHAT_TEMPLATE = "<|system|>\n{system_prompt}</s>\n<|user|>\n{instruction}{input_text}</s>\n<|assistant|>\n{output}</s>"


def format_instruction(sample: dict) -> str:
    input_text = ""
    if sample.get("input") and sample["input"].strip():
        input_text = f"\n\nContext: {sample['input']}"
    return CHAT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        instruction=sample["instruction"],
        input_text=input_text,
        output=sample["output"],
    )


def load_platypus_dataset(max_samples: int | None = None, seed: int = 42) -> "Dataset":
    logger.info("Loading Open-Platypus dataset from HuggingFace...")
    dataset = load_dataset("garage-bAInd/Open-Platypus", split="train")
    logger.info("Loaded %d samples", len(dataset))
    if max_samples and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
        logger.info("Sampled %d examples for training", max_samples)
    dataset = dataset.map(
        lambda sample: {"text": format_instruction(sample)},
        remove_columns=dataset.column_names,
        desc="Formatting dataset",
    )
    logger.info("Dataset ready: %d formatted samples", len(dataset))
    return dataset


def get_dataset_info() -> dict:
    return {
        "name": "Open-Platypus",
        "source": "garage-bAInd/Open-Platypus",
        "description": "Curated dataset of ~25K STEM and logic questions used to train the Platypus family of models.",
        "url": "https://huggingface.co/datasets/garage-bAInd/Open-Platypus",
        "paper": "https://arxiv.org/abs/2308.07317",
    }
