"""
Model Setup — TinyLlama + LoRA
Handles loading the base model, applying LoRA adapters,
and merging adapters for inference.

Compatible with: Apple Silicon (MPS), NVIDIA (CUDA), and CPU.
"""

import logging
from pathlib import Path

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Auto-detect the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_dtype(device: str) -> torch.dtype:
    """Get the best dtype for the given device."""
    if device == "cuda":
        return torch.float16
    # MPS and CPU — use float32 for stability (float16 can be unstable on MPS)
    return torch.float32


def create_lora_config(config: dict) -> LoraConfig:
    """Create LoRA configuration from the training config."""
    lora_cfg = config.get("lora", {})

    return LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load tokenizer, preferring local cache to prevent network delays and retries."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_causal_lm(model_name: str, dtype: torch.dtype) -> AutoModelForCausalLM:
    """Load causal LM, preferring local cache to prevent network delays."""
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            local_files_only=True,
        )
    except Exception:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )


def load_base_model(
    config: dict,
    for_training: bool = True,
):
    """
    Load the base model and tokenizer.

    Args:
        config: Training configuration dictionary.
        for_training: If True, load with LoRA adapters for training.
                      If False, load for inference.

    Returns:
        Tuple of (model, tokenizer).
    """
    model_name = config.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    device = config.get("device") or get_device()
    dtype = get_dtype(device)

    logger.info("Loading base model: %s", model_name)
    logger.info("Device: %s | Dtype: %s", device, dtype)

    # Load tokenizer (uses local cache if available)
    tokenizer = load_tokenizer(model_name)

    if for_training:
        # Load model in float32 for MPS compatibility
        model = load_causal_lm(model_name, dtype)

        # Move to device (MPS/CUDA/CPU)
        model = model.to(device)

        # Enable gradient checkpointing to save memory
        model.gradient_checkpointing_enable()

        # Apply LoRA
        lora_config = create_lora_config(config)
        model = get_peft_model(model, lora_config)

        # Print trainable parameters info
        trainable, total = model.get_nb_trainable_parameters()
        logger.info(
            "Model loaded with LoRA — trainable: %s / %s (%.2f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100 * trainable / total,
        )
    else:
        # Load for inference
        model = load_causal_lm(model_name, dtype)
        model = model.to(device)
        logger.info("Base model loaded for inference on %s", device)

    return model, tokenizer


def load_finetuned_model(
    base_model_name: str,
    adapter_path: str,
):
    """
    Load a fine-tuned model by combining the base model with saved LoRA adapters.

    Args:
        base_model_name: HuggingFace model ID for the base model.
        adapter_path: Path to the saved LoRA adapter directory.

    Returns:
        Tuple of (model, tokenizer).
    """
    device = get_device()
    dtype = get_dtype(device)

    logger.info("Loading fine-tuned model from %s", adapter_path)
    logger.info("Device: %s | Dtype: %s", device, dtype)

    tokenizer = load_tokenizer(base_model_name)

    # Check if adapter path exists; if not, gracefully fall back to base model
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        logger.warning(
            "Adapter directory not found at '%s'. "
            "Running with base model '%s' directly (run 'python scripts/train.py' to train adapters).",
            adapter_path,
            base_model_name,
        )
        model = load_causal_lm(base_model_name, dtype).to(device)
        return model, tokenizer

    # Load base model
    model = load_causal_lm(base_model_name, dtype)

    # Load and merge LoRA adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    # Move to device
    model = model.to(device)

    logger.info("Fine-tuned model loaded and adapters merged on %s", device)
    return model, tokenizer
