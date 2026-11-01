import logging
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.float16
    return torch.float32


def create_lora_config(config: dict) -> LoraConfig:
    lora_cfg = config.get("lora", {})
    return LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_tokenizer(model_name: str) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_causal_lm(model_name: str, dtype: torch.dtype) -> AutoModelForCausalLM:
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, local_files_only=True
        )
    except Exception:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)


def load_base_model(config: dict, for_training: bool = True):
    model_name = config.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    device = config.get("device") or get_device()
    dtype = get_dtype(device)
    logger.info("Loading base model: %s", model_name)
    logger.info("Device: %s | Dtype: %s", device, dtype)
    tokenizer = load_tokenizer(model_name)
    if for_training:
        model = load_causal_lm(model_name, dtype)
        model = model.to(device)
        model.gradient_checkpointing_enable()
        lora_config = create_lora_config(config)
        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()
        logger.info(
            "Model loaded with LoRA — trainable: %s / %s (%.2f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100 * trainable / total,
        )
    else:
        model = load_causal_lm(model_name, dtype)
        model = model.to(device)
        logger.info("Base model loaded for inference on %s", device)
    return (model, tokenizer)


def load_finetuned_model(base_model_name: str, adapter_path: str):
    device = get_device()
    dtype = get_dtype(device)
    logger.info("Loading fine-tuned model from %s", adapter_path)
    logger.info("Device: %s | Dtype: %s", device, dtype)
    tokenizer = load_tokenizer(base_model_name)
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        logger.warning(
            "Adapter directory not found at '%s'. Running with base model '%s' directly (run 'python scripts/train.py' to train adapters).",
            adapter_path,
            base_model_name,
        )
        model = load_causal_lm(base_model_name, dtype).to(device)
        return (model, tokenizer)
    model = load_causal_lm(base_model_name, dtype)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model = model.to(device)
    logger.info("Fine-tuned model loaded and adapters merged on %s", device)
    return (model, tokenizer)
