import logging
import os

import torch
import yaml
from transformers import TrainingArguments
from trl import SFTTrainer

from src.data.dataset_loader import load_platypus_dataset
from src.model.model import get_device, load_base_model

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/training_config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Training config not found: {config_path}. Expected a YAML file with model, dataset, and training settings."
        )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid config format in {config_path}. Expected a YAML dictionary."
        )
    return config


def create_training_args(config: dict) -> TrainingArguments:
    train_cfg = config.get("training", {})
    output_dir = config.get("output_dir", "./output")
    logging_dir = config.get("logging_dir", "./logs")
    device = config.get("device") or get_device()
    use_fp16 = False
    use_bf16 = False
    if device == "cuda":
        use_fp16 = train_cfg.get("fp16", True)
    optim = "adamw_torch" if device != "cuda" else "paged_adamw_8bit"
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 0.0002),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        fp16=use_fp16,
        bf16=use_bf16,
        logging_dir=logging_dir,
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        report_to="none",
        optim=optim,
        group_by_length=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        use_mps_device=device == "mps",
    )


def train(config_path: str = "config/training_config.yaml"):
    config = load_config(config_path)
    device = config.get("device") or get_device()
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    logger.info("=" * 50)
    logger.info("Personal LLM — Training Pipeline")
    logger.info("Device: %s | Seed: %d", device, seed)
    logger.info("=" * 50)
    max_samples = config.get("dataset", {}).get("max_samples")
    dataset = load_platypus_dataset(max_samples=max_samples)
    model, tokenizer = load_base_model(config, for_training=True)
    training_args = create_training_args(config)
    max_seq_length = config.get("training", {}).get("max_seq_length", 512)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        max_seq_length=max_seq_length,
    )
    logger.info("Starting training...")
    trainer.train()
    output_dir = config.get("output_dir", "./output")
    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Training complete! Model saved to: %s", final_dir)
    return trainer
