"""
Shared Constants — Single source of truth for values used across training and inference.

This module centralizes constants that are shared between the training pipeline
(dataset formatting) and the inference pipeline (prompt construction). Keeping
them in one place prevents training/inference mismatch bugs.
"""

# ─── System Prompt ────────────────────────────────────────────────
# Used in both training data formatting and inference prompt templates.
# Changing this AFTER training means the model sees a different prompt
# at inference time than it was trained on — only change before retraining.

SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable assistant. "
    "Answer questions clearly and accurately."
)

# ─── Chat Format Tokens ──────────────────────────────────────────
# TinyLlama uses a ChatML-like format with these special tokens.
# These are also used by the sanitizer to strip injected tokens from user input.

CONTROL_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>", "</s>"]

# ─── Generation Limits ───────────────────────────────────────────
# Hard cap on token generation — enforced regardless of caller (API, CLI, direct).
# Prevents GPU/memory exhaustion from malicious or accidental large requests.

MAX_GENERATION_TOKENS = 1024
