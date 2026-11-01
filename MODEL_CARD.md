# Model Card: Personal LLM (TinyLlama-1.1B-OpenPlatypus-LoRA)

## Model Overview

- **Base Model**: [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- **Architecture**: 1.1 Billion parameter decoder-only transformer (Llama-2 architecture adapted for compact size)
- **Fine-Tuning Method**: Parameter-Efficient Fine-Tuning (PEFT) via LoRA (Low-Rank Adaptation)
- **Dataset**: [garage-bAInd/Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) (~25,000 STEM and logic question-answer pairs)
- **Target Hardware**: Apple Silicon (M-series / MPS), NVIDIA CUDA, or CPU fallback
- **Precision**: `float32` on Apple Silicon (MPS) for numerical stability; `float16` on CUDA

---

## Architecture & LoRA Configuration

### Base Model Specs
- **Layers**: 22 transformer decoder blocks
- **Hidden Dimension**: 2,048
- **Attention Heads**: 32 query heads, 4 key/value heads (Grouped-Query Attention)
- **Vocabulary Size**: 32,000 tokens (SentencePiece)
- **Max Sequence Length**: 2,048 tokens

### LoRA Hyperparameters
- **Rank ($r$)**: 8
- **Scaling Factor ($\alpha$)**: 16 ($2 \times r$)
- **Dropout**: 0.05
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj` (query, key, value, and output projections in self-attention)
- **Trainable Parameters**: ~1.1M (~0.1% of base model weights)
- **Adapter Storage Size**: ~4.5 MB (vs. ~2.2 GB for full weights)

---

## Tokenizer & Prompt Formatting

The model expects inputs formatted in the ChatML conversation format:

```text
<|system|>
You are a helpful, knowledgeable assistant. Answer questions clearly and accurately.</s>
<|user|>
{instruction}

Context: {optional_context}</s>
<|assistant|>
{output}</s>
```

### Control Tokens
| Token | Role / Meaning |
|---|---|
| `<|system|>` | System instruction boundary |
| `<|user|>` | User prompt boundary |
| `<|assistant|>` | Assistant response boundary |
| `</s>` | Turn termination (EOS) |

> **Security Note**: All user-supplied prompts pass through `src.inference.generate.sanitize_input()` prior to tokenization. This strips control tokens to prevent prompt injection and role confusion.

---

## Training Setup & Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| Epochs | 3 | Sufficient convergence on ~25k samples without catastrophic forgetting |
| Batch Size (per device) | 2 | Fits within 16 GB unified RAM on Apple Silicon |
| Gradient Accumulation | 8 | Effective batch size of 16 ($2 \times 8$) |
| Learning Rate | `2e-4` | Standard LoRA learning rate with cosine decay |
| Weight Decay | `0.01` | L2 regularization |
| Warmup Ratio | `0.03` | Gradual warmup over initial 3% of training steps |
| LR Scheduler | `cosine` | Smooth decay to zero |
| Max Sequence Length | 512 tokens | Captures instruction and context while constraining memory |
| Optimizer | `adamw_torch` (MPS) / `paged_adamw_8bit` (CUDA) | Native PyTorch AdamW for Apple Silicon compatibility |
| Seed | 42 | Full reproducibility for data shuffling and weight initialization |

---

## Intended Use

- **Primary Use**: Personal educational and demonstration project for fine-tuning compact LLMs on domain-specific STEM instruction datasets.
- **Inference Interfaces**: Python library (`LLMGenerator`), CLI script (`scripts/evaluate.py`), and REST API (`FastAPI` backend).
- **Domains**: Elementary STEM explanations, basic code generation, mathematical reasoning, and logical problem solving.

---

## Limitations & Risks

1. **Model Scale (1.1B Parameters)**: Small models have limited world knowledge, frequently hallucinate facts, and struggle with complex multi-step reasoning.
2. **Safety Alignment**: While the base model received lightweight instruction tuning, TinyLlama does not have extensive RLHF (Reinforcement Learning from Human Feedback) or constitutional AI alignment.
3. **Not Suitable for Critical Systems**: Never deploy this model in medical, legal, financial, or automated safety-critical environments.
4. **Context Window Constraint**: Trained with a maximum sequence length of 512 tokens. Prompts exceeding this limit are truncated.

