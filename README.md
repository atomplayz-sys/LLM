# 🧠 Personal LLM — Fine-Tuned on Open-Platypus

A professionally engineered, reproducible, and secure Large Language Model project. Built on top of **TinyLlama-1.1B** and fine-tuned on the [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) dataset using **LoRA (Low-Rank Adaptation)** with a hardened **FastAPI** inference backend.

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)
![Bandit](https://img.shields.io/badge/Security-Bandit%20Passing-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture & Design](#-architecture--design)
- [Directory Structure](#-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Configuration & Environment](#-configuration--environment)
- [Training the Model](#-training-the-model)
- [Running Inference](#-running-inference)
- [API Server & Endpoints](#-api-server--endpoints)
- [Security Hardening](#-security-hardening)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Model Card & Limitations](#-model-card--limitations)
- [License](#-license)

---

## 🌟 Overview

The **Personal LLM** project demonstrates the complete lifecycle of a lightweight, domain-adapted language model:
1. **Data Pipeline**: Automated retrieval and formatting of STEM/logic QA samples into ChatML instruction pairs.
2. **LoRA Fine-Tuning**: Parameter-efficient training with Hugging Face `TRL` (SFTTrainer) and `PEFT` targeting Apple Silicon (MPS), NVIDIA GPUs (CUDA), or CPU.
3. **Prompt Injection Mitigation**: Defense-in-depth sanitization stripping LLM control tokens and evasion patterns.
4. **Hardened REST API**: Production-ready FastAPI service with constant-time API key verification, rate limiting, request body boundaries, and CORS isolation.

---

## ⚡ Quick Start (Run & Chat in 30 Seconds)

```bash
# 1. Clone & enter repository
git clone https://github.com/<your-username>/personal-llm.git
cd personal-llm

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# 3. Start the server
uvicorn backend.main:app --reload
```

Open your browser at **[http://localhost:8000](http://localhost:8000)** (or `http://localhost:8000/chat`) to chat immediately in the Linear-inspired dark workspace!

---

## 🏛️ Architecture & Design

```
                     ┌───────────────────────────────────┐
                     │          User Request             │
                     └─────────────────┬─────────────────┘
                                       │
                         [ HTTP Header / Payload ]
                                       ▼
                     ┌───────────────────────────────────┐
                     │  FastAPI Application              │
                     │  - Body Size Limit Middleware (1MB)│
                     │  - CORS Policy Restriction        │
                     │  - Rate Limiter (SlowAPI)         │
                     │  - Timing-Safe API Key Auth       │
                     └─────────────────┬─────────────────┘
                                       │
                                [ Validated ]
                                       ▼
                     ┌───────────────────────────────────┐
                     │  Prompt Sanitizer Engine          │
                     │  - Strips <|system|>, </s>, etc.  │
                     │  - Blocks Case & Whitespace Hacks │
                     │  - Token Length Cap Enforcement   │
                     └─────────────────┬─────────────────┘
                                       │
                                [ Sanitized ]
                                       ▼
                     ┌───────────────────────────────────┐
                     │  LLM Generator (PyTorch / PEFT)   │
                     │  - TinyLlama-1.1B Base Weights    │
                     │  - Merged LoRA Adapter Layers     │
                     │  - Device: MPS (Mac) / CUDA / CPU │
                     └─────────────────┬─────────────────┘
                                       │
                                [ Generation ]
                                       ▼
                     ┌───────────────────────────────────┐
                     │  Safe Response Formatter          │
                     │  - Genericized Error Handling     │
                     │  - Zero Raw Prompt Reflection     │
                     └───────────────────────────────────┘
```

---

## 📁 Directory Structure

```
personal-llm/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI matrix (3.10, 3.11, 3.12)
├── backend/
│   ├── __init__.py              # Backend package docstring
│   ├── main.py                  # FastAPI server, lifespan, and middlewares
│   ├── routes.py                # Endpoints (/generate, /chat, /health, /model/info)
│   ├── schemas.py               # Pydantic v2 validation models with strict bounds
│   └── static/                  # Linear-inspired Web Workspace
│       ├── app.js               # Client state, conversation store & API client
│       ├── index.html           # Accessible workspace layout
│       └── styles.css           # Monochromatic design system tokens
├── config/
│   └── training_config.yaml     # Model and training hyperparameters
├── scripts/
│   ├── evaluate.py              # CLI evaluation suite across standard benchmark prompts
│   └── train.py                 # Reproducible training entry point
├── src/
│   ├── __init__.py              # Source package definition
│   ├── config.py                # Centralized Settings loaded via python-dotenv
│   ├── constants.py             # Single source of truth for prompts, tokens, and caps
│   ├── data/
│   │   ├── __init__.py          # Exported dataset interfaces
│   │   └── dataset_loader.py    # Open-Platypus preprocessing & ChatML mapping
│   ├── inference/
│   │   ├── __init__.py          # Exported generator interface
│   │   └── generate.py          # LLMGenerator & regex sanitization pipeline
│   ├── model/
│   │   ├── __init__.py          # Model loader exports
│   │   └── model.py             # Base model loading, PEFT config & adapter merge
│   └── training/
│       ├── __init__.py          # Trainer exports
│       └── trainer.py           # SFTTrainer wrapper with deterministic seeding
├── tests/
│   ├── conftest.py              # Shared fixtures & mock TestClient
│   ├── test_api.py              # Endpoint integration & prompt injection tests
│   ├── test_config.py           # Settings integrity & secret masking tests
│   ├── test_dataset.py          # ChatML formatting & dataset schema tests
│   └── test_schemas.py          # Pydantic boundary, range & error tests
├── .env.example                 # Sanitized environment template
├── .gitignore                   # Multi-layer secret & cache exclusions
├── LICENSE                      # MIT Open Source License
├── MODEL_CARD.md                # Detailed ML engineering specifications
├── pyproject.toml               # Modern PEP 517/621 build configuration
└── requirements.txt             # Primary pinned dependencies
```

---

## ⚡ Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- macOS (Apple Silicon M-series recommended) or Linux with NVIDIA GPU

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/personal-llm.git
cd personal-llm

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### 3. Environment Variables Configuration

Copy the example configuration to `.env`:

```bash
cp .env.example .env
```

Review and adjust settings in `.env` as needed:

| Variable | Default | Purpose |
|---|---|---|
| `ENVIRONMENT` | `development` | Set to `production` to disable interactive `/docs` |
| `API_KEY` | *(None)* | Secret token for client authentication (disabled if blank) |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Whitelisted CORS domains |
| `RATE_LIMIT` | `10/minute` | Per-client rate limit window |
| `HOST` | `127.0.0.1` | Local loopback bind (use `0.0.0.0` for Docker) |
| `PORT` | `8000` | Listening port |
| `LOG_LEVEL` | `INFO` | Logging threshold (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 🏋️ Training the Model

### Quick Smoke Test (100 samples, 1 epoch)
Verifies the complete pipeline without waiting hours:

```bash
python scripts/train.py --quick-test
```

### Full Fine-Tuning Run
Trains on the complete Open-Platypus dataset according to `config/training_config.yaml`:

```bash
python scripts/train.py
```

Weights and merged adapter checkpoints are saved to `./output/final/`.

---

## 🤖 Running Inference

### CLI Generator
Test a single query directly from the terminal:

```bash
python -m src.inference.generate --prompt "Explain the concept of recursion in programming."
```

### Automated Capability Evaluation
Run through the benchmark prompt suite:

```bash
python scripts/evaluate.py
```

---

## 🌐 API Server & Endpoints

### Starting the Server

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Linear-Inspired Web Workspace**: Open **[http://localhost:8000/chat](http://localhost:8000/chat)** for the distraction-free dark UI with conversation history, markdown code highlighting, and parameter controls.
- **Interactive API Documentation**: Open **[http://localhost:8000/docs](http://localhost:8000/docs)** for the interactive OpenAPI / Swagger UI.

### Core Endpoints

#### 1. Single Prompt Generation (`POST /api/v1/generate`)
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the difference between supervised and unsupervised learning?",
    "max_new_tokens": 256,
    "temperature": 0.7
  }'
```

#### 2. Multi-Turn Conversation (`POST /api/v1/chat`)
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is Python?"},
      {"role": "assistant", "content": "Python is a high-level, interpreted programming language."},
      {"role": "user", "content": "What are its key features?"}
    ]
  }'
```

#### 3. Health Verification (`GET /api/v1/health`)
```bash
curl http://localhost:8000/api/v1/health
```

---

## 🛡️ Security Hardening

This codebase has undergone 4 rounds of security auditing:

1. **Prompt Injection Sanitization**: Strips ChatML control tokens (`<|system|>`, `<|user|>`, `<|assistant|>`, `</s>`) and handles case evasion and whitespace insertion.
2. **Denial of Service Prevention**:
   - HTTP Request body cap: `1 MB` maximum payload.
   - Pydantic schema validation: Prompts capped at `10,000` characters, chat turns capped at `50`.
   - Token output cap: Maximum `1,024` tokens per generation.
3. **Reflected XSS Defense**: API responses do not echo raw prompt strings back to clients.
4. **Timing Attack Protection**: API key comparison uses `hmac.compare_digest()` for constant-time evaluation.
5. **No Supply-Chain Execution**: `trust_remote_code=True` is prohibited throughout all model loading logic.

---

## 🧪 Testing & Quality Assurance

### Run the Automated Test Suite (42 Tests)

```bash
pytest tests/ -v
```

### Static Analysis & Linting

```bash
# Check code style with Ruff
ruff check .

# Check formatting
ruff format --check .

# Run Bandit security audit
bandit -c pyproject.toml -r src/ backend/
```

---

## 📖 Model Card & Limitations

For comprehensive details on training dynamics, dataset splits, GQA attention architecture, and failure modes, consult [`MODEL_CARD.md`](MODEL_CARD.md).

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
