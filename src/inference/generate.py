import logging
import re

import torch

from src.constants import CONTROL_TOKENS, MAX_GENERATION_TOKENS, SYSTEM_PROMPT
from src.model.model import load_finetuned_model

logger = logging.getLogger(__name__)
CONTROL_TOKEN_PATTERNS = [
    re.compile(re.escape(token), re.IGNORECASE) for token in CONTROL_TOKENS
]


def sanitize_input(text: str) -> str:
    sanitized = text
    for pattern in CONTROL_TOKEN_PATTERNS:
        sanitized = pattern.sub("", sanitized)
    sanitized = re.sub(
        "<\\s*\\|\\s*system\\s*\\|\\s*>", "", sanitized, flags=re.IGNORECASE
    )
    sanitized = re.sub(
        "<\\s*\\|\\s*user\\s*\\|\\s*>", "", sanitized, flags=re.IGNORECASE
    )
    sanitized = re.sub(
        "<\\s*\\|\\s*assistant\\s*\\|\\s*>", "", sanitized, flags=re.IGNORECASE
    )
    sanitized = re.sub("<\\s*/\\s*s\\s*>", "", sanitized, flags=re.IGNORECASE)
    return sanitized.strip()


PROMPT_TEMPLATE = (
    "<|system|>\n{system_prompt}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
)


class LLMGenerator:
    def __init__(
        self,
        base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        adapter_path: str = "./output/final",
    ):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        self.model, self.tokenizer = load_finetuned_model(
            self.base_model_name, self.adapter_path
        )
        self.model.eval()
        self._loaded = True

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.15,
    ) -> str:
        self.load()
        safe_prompt = sanitize_input(prompt)
        max_new_tokens = min(max_new_tokens, MAX_GENERATION_TOKENS)
        formatted_prompt = PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT, prompt=safe_prompt
        )
        inputs = self.tokenizer(
            formatted_prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        for token in ["<|user|>", "<|assistant|>", "<|system|>"]:
            response = response.split(token)[0]
        return response.strip()

    def chat(self, messages: list[dict], **kwargs) -> str:
        self.load()
        prompt_parts = [f"<|system|>\n{SYSTEM_PROMPT}</s>"]
        for msg in messages:
            role = msg["role"]
            content = sanitize_input(msg["content"])
            if role == "user":
                prompt_parts.append(f"<|user|>\n{content}</s>")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}</s>")
        prompt_parts.append("<|assistant|>\n")
        full_prompt = "\n".join(prompt_parts)
        capped_tokens = min(kwargs.get("max_new_tokens", 512), MAX_GENERATION_TOKENS)
        inputs = self.tokenizer(
            full_prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=capped_tokens,
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                repetition_penalty=kwargs.get("repetition_penalty", 1.15),
                do_sample=kwargs.get("temperature", 0.7) > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        for token in ["<|user|>", "<|assistant|>", "<|system|>"]:
            response = response.split(token)[0]
        return response.strip()

    def get_model_info(self) -> dict:
        return {
            "base_model": self.base_model_name,
            "adapter_path": self.adapter_path,
            "loaded": self._loaded,
            "device": str(self.model.device) if self._loaded else "not loaded",
            "parameters": sum(p.numel() for p in self.model.parameters())
            if self._loaded
            else None,
        }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Generate text with Personal LLM")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./output/final",
        help="Path to fine-tuned adapter",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    args = parser.parse_args()
    generator = LLMGenerator(args.base_model, args.model_path)
    response = generator.generate(
        args.prompt, max_new_tokens=args.max_tokens, temperature=args.temperature
    )
    print("\n" + "=" * 60)
    print("📝 Prompt:", args.prompt)
    print("=" * 60)
    print("🤖 Response:", response)
    print("=" * 60)
