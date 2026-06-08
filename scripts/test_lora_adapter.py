from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from src.domain.schemas import ALLOWED_ACTIONS, ALLOWED_RISK_LABELS, ALLOWED_TOPICS


DEFAULT_MESSAGE = "Please stop posting the same scam link in every channel."


def build_test_messages(message: str) -> list[dict[str, str]]:
    topics = ", ".join(ALLOWED_TOPICS)
    risk_labels = ", ".join(ALLOWED_RISK_LABELS)
    actions = ", ".join(ALLOWED_ACTIONS)
    return [
        {
            "role": "system",
            "content": (
                "You are a conservative Discord moderation classifier. "
                "Return only valid JSON and no markdown."
            ),
        },
        {
            "role": "user",
            "content": f"""
Classify this English message for Discord moderation.

Allowed topics: [{topics}]
Allowed risk_labels: [{risk_labels}]
Allowed actions: [{actions}]

Return only valid JSON with exactly these fields:
- "topic"
- "risk_labels"
- "action"

Message:
{message}
""".strip(),
        },
    ]


def get_torch_dtype(*, bf16: bool, fp16: bool):
    import torch

    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return "auto"


def load_tokenizer(adapter_dir: str):
    from transformers import AutoTokenizer

    tokenizer: Any = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "Tokenizer does not define chat_template; the adapter directory should contain "
            "the tokenizer saved by train_lora.py."
        )
    return tokenizer


def generate_response(args: argparse.Namespace) -> str:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(args.adapter_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        dtype=get_torch_dtype(bf16=args.bf16, fp16=args.fp16),
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    messages = build_test_messages(args.message)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    do_sample = args.temperature > 0
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        temperature=args.temperature if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = outputs[:, inputs.input_ids.shape[1] :]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a LoRA adapter and print one response.")
    parser.add_argument(
        "--adapter-dir", required=True, help="Directory containing the LoRA adapter."
    )
    parser.add_argument(
        "--model-name-or-path", required=True, help="Base Transformers model path/name."
    )
    parser.add_argument("--message", default=DEFAULT_MESSAGE, help="Message to classify.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true")
    precision.add_argument("--fp16", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> str:
    args = build_parser().parse_args(argv)
    response = generate_response(args)
    print(response)
    return response


if __name__ == "__main__":
    main()
