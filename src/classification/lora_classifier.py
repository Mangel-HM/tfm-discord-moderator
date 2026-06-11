from __future__ import annotations

from typing import Any

from src.classification.baseline_classifier import parse_normalized_classification
from src.domain.schemas import (
    ALLOWED_ACTIONS,
    ALLOWED_RISK_LABELS,
    ALLOWED_TOPICS,
    DiscordMessage,
    NormalizedClassification,
)


LORA_SYSTEM_PROMPT = """You are a conservative Discord moderation classifier.
Classify English Discord messages only. Return only valid JSON and no markdown.
Use only the allowed topics, risk labels, and actions listed by the user."""


def build_lora_user_prompt(message: DiscordMessage) -> str:
    topics = ", ".join(ALLOWED_TOPICS)
    risk_labels = ", ".join(ALLOWED_RISK_LABELS)
    actions = ", ".join(ALLOWED_ACTIONS)
    context = "\n".join(f"{index}. {entry}" for index, entry in enumerate(message.context, start=1))
    if not context:
        context = "(no previous channel context)"

    return f"""
Classify the current English Discord message for moderation.

Allowed topics: [{topics}]
Allowed risk_labels: [{risk_labels}]
Allowed actions: [{actions}]

Rules:
- Classify only the current message.
- Use previous channel context only as supporting context.
- If there is no moderation risk, use risk_labels = ["sin_riesgo"] and action = "allow".
- Do not combine sin_riesgo with any other risk label.
- Do not invent labels outside the allowed lists.
- Prefer "review" for risky content unless the message clearly suggests a warning or deletion candidate.
- Return only valid JSON with exactly these fields: "topic", "risk_labels", "action".

Previous channel context:
{context}

Current message to classify:
{message.author_role}: {message.text}
""".strip()


def build_lora_messages(message: DiscordMessage) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": LORA_SYSTEM_PROMPT},
        {"role": "user", "content": build_lora_user_prompt(message)},
    ]


def get_torch_dtype(*, bf16: bool, fp16: bool):
    import torch

    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return "auto"


def load_tokenizer_with_chat_template(*, adapter_dir: str, model_name_or_path: str):
    from transformers import AutoTokenizer

    errors: list[str] = []
    for candidate in (adapter_dir, model_name_or_path):
        try:
            tokenizer: Any = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(tokenizer, "chat_template", None) is None:
            errors.append(f"{candidate}: tokenizer does not define chat_template")
            continue
        return tokenizer
    raise ValueError("Could not load a tokenizer with chat_template. " + " | ".join(errors))


class LoraModerationClassifier:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        adapter_dir: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        bf16: bool = True,
        fp16: bool = False,
    ):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        self.tokenizer = load_tokenizer_with_chat_template(
            adapter_dir=adapter_dir,
            model_name_or_path=model_name_or_path,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            dtype=get_torch_dtype(bf16=bf16, fp16=fp16),
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def classify_message(self, message: DiscordMessage) -> NormalizedClassification:
        raw_output = self.generate(message)
        return parse_normalized_classification(raw_output)

    def generate(self, message: DiscordMessage) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            build_lora_messages(message),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        generated_ids = outputs[:, inputs.input_ids.shape[1] :]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
