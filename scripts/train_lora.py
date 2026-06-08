from __future__ import annotations

import argparse
import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

REQUIRED_SUMMARY_KEYS = {
    "model_name_or_path",
    "train_file",
    "output_dir",
    "max_examples",
    "max_steps",
    "num_train_epochs",
    "learning_rate",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "max_seq_length",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "bf16",
    "fp16",
    "gradient_checkpointing",
    "examples_used",
    "trainable_parameters",
    "total_parameters",
    "trainable_percentage",
    "final_train_loss",
    "status",
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def read_sft_jsonl(path: str | Path):
    from datasets import Dataset

    records: list[dict[str, Any]] = []
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {input_path}:{line_number}") from exc
            records.append(validate_sft_record(payload, input_path, line_number))
    return Dataset.from_list(records)


def validate_sft_record(payload: Any, path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"SFT record must be an object at {path}:{line_number}")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"SFT record must contain messages at {path}:{line_number}")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{index}] must be an object at {path}:{line_number}")
        message = cast(dict[str, Any], message)
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"messages[{index}].role is invalid at {path}:{line_number}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"messages[{index}].content is invalid at {path}:{line_number}")
    assistant_messages = [message for message in messages if message["role"] == "assistant"]
    if not assistant_messages:
        raise ValueError(f"SFT record must contain an assistant message at {path}:{line_number}")
    try:
        json.loads(assistant_messages[-1]["content"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"assistant content must be valid JSON at {path}:{line_number}") from exc
    metadata = payload.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata must be an object at {path}:{line_number}")
    return {"messages": messages, "metadata": metadata}


def maybe_limit_dataset(dataset, max_examples: int | None):
    if max_examples is None:
        return dataset
    return dataset.select(range(min(max_examples, len(dataset))))


def build_training_summary(
    *,
    model_name_or_path: str,
    train_file: str | Path,
    output_dir: str | Path,
    max_examples: int | None,
    max_steps: int,
    num_train_epochs: int,
    learning_rate: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    bf16: bool,
    fp16: bool,
    gradient_checkpointing: bool,
    examples_used: int,
    parameter_counts: Mapping[str, int | float | None] | None = None,
    final_train_loss: float | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    counts = dict(parameter_counts or {})
    trainable_parameters = counts.get("trainable_parameters")
    total_parameters = counts.get("total_parameters")
    trainable_percentage = counts.get("trainable_percentage")
    if (
        trainable_percentage is None
        and isinstance(trainable_parameters, int)
        and isinstance(total_parameters, int)
        and total_parameters > 0
    ):
        trainable_percentage = 100 * trainable_parameters / total_parameters

    return {
        "model_name_or_path": model_name_or_path,
        "train_file": str(train_file),
        "output_dir": str(output_dir),
        "max_examples": max_examples,
        "max_steps": max_steps,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "bf16": bf16,
        "fp16": fp16,
        "gradient_checkpointing": gradient_checkpointing,
        "examples_used": examples_used,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_percentage": trainable_percentage,
        "final_train_loss": final_train_loss,
        "status": status,
    }


def get_torch_dtype(*, bf16: bool, fp16: bool):
    import torch

    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return "auto"


def load_tokenizer(model_name_or_path: str):
    from transformers import AutoTokenizer

    tokenizer: Any = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "Tokenizer does not define chat_template; use a Transformers chat model/tokenizer "
            "with a chat template for this smoke test."
        )
    return tokenizer


def load_model(
    *,
    model_name_or_path: str,
    bf16: bool,
    fp16: bool,
    gradient_checkpointing: bool,
):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        dtype=get_torch_dtype(bf16=bf16, fp16=fp16),
        device_map="auto",
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    return model


def build_lora_config(args: argparse.Namespace):
    from peft import LoraConfig

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )


def build_sft_config(args: argparse.Namespace):
    from trl import SFTConfig

    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 1,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "report_to": "none",
        "seed": args.seed,
    }
    parameters = inspect.signature(SFTConfig).parameters
    if "max_length" in parameters:
        kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in parameters:
        kwargs["max_seq_length"] = args.max_seq_length
    else:
        raise RuntimeError("Installed TRL SFTConfig does not expose max_length/max_seq_length")
    return SFTConfig(**kwargs)


def build_sft_trainer(
    *,
    model,
    tokenizer,
    train_dataset,
    args: argparse.Namespace,
):
    from trl import SFTTrainer

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": build_sft_config(args),
        "train_dataset": train_dataset,
        "peft_config": build_lora_config(args),
    }
    parameters = inspect.signature(SFTTrainer).parameters
    if "processing_class" in parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**trainer_kwargs)


def count_parameters(model) -> dict[str, int | float]:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    percentage = 100 * trainable / total if total else 0.0
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percentage": percentage,
    }


def extract_final_train_loss(train_result: Any) -> float | None:
    metrics = getattr(train_result, "metrics", None)
    if isinstance(metrics, dict):
        value = metrics.get("train_loss")
        if isinstance(value, int | float):
            return float(value)
    return None


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = maybe_limit_dataset(read_sft_jsonl(args.train_file), args.max_examples)
    tokenizer = load_tokenizer(args.model_name_or_path)
    model = load_model(
        model_name_or_path=args.model_name_or_path,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=args,
    )
    parameter_counts = count_parameters(trainer.model)
    print(
        "Trainable parameters: "
        f"{parameter_counts['trainable_parameters']} / {parameter_counts['total_parameters']} "
        f"({parameter_counts['trainable_percentage']:.4f}%)"
    )

    train_result = trainer.train()
    final_train_loss = extract_final_train_loss(train_result)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    summary = build_training_summary(
        model_name_or_path=args.model_name_or_path,
        train_file=args.train_file,
        output_dir=output_dir,
        max_examples=args.max_examples,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        examples_used=len(dataset),
        parameter_counts=parameter_counts,
        final_train_loss=final_train_loss,
        status="completed",
    )
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved LoRA adapter and summary to {output_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small LoRA smoke-test adapter.")
    parser.add_argument("--train-file", required=True, help="Input SFT/chat JSONL path.")
    parser.add_argument("--output-dir", required=True, help="Directory for the LoRA adapter.")
    parser.add_argument("--model-name-or-path", required=True, help="Transformers model path/name.")
    parser.add_argument("--max-examples", type=positive_int, default=None)
    parser.add_argument("--max-steps", type=positive_int, default=20)
    parser.add_argument("--num-train-epochs", type=positive_int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=positive_int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=4)
    parser.add_argument("--max-seq-length", type=positive_int, default=1024)
    parser.add_argument("--lora-r", type=positive_int, default=8)
    parser.add_argument("--lora-alpha", type=positive_int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true")
    precision.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--logging-steps", type=positive_int, default=1)
    parser.add_argument("--save-steps", type=positive_int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    return run_training(args)


if __name__ == "__main__":
    main()
