import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.append(".")
import numpy as np

from src.evaluation.get_eval_dataloaders import (
    get_mmlu_dataloader_for_router,
    get_wj_test_dataloader_for_router,
    load_general_benchmark,
    load_prompt_harmfulness_dataset,
)
from src.evaluation.schemas import get_true_false_formatter
from src.evaluation.templates import BASE_CHAT_TEMPLATE, CLASSIFICATION_TEMPLATE
from src.evaluation.utils import calculate_scores


def load_model(
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3", quantize: bool = False
):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) if quantize else None

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
    return model, tokenizer


def process_model_output(output, tokenizer: AutoTokenizer):
    out = float(tokenizer.decode(output, skip_special_tokens=True).strip())
    return out


def process_dataloader(dataloader, model, tokenizer, true_false_formatter):
    preds = []
    labels = []
    for examples in tqdm(dataloader):
        labels.append(examples.pop("labels"))

        with torch.no_grad():
            examples = examples.to(model.device)
            out = model.generate(
                **examples,
                max_new_tokens=200,
                prefix_allowed_tokens_fn=true_false_formatter,
                pad_token_id=tokenizer.pad_token_id
                if tokenizer.pad_token
                else tokenizer.unk_token_id,
            )
            preds += [
                process_model_output(
                    out[i][examples["input_ids"].shape[1] :], tokenizer=tokenizer
                )
                for i in range(out.shape[0])
            ]
    labels = torch.cat(labels, dim=0).tolist()
    return preds, labels


def main(
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    mmlu_path: str = "datasets/mmlu_redux/",
    harmful_benchmarks_folder: Optional[Path] = None,
    non_harmful_benchmarks_folder: Optional[Path] = None,
    save_folder: Path = Path("base_model_safety_preds"),
    batch_size: int = 8,
    qunatize: bool = False,
    use_base_template: bool = False,
):
    model, tokenizer = load_model(base_model, quantize=qunatize)
    if tokenizer.chat_template is None or use_base_template:
        tokenizer.chat_template = BASE_CHAT_TEMPLATE

    true_false_formatter = get_true_false_formatter(tokenizer)

    mmlu_dataloader = get_mmlu_dataloader_for_router(
        mmlu_path,
        tokenizer,
        prompt_template=CLASSIFICATION_TEMPLATE,
        batch_size=batch_size,
    )
    mmlu_preds, mmlu_labels = process_dataloader(
        mmlu_dataloader,
        model=model,
        tokenizer=tokenizer,
        true_false_formatter=true_false_formatter,
    )
    mmlu_scores = calculate_scores(np.array(mmlu_preds), mmlu_labels)

    with open(save_folder / "mmlu_preds.json", "w", encoding="utf-8") as f:
        json.dump(
            {"preds": mmlu_preds, "labels": mmlu_labels},
            f,
            ensure_ascii=True,
            indent=True,
        )
    with open(save_folder / "mmlu_scores.json", "w", encoding="utf-8") as f:
        json.dump(mmlu_scores, f, ensure_ascii=True, indent=True)

    wj_dataloader = get_wj_test_dataloader_for_router(
        tokenizer, prompt_template=CLASSIFICATION_TEMPLATE, batch_size=batch_size
    )
    wj_preds, wj_labels = process_dataloader(
        wj_dataloader,
        model=model,
        tokenizer=tokenizer,
        true_false_formatter=true_false_formatter,
    )
    wj_scores = calculate_scores(np.array(wj_preds), wj_labels)
    with open(save_folder / "wj_preds.json", "w", encoding="utf-8") as f:
        json.dump(
            {"preds": wj_preds, "labels": wj_labels}, f, ensure_ascii=True, indent=True
        )
    with open(save_folder / "wj_scores.json", "w", encoding="utf-8") as f:
        json.dump(wj_scores, f, ensure_ascii=True, indent=True)

    if harmful_benchmarks_folder is not None:
        for harmful_benchmark in harmful_benchmarks_folder.iterdir():
            print(f"Processing harmful benchmark: {harmful_benchmark}")
            harmfulness_dataloader = load_prompt_harmfulness_dataset(
                harmful_benchmark,
                tokenizer,
                prompt_template=CLASSIFICATION_TEMPLATE,
                batch_size=batch_size,
            )
            save_path = save_folder / "harmful" / harmful_benchmark.name
            save_path.mkdir(parents=True, exist_ok=True)
            harmfulness_preds, harmfulness_labels = process_dataloader(
                harmfulness_dataloader,
                model=model,
                tokenizer=tokenizer,
                true_false_formatter=true_false_formatter,
            )
            harmfulness_scores = calculate_scores(
                np.array(harmfulness_preds), harmfulness_labels
            )

            with open(save_path / "scores.json", "w", encoding="utf-8") as f:
                json.dump(harmfulness_scores, f, ensure_ascii=True, indent=True)
            with open(save_path / "preds.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"preds": harmfulness_preds, "labels": harmfulness_labels},
                    f,
                    ensure_ascii=True,
                    indent=True,
                )
    if non_harmful_benchmarks_folder is not None:
        for non_harmful_benchmark in non_harmful_benchmarks_folder.iterdir():
            print(f"Processing non_harmful benchmark: {non_harmful_benchmark}")
            non_harmfulness_dataloader = load_general_benchmark(
                non_harmful_benchmark,
                tokenizer,
                prompt_template=CLASSIFICATION_TEMPLATE,
                batch_size=batch_size,
            )
            save_path = save_folder / "general_benchmarks" / non_harmful_benchmark.name
            save_path.mkdir(parents=True, exist_ok=True)
            non_harmfulness_preds, non_harmfulness_labels = process_dataloader(
                non_harmfulness_dataloader,
                model=model,
                tokenizer=tokenizer,
                true_false_formatter=true_false_formatter,
            )
            non_harmfulness_scores = calculate_scores(
                np.array(non_harmfulness_preds), non_harmfulness_labels
            )
            with open(save_path / "scores.json", "w", encoding="utf-8") as f:
                json.dump(non_harmfulness_scores, f, ensure_ascii=True, indent=True)
            with open(save_path / "preds.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"preds": non_harmfulness_preds, "labels": non_harmfulness_labels},
                    f,
                    ensure_ascii=True,
                    indent=True,
                )
    print(f"Predictions saved to {save_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model safety.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Base model to use.",
    )
    parser.add_argument(
        "--mmlu_path",
        type=str,
        default="data/mmlu_redux/",
        help="Path to MMLU dataset.",
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default=Path("base_model_safety_preds"),
        help="Folder to save predictions.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size used for generation"
    )
    parser.add_argument(
        "--qunatize_model",
        action="store_true",
        help="Wether to quantize model for generation.",
    )
    parser.add_argument(
        "--use_base_template",
        type=bool,
        default=False,
        help="Use base template for generation.",
    )
    parser.add_argument("--harmful_benchmarks_folder", type=Path, default=None)
    parser.add_argument("--non_harmful_benchmarks_folder", type=Path, default=None)
    args = parser.parse_args()

    args.save_folder.mkdir(parents=True, exist_ok=True)
    main(
        base_model=args.base_model,
        mmlu_path=args.mmlu_path,
        harmful_benchmarks_folder=args.harmful_benchmarks_folder,
        non_harmful_benchmarks_folder=args.non_harmful_benchmarks_folder,
        save_folder=args.save_folder,
        batch_size=args.batch_size,
        qunatize=args.qunatize_model,
        use_base_template=args.use_base_template,
    )
