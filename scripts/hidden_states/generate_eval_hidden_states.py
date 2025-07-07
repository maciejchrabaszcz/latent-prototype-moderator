import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.append(".")

from src.evaluation.get_eval_dataloaders import (
    get_mmlu_dataloader_for_router,
    get_wj_test_dataloader_for_router,
    load_general_benchmark,
    load_prompt_harmfulness_dataset,
)
from src.evaluation.templates import BASE_CHAT_TEMPLATE, CLASSIFIER_PROMPT_TEMPLATE
from src.inference.hidden_states import generate_hidden_states
from src.utils import load_model_for_pred


def main(
    mmlu_path: Path,
    base_model: str,
    batch_size: int,
    save_folder: Path,
    harmful_benchmarks_folder: Optional[Path] = None,
    non_harmful_benchmarks_folder: Optional[Path] = None,
    add_generation_prompt: bool = False,
    remove_files_if_exists: bool = True,
):
    model, tokenizer = load_model_for_pred(
        base_model=base_model, attn_implementation=None
    )
    if tokenizer.chat_template is None:
        if base_model == "allenai/wildguard":
            tokenizer.chat_template = CLASSIFIER_PROMPT_TEMPLATE
        else:
            tokenizer.chat_template = BASE_CHAT_TEMPLATE

    wj_dataloader = get_wj_test_dataloader_for_router(
        tokenizer, batch_size=batch_size, add_generation_prompt=add_generation_prompt
    )

    save_path = save_folder / "harmful/wildjailbreak_test"
    (save_path).mkdir(parents=True, exist_ok=True)
    generate_hidden_states(
        model,
        wj_dataloader,
        save_folder=save_path,
        remove_files_if_exists=remove_files_if_exists,
    )

    mmlu_dataloader = get_mmlu_dataloader_for_router(
        mmlu_path=mmlu_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        add_generation_prompt=add_generation_prompt,
    )
    save_path = save_folder / "general_benchmarks/mmlu"
    (save_path).mkdir(parents=True, exist_ok=True)
    generate_hidden_states(
        model,
        mmlu_dataloader,
        save_folder=save_path,
        remove_files_if_exists=remove_files_if_exists,
    )
    if harmful_benchmarks_folder is not None:
        for harmful_benchmark in harmful_benchmarks_folder.iterdir():
            print(f"Processing harmful benchmark: {harmful_benchmark}")
            dataset = load_prompt_harmfulness_dataset(
                harmful_benchmark,
                tokenizer=tokenizer,
                add_generation_prompt=add_generation_prompt,
                batch_size=batch_size,
            )
            save_path = save_folder / "harmful" / harmful_benchmark.name
            (save_path).mkdir(parents=True, exist_ok=True)
            generate_hidden_states(
                model,
                dataset,
                save_folder=save_path,
                remove_files_if_exists=remove_files_if_exists,
            )
    if non_harmful_benchmarks_folder is not None:
        for non_harmful_benchmark in non_harmful_benchmarks_folder.iterdir():
            print(f"Processing non harmful benchmark: {non_harmful_benchmark}")
            dataset = load_general_benchmark(
                non_harmful_benchmark,
                tokenizer=tokenizer,
                add_generation_prompt=add_generation_prompt,
                batch_size=batch_size,
            )
            save_path = save_folder / "general_benchmarks" / non_harmful_benchmark.name
            (save_path).mkdir(parents=True, exist_ok=True)
            generate_hidden_states(
                model,
                dataset,
                save_folder=save_path,
                remove_files_if_exists=remove_files_if_exists,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation hidden states.")
    parser.add_argument(
        "--mmlu_path",
        type=Path,
        default=Path("data/mmlu_redux"),
        help="Path to the MMLU dataset.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Base model to use.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for dataloaders."
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default=Path("hidden_states"),
        help="Folder to save hidden states.",
    )
    parser.add_argument(
        "--add_generation_prompt",
        action="store_true",
        help="Whether to add generation prompt when applying chat template",
    )
    parser.add_argument("--harmful_benchmarks_folder", type=Path, default=None)
    parser.add_argument("--non_harmful_benchmarks_folder", type=Path, default=None)
    parser.add_argument("--remove_files_if_exists", type=bool, default=True)
    args = parser.parse_args()

    main(
        mmlu_path=args.mmlu_path,
        base_model=args.base_model,
        batch_size=args.batch_size,
        save_folder=args.save_folder,
        harmful_benchmarks_folder=args.harmful_benchmarks_folder,
        non_harmful_benchmarks_folder=args.non_harmful_benchmarks_folder,
        add_generation_prompt=args.add_generation_prompt,
        remove_files_if_exists=args.remove_files_if_exists,
    )
