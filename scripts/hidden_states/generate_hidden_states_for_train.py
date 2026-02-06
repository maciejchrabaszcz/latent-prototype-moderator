import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(".")
from src.evaluation.get_eval_dataloaders import (
    get_wj_train_dataset_for_router,
)
from src.evaluation.templates import BASE_CHAT_TEMPLATE, CLASSIFIER_PROMPT_TEMPLATE
from src.inference.hidden_states import generate_hidden_states
from src.utils import load_model_for_pred

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Path to the base model",
    )
    parser.add_argument(
        "--adapter_path", type=str, default=None, help="Path to the adapter"
    )
    parser.add_argument(
        "--adapter_weight",
        type=float,
        default=None,
        help="Weight with which adapter will be added into base model weights.",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default="data/wildjailbreak",
        help="Path to wildjailbreak dataset saved to disk",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--add_generation_prompt",
        action="store_true",
        help="Whether to add generation prompt when applying chat template",
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default="wildjailbreak_hidden_states",
        help="Path to save predictions",
    )
    parser.add_argument(
        "--remove_files_if_exists",
        type=bool,
        default=True,
        help="Wheter to remove parquet files if they exists",
    )
    parser.add_argument(
        "--response_dataset",
        action="store_true",
        help="Whether to use only the prompt part of the input",
    )
    parser.add_argument(
        "--calculate_mean_hidden_states",
        action="store_true",
        help="Whether to calculate mean hidden states for each example",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum length of the input sequences. If None, no truncation will be applied.",
    )
    args = parser.parse_args()
    args.save_folder.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_for_pred(
        args.base_model,
        args.adapter_path,
        attn_implementation=None,
    )
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.chat_template is None:
        if args.base_model == "allenai/wildguard":
            tokenizer.chat_template = CLASSIFIER_PROMPT_TEMPLATE
        else:
            tokenizer.chat_template = BASE_CHAT_TEMPLATE
    dataloader = get_wj_train_dataset_for_router(
        args.dataset_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        add_generation_prompt=args.add_generation_prompt,
        prompt_only=not args.response_dataset,
        max_length=args.max_length,
    )
    generate_hidden_states(
        model=model,
        dataloader=dataloader,
        save_folder=args.save_folder,
        remove_files_if_exists=args.remove_files_if_exists,
        calculate_mean_hidden_states=args.calculate_mean_hidden_states,
    )
