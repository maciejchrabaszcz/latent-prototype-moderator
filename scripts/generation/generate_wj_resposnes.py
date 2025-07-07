import json
import math
from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_and_convert_eval_data(
    conversational=True, tokenizer=None, add_generation_prompt: bool = False
):
    data = load_dataset(
        "allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False
    )["train"]
    data = data["adversarial"]
    if conversational:
        data = [[{"role": "user", "content": x}] for x in data]
        data = tokenizer.apply_chat_template(
            data, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    return data


def generate_predictions(
    model: LLM,
    sampling_params: SamplingParams,
    data,
    batch_size: int = 1,
):
    predictions = []

    num_batches = math.ceil(len(data) / batch_size)
    for i in tqdm(range(num_batches), desc="Processing"):
        prompts = data[i * batch_size : (i + 1) * batch_size]
        outputs = model.generate(
            prompts,
            sampling_params=sampling_params,
        )
        for output in outputs:
            prompt = output.prompt
            predictions.append({"prompt": prompt, "response": output.outputs[0].text})
    return predictions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base_model", type=str, required=True, help="Path to the base model"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate",
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
        "--output_file",
        type=Path,
        default="predictions.json",
        help="Path to save predictions",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="left",
    )
    model = LLM(model=args.base_model)
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens)
    eval_data = load_and_convert_eval_data(
        add_generation_prompt=args.add_generation_prompt, tokenizer=tokenizer
    )

    predictions = generate_predictions(
        model=model,
        sampling_params=sampling_params,
        data=eval_data,
        batch_size=args.batch_size,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=True, indent=4)

    print(f"Predictions saved to {args.output_file}")
