import json
import sys
from argparse import ArgumentParser

sys.path.append(".")
from pathlib import Path

from activation_steering import MalleableModel, SteeringVector
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_chat_template(tokenizer, messages):
    if tokenizer.chat_template:
        return [
            tokenizer.apply_chat_template(
                messages[:1], tokenize=False, add_generation_prompt=True
            )
            for messages in messages
        ]
    return messages


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Path to the base model",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default="../data/processed_benchmarks/harmfulness/prompt/wildguardtest",
        help="Path to dataset saved to disk",
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
        "--steering_vector_path",
        type=Path,
        default="activation-steering/behavior_vector.pkl",
        help="Path to the steering vector",
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default="responses",
        help="Path to folder at which responses will be saved",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default="mistral_steered_responses_wg_test.json",
        help="Path to save responses",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum length of the input sequences. If None, no truncation will be applied.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--system_prompt_file",
        type=Path,
        default=None,
        help="Path to a file containing the system prompt.",
    )
    parser.add_argument(
        "--num_prompts_to_generate",
        type=int,
        default=None,
        help="Number of prompts to generate responses for. If None, generate for all prompts in the dataset.",
    )
    args = parser.parse_args()
    args.save_folder.mkdir(parents=True, exist_ok=True)

    # Initialize vLLM
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype="auto",
        # attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    steering_vector = SteeringVector.load(str(args.steering_vector_path))
    llm = MalleableModel(model, tokenizer=tokenizer)
    llm.steer(
        behavior_vector=steering_vector,
        behavior_layer_ids=[15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        if any(x in args.base_model.lower() for x in ["mistral", "llama"])
        else [18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        behavior_vector_strength=1.2,
    )
    if args.system_prompt_file is not None:
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    else:
        system_prompt = None

    # Load dataset using Hugging Face's load_from_disk
    dataset = load_from_disk(args.dataset_path)
    if "messages" not in dataset.column_names:
        messages = [[{"role": "user", "content": x}] for x in dataset["prompt"]]
    else:
        messages = dataset["messages"]
    generation_settings = {
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "repetition_penalty": 1.1,
        "top_p": 0.95,
        "temperature": 0.3,
        "top_k": 50,
    }
    responses = []
    prompts = [
        tokenizer.apply_chat_template(
            m,
            tokenize=False,
            add_generation_prompt=args.add_generation_prompt,
            enable_thinking=False,
        )
        for m in messages
    ]
    if args.num_prompts_to_generate is not None:
        num_prompts_to_generate = min(args.num_prompts_to_generate, len(prompts))
    else:
        num_prompts_to_generate = len(prompts)
    for i in tqdm(range(0, num_prompts_to_generate, args.batch_size)):
        batch_prompts = prompts[i : i + args.batch_size]

        batch_responses = llm.respond_batch_sequential(
            prompts=batch_prompts,
            settings=generation_settings,
            use_chat_template=False,
        )
        responses.extend(batch_responses)

    # Save responses
    with open(args.save_folder / args.output_file, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=True, indent=4)

    print(f"Predictions saved to {args.output_file}")
