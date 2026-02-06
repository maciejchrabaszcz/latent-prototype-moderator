from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from .templates import MMLU_TEMPLATE


def add_classification_template(
    messages: List[Dict[str, str]], prompt_template: str, prompt_only: bool = True
) -> List[Dict[str, str]]:
    if prompt_only:
        messages[0]["content"] = prompt_template.format(prompt=messages[0]["content"])
    else:
        messages[0]["content"] = prompt_template.format(
            prompt=messages[0]["content"], response=messages[1]["content"]
        )
        messages = messages[:1]
    return messages


def get_apply_template_func(
    tokenizer,
    add_generation_prompt: bool = False,
    prompt_template: Optional[str] = None,
):
    def process_dataset(example):
        messages = example["messages"]
        if prompt_template is not None:
            messages = add_classification_template(
                messages, prompt_template, prompt_only=True
            )
        outputs = {
            "input_ids": tokenizer.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt
            )
        }
        return outputs

    return process_dataset


def apply_task_template(example, prompt_template: Optional[str] = None):
    prompt = MMLU_TEMPLATE.format(
        subject=example["config"],
        question=example["question"],
        a=example["choices"][0],
        b=example["choices"][1],
        c=example["choices"][2],
        d=example["choices"][3],
    )
    if prompt_template is not None:
        prompt = prompt_template.format(prompt=prompt)
    output = {"messages": [{"role": "user", "content": prompt}]}
    return output


def get_mmlu_dataloader_for_router(
    mmlu_path: Path,
    tokenizer: PreTrainedTokenizer,
    add_generation_prompt: bool = False,
    return_categories: bool = False,
    prompt_template: Optional[str] = None,
    **dataloader_kwargs,
):
    dataset = load_from_disk(mmlu_path, keep_in_memory=True)
    if return_categories:
        categories = dataset["config"]
    dataset = dataset.map(apply_task_template).remove_columns(dataset.column_names)
    dataset = dataset.map(
        get_apply_template_func(
            tokenizer=tokenizer,
            add_generation_prompt=add_generation_prompt,
            prompt_template=prompt_template,
        )
    ).remove_columns(dataset.column_names)

    def add_zeros(example):
        example["labels"] = 0
        return example

    dataset = dataset.map(add_zeros)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, collate_fn=collator, **dataloader_kwargs)
    if return_categories:
        return dataloader, categories
    return dataloader


def get_wj_processor(
    tokenizer: PreTrainedTokenizer,
    add_generation_prompt: bool = False,
    prompt_template: Optional[str] = None,
):
    def process_wj_examples(example):
        messages = [{"role": "user", "content": example["adversarial"]}]
        if prompt_template is not None:
            messages = add_classification_template(
                messages, prompt_template, prompt_only=True
            )
        return {
            "labels": example["label"],
            "input_ids": tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
            ),
        }

    return process_wj_examples


def get_wj_test_dataloader_for_router(
    tokenizer,
    prompt_template: Optional[str] = None,
    add_generation_prompt: bool = False,
    **dataloader_kwargs,
):
    dataset = load_dataset(
        "allenai/wildjailbreak",
        "eval",
        delimiter="\t",
        keep_default_na=False,
        keep_in_memory=True,
    )["train"]
    dataset = dataset.map(
        get_wj_processor(
            tokenizer,
            prompt_template=prompt_template,
            add_generation_prompt=add_generation_prompt,
        )
    ).remove_columns(dataset.column_names)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return DataLoader(dataset, collate_fn=collator, **dataloader_kwargs)


def get_wj_train_dataset_for_router(
    dataset_path,
    tokenizer,
    prompt_template: Optional[str] = None,
    sample_size: Optional[int] = None,
    seed: Optional[Any] = None,
    add_generation_prompt: bool = False,
    prompt_only: bool = True,
    max_length: Optional[int] = None,
    **dataloader_kwargs,
):
    dataset = load_from_disk(dataset_path, keep_in_memory=True)
    if sample_size is not None:
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(sample_size))

    def process_dataset(example):
        messages = example["messages"]
        if prompt_template is not None:
            messages = add_classification_template(
                messages, prompt_template, prompt_only=prompt_only
            )
        if prompt_only:
            messages = messages[:1]
        outputs = {
            "input_ids": tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                truncation=max_length is not None,
                max_length=max_length,
            )
        }
        outputs["labels"] = example["labels"]
        return outputs

    dataset = dataset.map(process_dataset).remove_columns(
        [x for x in dataset.column_names if x != "labels"]
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=max_length)
    return DataLoader(dataset, collate_fn=collator, **dataloader_kwargs)


def load_response_harmfulness_dataset(
    dataset_path,
    tokenizer,
    prompt_template: Optional[str] = None,
    add_generation_prompt: bool = False,
    **dataloader_kwargs,
):
    dataset = load_from_disk(dataset_path, keep_in_memory=True)

    def process_dataset(example):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
        if prompt_template is not None:
            messages = add_classification_template(
                messages, prompt_template, prompt_only=False
            )
        outputs = {
            "input_ids": tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
            )
        }
        outputs["labels"] = float(example["response_harmfulness"])
        return outputs

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataset = dataset.map(process_dataset).remove_columns(dataset.column_names)

    return DataLoader(dataset, collate_fn=collator, **dataloader_kwargs)


def load_prompt_harmfulness_dataset(
    dataset_path,
    tokenizer,
    prompt_template: Optional[str] = None,
    add_generation_prompt: bool = False,
    **dataloader_kwargs,
):
    dataset = load_from_disk(dataset_path, keep_in_memory=True)

    def process_dataset(example):
        if "messages" not in example:
            messages = [{"role": "user", "content": example["prompt"]}]
        else:
            messages = example["messages"]
        if prompt_template is not None:
            messages = add_classification_template(
                messages, prompt_template, prompt_only=True
            )
        outputs = {
            "input_ids": tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
            )
        }
        outputs["labels"] = float(example["prompt_harmfulness"])
        return outputs

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataset = dataset.map(process_dataset).remove_columns(dataset.column_names)

    return DataLoader(dataset, collate_fn=collator, **dataloader_kwargs)


def load_general_benchmark(
    dataset_path,
    tokenizer,
    add_generation_prompt: bool = False,
    prompt_template: Optional[str] = None,
    **dataloader_kwargs,
):
    dataset = load_from_disk(dataset_path, keep_in_memory=True)

    def process_dataset(example):
        messages = [{"role": "user", "content": example["instruction"]}]
        if prompt_template is not None:
            messages = add_classification_template(
                messages, prompt_template, prompt_only=True
            )
        outputs = {
            "input_ids": tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
            )
        }
        outputs["labels"] = 0.0
        return outputs

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataset = dataset.map(process_dataset).remove_columns(dataset.column_names)

    return DataLoader(dataset, collate_fn=collator, **dataloader_kwargs)
