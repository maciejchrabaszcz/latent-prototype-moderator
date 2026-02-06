from typing import Optional

import torch
from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_for_pred(
    base_model: str,
    adapter_path: Optional[str] = None,
    merge_and_unload: bool = True,
    device_map: Optional[str] = "auto",
    use_fast_tokenizer: bool = True,
    adapter_weight: Optional[float] = None,
    attn_implementation: str = "flash_attention_2",
    pad_token: str = "<unk>",
):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=use_fast_tokenizer,
        padding_side="left",
        truncation_side="right",
    )
    if adapter_path is not None:
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            tokenizer.pad_token_id = len(tokenizer) - 1
            model.resize_token_embeddings(len(tokenizer))
        model = PeftModelForCausalLM.from_pretrained(
            model,
            model_id=adapter_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        if adapter_weight is not None:
            model.add_weighted_adapter(
                adapters=[model.active_adapter],
                weights=[adapter_weight],
                combination_type="linear",
                adapter_name="scaled",
            )
            model.set_adapter("scaled")
        if merge_and_unload:
            model = model.merge_and_unload()
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            tokenizer.pad_token_id = len(tokenizer) - 1
            model.resize_token_embeddings(len(tokenizer))
        else:
            tokenizer.pad_token = pad_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
    return model, tokenizer
