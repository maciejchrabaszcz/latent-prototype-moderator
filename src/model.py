from peft import PeftConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


def load_model(
    model_id: str,
    chat_template: str = None,
    attn_implementation: str = "flash_attention_2",
):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation=attn_implementation
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
    if chat_template:
        tokenizer.chat_template = chat_template
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def add_adapter(model: PreTrainedModel, peft_config: PeftConfig):
    return get_peft_model(model, peft_config)
