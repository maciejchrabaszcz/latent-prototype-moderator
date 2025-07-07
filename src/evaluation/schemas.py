from typing import Literal

from lmformatenforcer import JsonSchemaParser, RegexParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from pydantic import BaseModel


class MultipleChoiceSchema(BaseModel):
    answer: Literal["A", "B", "C", "D"]


class TrueFalseSchema(BaseModel):
    answer: Literal["0", "1"]


def get_multiple_choice_formatter(tokenizer):
    parser = JsonSchemaParser(MultipleChoiceSchema.model_json_schema())
    prefix_function = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    return prefix_function


def get_true_false_json_formatter(tokenizer):
    parser = JsonSchemaParser(TrueFalseSchema.model_json_schema())
    prefix_function = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    return prefix_function

def get_true_false_formatter(tokenizer):
    parser = RegexParser(r'(0|1)')
    prefix_function = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    return prefix_function