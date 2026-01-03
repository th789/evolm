#https://github.com/huggingface/transformers/blob/v4.48-release/src/transformers/modeling_utils.py

from typing import Dict, Callable


from .flash_attention import flash_attention_forward
from .flex_attention import flex_attention_forward
from .sdpa_attention import sdpa_attention_forward

ALL_ATTENTION_FUNCTIONS: Dict[str, Dict[str, Callable]] = {}

ALL_ATTENTION_FUNCTIONS.update(
    {
        "flash_attention_2": flash_attention_forward,
        "flex_attention": flex_attention_forward,
        "sdpa": sdpa_attention_forward,
    }
)