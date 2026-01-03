#pasted from https://github.com/huggingface/transformers/blob/v4.48-release/src/transformers/utils/generic.py

from typing import Optional, TypedDict

class LossKwargs(TypedDict, total=False):
    """
    Keyword arguments to be passed to the loss function

    Attributes:
        num_items_in_batch (`int`, *optional*):
            Number of items in the batch. It is recommended to pass it when
            you are doing gradient accumulation.
    """

    num_items_in_batch: Optional[int]