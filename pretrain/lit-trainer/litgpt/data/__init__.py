# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from litgpt.data.base import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.data.alpaca import Alpaca
from litgpt.data.alpaca_2k import Alpaca2k
from litgpt.data.alpaca_gpt4 import AlpacaGPT4
from litgpt.data.json_data import JSON
from litgpt.data.deita import Deita
from litgpt.data.dolly import Dolly
from litgpt.data.flan import FLAN
from litgpt.data.lima import LIMA
from litgpt.data.lit_data import LitData
from litgpt.data.longform import LongForm
from litgpt.data.text_files import TextFiles
from litgpt.data.tinyllama import TinyLlama
from litgpt.data.tinystories import TinyStories
from litgpt.data.openwebtext import OpenWebText
from litgpt.data.microllama import MicroLlama
from litgpt.data.fineweb import FineWeb
from litgpt.data.finefineweb import FineFineWeb
from litgpt.data.finemath import FineMath
from litgpt.data.codepretrain import CodePretrain
from litgpt.data.stackedu import StackEdu
from litgpt.data.openwebmath import OpenWebMath
from litgpt.data.mixedfw1_6fm48_4 import MixedFW1_6FM48_4
from litgpt.data.mixedfw8fm24st18 import MixedFW8FM24ST18
from litgpt.data.mixedfw8fm30cp12 import MixedFW8FM30CP12
from litgpt.data.mixedfw8fm30st12 import MixedFW8FM30ST12
from litgpt.data.mixedfw8fm36cp6 import MixedFW8FM36CP6
from litgpt.data.mixedfw8fm36st6 import MixedFW8FM36ST6
from litgpt.data.mixedfw8fm42 import MixedFW8FM42
from litgpt.data.mixedfw8fm32 import MixedFW8FM32
from litgpt.data.mixedfw8fm22 import MixedFW8FM22
from litgpt.data.mixedfw8fm12 import MixedFW8FM12
from litgpt.data.mixedfw8fm2 import MixedFW8FM2
from litgpt.data.mixedfw16fm34 import MixedFW16FM34


__all__ = [
    "Alpaca",
    "Alpaca2k",
    "AlpacaGPT4",
    "Deita",
    "Dolly",
    "FLAN",
    "JSON",
    "LIMA",
    "LitData",
    "DataModule",
    "LongForm",
    "OpenWebText",
    "SFTDataset",
    "TextFiles",
    "TinyLlama",
    "TinyStories",
    "MicroLlama",
    "get_sft_collate_fn",
    "FineWeb",
    "FineFineWeb",
    "FineMath",
    "CodePretrain",
    "StackEdu",
    "OpenWebMath",
    "MixedFW1_6FM48_4",
    "MixedFW8FM24ST18",
    "MixedFW8FM30CP12",
    "MixedFW8FM30ST12",
    "MixedFW8FM36CP6",
    "MixedFW8FM36ST6",
    "MixedFW8FM42",
    "MixedFW8FM32",
    "MixedFW8FM22",
    "MixedFW8FM12",
    "MixedFW8FM2",
    "MixedFW16FM34",
]
