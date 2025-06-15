# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from torch.utils.data import DataLoader

from litgpt.tokenizer import Tokenizer
from litgpt.data import DataModule


@dataclass
class MixedFW8FM22(DataModule):
    """The MixedFW8FM22 data module is composed of a mix of data.

    Provides training and validation streaming dataloaders that return batches of tokens.
    """
    data_path: Union[str, Path] = Path("/path/to")
    val_split_fraction: float = 0.0005
    """The fraction of data that should be put aside for validation."""
    seed: int = 42
    """The random seed for shuffling the dataset."""
    num_workers: int = 8
    """How many DataLoader processes to use for loading."""

    tokenizer: Optional[Tokenizer] = field(default=None, repr=False, init=False)
    batch_size: int = field(init=False, repr=False, default=1)
    seq_length: int = field(init=False, repr=False, default=2048)

    def __post_init__(self):
        super().__init__()
        # Could be a remote path (s3://) or a local path
        self.fineweb = str(self.data_path).rstrip("/") + "/pretrain/train"
        self.finemath = str(self.data_path).rstrip("/") + "/continued_pretrain/finemath/train"
        self.required_paths = [self.fineweb, self.finemath]

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def prepare_data(self) -> None:
        for path in self.required_paths:
            if not path.startswith("s3://") and not Path(path).is_dir():
                raise FileNotFoundError(
                    "The data path for MixedFW8FM22 is expected to be the directory containing these subdirectories:"
                    f" `pretrain/train`, `continued_pretrain/finemath/train`."
                    f" The directory {path} does not exist."
                    " Set it via `--data.data_path=...`"
                )

    def train_dataloader(self) -> DataLoader:
        from litgpt.litdata.streaming import CombinedStreamingDataset, StreamingDataLoader, StreamingDataset, TokensLoader
        
        train_datasets = [
            StreamingDataset(
                input_dir=self.fineweb,
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=True,
                drop_last=True,
            ),
            StreamingDataset(
                input_dir=self.finemath,
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=True,
                drop_last=True,
            ),
        ]

        # Total: 30BT = fineweb: 8BT + finemath: 22BT
        weights = (8, 22)
        train_data = CombinedStreamingDataset(
            datasets=train_datasets, seed=self.seed, weights=weights, iterate_over_all=False
        )
        train_dataloader = StreamingDataLoader(
            train_data, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        return None
