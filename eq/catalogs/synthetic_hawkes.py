from pathlib import Path
from typing import Union

import torch
import numpy as np
import pandas as pd

from eq.data import Catalog, InMemoryDataset, Sequence, default_catalogs_dir

from eq.catalogs.utils import train_val_test_split_sequence

# Todo:
# 1- Implement catalog generation
# 2- Multi catalog generation
# 3- Implement exp, n-exp


class Hawkes_SingleCatalog(Catalog):
    """
    Generate a catalog of event times solely from a generalized Omori law
    (Power-law Hawkes process). For now, reads input, in the future, implement
    generation on the go.

    """

    def __init__(self,
                 arrival_times: np.ndarray, # input arrival times in days
                 root_dir: Union[Path, str] = default_catalogs_dir/"Hawkes_SingleCatalog",
                 duration: float = 1000, # in days
                 start_ts: pd.Timestamp = pd.Timestamp(0),
                 train_frac: float = 0.6,
                 val_frac: float = 0.2,
                 test_frac: float = 0.1,
                 mag_completeness = None,
                ):
        metadata = {
            "name": f"Hawkes_SingleCaltalog",
            "mag_completeness": mag_completeness,
            "start_ts": start_ts,
            "end_ts": start_ts+pd.Timedelta(days=duration),
        }
        self.arrival_times = arrival_times  # from input

        super().__init__(root_dir=root_dir, metadata=metadata)

        # --> Load in sequence
        self.full_sequence = InMemoryDataset.load_from_disk(self.root_dir/"full_sequence.pt")[0]

        # --> Split full sequence into train / val / test parts
        self.metadata["train_start_ts"] = pd.Timestamp(start_ts)
        self.metadata["val_start_ts"] = pd.Timestamp(start_ts) + pd.Timedelta(days=train_frac*duration)
        self.metadata["test_start_ts"] = pd.Timestamp(start_ts) + pd.Timedelta(days=(train_frac+val_frac)*duration)
        seq_train, seq_val, seq_test = train_val_test_split_sequence(
            seq=self.full_sequence,
            start_ts=self.metadata["start_ts"],
            train_start_ts=self.metadata["train_start_ts"],
            val_start_ts=self.metadata["val_start_ts"],
            test_start_ts=self.metadata["test_start_ts"],
        )
        self.train = InMemoryDataset([seq_train])
        self.val = InMemoryDataset([seq_val])
        self.test = InMemoryDataset([seq_test])

    @property
    def required_files(self):
        return ["full_sequence.pt", "metadata.pt"]

    def generate_catalog(self):
        # --> Check start/end of time series consistent with metadata
        start_ts = self.metadata['start_ts']
        end_ts = self.metadata['end_ts']
        assert pd.Timedelta(days=self.arrival_times.min()) > pd.Timedelta(0)
        assert pd.Timedelta(days=self.arrival_times.max()) < end_ts - start_ts

        # --> Build inter-event times
        t_start = 0.0
        t_end = (end_ts - start_ts) / pd.Timedelta("1 day")
        inter_times = np.diff(self.arrival_times, prepend=[t_start], append=[t_end])

        # --> Torch the sequence
        seq = Sequence(inter_times=torch.as_tensor(inter_times, dtype=torch.float32))
        dataset = InMemoryDataset(sequences=[seq])
        dataset.save_to_disk(self.root_dir / "full_sequence.pt")
