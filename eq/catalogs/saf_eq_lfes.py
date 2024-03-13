import io
from pathlib import Path
from typing import Union

import numpy as np
import datetime as dt
import pandas as pd
import requests
import torch

from eq.data import Catalog, InMemoryDataset, Sequence, default_catalogs_dir

from .utils import train_val_test_split_sequence

PATH = '/home/gafarge/projects/data/Catalogs/parkfield_eq_lfe/saf_eq_lfe.csv'

class SAF_EQ_LFEs(Catalog):

    def __init__(self,
        root_dir: Union[str, Path] = default_catalogs_dir / "SAF_EQ_LFEs",
        mag_completeness: float = 0,
        train_start_ts: pd.Timestamp = pd.Timestamp("2004-01-01"),
        val_start_ts: pd.Timestamp = pd.Timestamp("2015-05-27"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2019-03-15"),
    ):
        metadata = {
            "name": f"saf_eq_lfe",
            "freq": "1D",
            "mag_completeness": mag_completeness,
            "start_ts": pd.Timestamp("2004-01-01"),
            "end_ts": pd.Timestamp("2023-01-01")
        }
        super().__init__(root_dir=root_dir, metadata=metadata)

        # Load the full sequence
        self.full_sequence = InMemoryDataset.load_from_disk(
            self.root_dir / "full_sequence.pt"
        )[0]

        # Split full sequence into train / val / test parts
        self.metadata["train_start_ts"] = pd.Timestamp(train_start_ts)
        self.metadata["val_start_ts"] = pd.Timestamp(val_start_ts)
        self.metadata["test_start_ts"] = pd.Timestamp(test_start_ts)
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
        print("Downloading...")
        raw_df = pd.read_csv(
                PATH,
                skiprows=12,
                parse_dates=["date"]
            )
        
        print("Processing...")
        raw_df = raw_df[raw_df.date >= self.metadata["start_ts"]]

        raw_df.drop_duplicates(subset=["date"], inplace=True)
        raw_df.sort_values(by="date", inplace=True)
        raw_df.reset_index(drop=True, inplace=True)

        timestamps = raw_df.date.to_numpy()
        mag = raw_df.mag.to_numpy()

        # Compute inter-event times
        start_ts = np.datetime64(self.metadata["start_ts"])
        end_ts = np.datetime64(self.metadata["end_ts"])
        assert timestamps.min() > start_ts
        assert timestamps.max() < end_ts

        t_start = 0.0
        t_end = (end_ts - start_ts) / pd.Timedelta("1 day")
        arrival_times = ((raw_df.date - start_ts) / pd.Timedelta("1 day")).values
        inter_times = np.diff(arrival_times, prepend=[t_start], append=[t_end])
        seq = Sequence(
            inter_times=torch.as_tensor(inter_times, dtype=torch.float32),
            mag=torch.as_tensor(mag, dtype=torch.float32),
        )
        dataset = InMemoryDataset(sequences=[seq])
        dataset.save_to_disk(self.root_dir / "full_sequence.pt")