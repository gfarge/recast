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

COL_NAMES = ["event_id", "log_M0", "std_log", "fc", "std_fc", "gamma", "std_ga", "lat", "lon", "depth", "date_str"]

class Supino_LFEs(Catalog):
    url = "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/HCWJUI/LLGGXH"

    def __init__(self,
        root_dir: Union[str, Path] = default_catalogs_dir / "Supino_LFEs",
        mag_completeness: float = 1.0,
        train_start_ts: pd.Timestamp = pd.Timestamp("2014-02-01"),
        val_start_ts: pd.Timestamp = pd.Timestamp("2015-03-01"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2015-11-01"),
    ):
        metadata = {
            "name": f"SupinoEtAl",
            "freq": "1D",
            "mag_completeness": mag_completeness,
            "start_ts": pd.Timestamp("2014-02-01"),
            "end_ts": pd.Timestamp("2016-11-09")
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
        stream = requests.get(self.url).content
        raw_df = pd.read_csv(
            io.StringIO(stream.decode("utf-8")),
            names=COL_NAMES,
            skiprows=1,
            sep='\t'
            )
        print("Processing...")
        raw_df["date"] = pd.to_datetime(raw_df['date_str'],
                format='%Y-%m-%d_%H:%M:%S.%f')
        raw_df.sort_values(by=["date"], inplace=True)

        # >> Find and remove exact duplicates
        raw_df.drop(np.where(raw_df.duplicated(["date"]))[0], inplace=True)
        raw_df.drop(np.where(raw_df["date"]<dt.datetime(2014, 2, 1))[0],
                inplace=True)
        raw_df.index = [ii for ii in range(len(raw_df))]

        timestamps = raw_df.date.to_numpy()

        mag = 2/3 * raw_df.log_M0.values - 6.07

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
