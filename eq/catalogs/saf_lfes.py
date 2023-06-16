import io
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import requests
import torch

from eq.data import Catalog, InMemoryDataset, Sequence, default_catalogs_dir

from .utils import train_val_test_split_sequence

CAT_PATH = '/home/gafarge/projects/data/catalogs/saf_lfes_shelly17.txt'
# sorry about this, make url work, but I can't... :
# https://agupubs.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2F2017JB014047&file=jgrb52060-sup-0002-DataS1.txt

COL_NAMES = [ "year", "month", "day", "s_of_day", "hour", "minute", "second", "ccsum", "meancc", "med_cc", "seqday", "ID", "latitude", "longitude", "depth", "n_chan" ]

class SAF_LFEs(Catalog):
    def __init__(self,
        root_dir: Union[str, Path] = default_catalogs_dir / "SAF_LFEs",
        mag_completeness: float = 1.0,
        train_start_ts: pd.Timestamp = pd.Timestamp("2010-01-01"),
        val_start_ts: pd.Timestamp = pd.Timestamp("2013-01-01"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2015-01-01"),
    ):
        metadata = {
            "name": f"ShellyEtAl",
            "freq": "1D",
            "mag_completeness": mag_completeness,
            "start_ts": pd.Timestamp("2001-04-06"),
            "end_ts": pd.Timestamp("2016-09-20")
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
            CAT_PATH,
            names=COL_NAMES,
            comment='%',
            delim_whitespace=True,
            index_col=False
            )
        raw_df["date"] = pd.to_datetime(raw_df[["year", "month", "day", "hour",
            "minute", "second"]])
        raw_df.sort_values(by=["date"], inplace=True)

        print("Processing...")
        # LFEs do not have magnitude in this catalog... let's make a fake one
        mag = [1 for ii in range(raw_df.shape[0])]
        timestamps = raw_df.date.to_numpy()

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
