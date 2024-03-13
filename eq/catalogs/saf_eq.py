import io
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch

from eq.data import Catalog, InMemoryDataset, Sequence, default_catalogs_dir

from .utils import train_val_test_split_sequence

if os.getlogin() == 'gafarge':
    workdir = '/home/gafarge/'
elif os.getlogin() == 'root':
    workdir = '/Users/gaspard/work/'

PATH = workdir + "projects/data/Catalogs/parkfield_ncsn_ddrt_00_23/parkfield_ncsn_ddrt_00_23.txt"
COLS = ['DateTime', 'latitude', 'longitude', 'depth', 'mag', 'MagType', 'NbStations', 'Gap', 'Distance', 'RMS', 'Source', 'EventID']
lon_range = [-121.2, -119.8]  # box around Parkfield section
lat_range = [35.3, 36.5]

class SAF_EQ(Catalog):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "SAF_EQ",
        mag_completeness: float = 0.0,
        train_start_ts: pd.Timestamp = pd.Timestamp("2004-01-01"),
        val_start_ts: pd.Timestamp = pd.Timestamp("2015-05-27"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2019-03-15"),
    ):
        metadata = {
            "name": f"NCEDC",
            "freq": "1D",
            "mag_roundoff_error": 0.1,
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
        if train_start_ts is None:
            train_start_ts = metadata["start_ts"]
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
        raw_df = pd.read_csv(PATH, names=COLS, skiprows=1)
        raw_df["date"] = pd.to_datetime(raw_df["DateTime"], format='%Y/%m/%d %H:%M:%S.%f')
        raw_df.sort_values(by=["date"], inplace=True)

        print('Processing...')
        # events in geographical box
        print('Filtering events in geographical box')
        indicator = (
            (raw_df.latitude > lat_range[0])
            & (raw_df.latitude < lat_range[1])
            & (raw_df.longitude > lon_range[0])
            & (raw_df.longitude < lon_range[1])
        )
        raw_df = raw_df[indicator]

        # events "on" fault
        print('Filtering events on fault')
        offset = 5  # in km
        fault_line = dict(N=dict(longitude=-121.029665, latitude=36.446555),
                          S=dict(longitude=-119.870418, latitude=35.323809))
        N = np.array([fault_line['N']['longitude']*111*np.cos(fault_line['N']['latitude']/180), fault_line['N']['latitude'] * 111])
        S = np.array([fault_line['S']['longitude']*111*np.cos(fault_line['N']['latitude']/180), fault_line['S']['latitude'] * 111])

        eq_loc_km = np.vstack([raw_df.longitude.values, raw_df.latitude.values]).T* np.array([111*np.cos(fault_line['N']['latitude']/180), 111])
        eq_dist = np.linalg.norm(np.expand_dims(np.cross(N-S, S-eq_loc_km), -1), axis=-1)/np.linalg.norm(N-S)
        raw_df = raw_df[(eq_dist < offset)]

        # removing duplicates
        print('Removing duplicates')
        raw_df.drop_duplicates(subset=["date"], inplace=True)
        raw_df.sort_values(by="date", inplace=True)
        raw_df.reset_index(drop=True, inplace=True)

        # filter events above magnitude of completeness and within the time range
        print('Filtering events above magnitude of completeness and within the time range')
        subset_df = raw_df.loc[(raw_df["mag"] > self.metadata["mag_completeness"]) 
                               & (raw_df["date"] >= self.metadata["start_ts"]) 
                               & (raw_df["date"] <= self.metadata["end_ts"])]

        # format the catalog into a sequence
        print('Formatting the catalog into a sequence')
        start_ts = self.metadata["start_ts"]
        end_ts = self.metadata["end_ts"]

        assert subset_df.date.min() > start_ts
        assert subset_df.date.max() < end_ts

        t_start = 0.0
        t_end = (end_ts - start_ts) / pd.Timedelta("1 day")

        arrival_times = ((subset_df.date - start_ts) / pd.Timedelta("1 day")).values
        inter_times = np.diff(arrival_times, prepend=[t_start], append=[t_end])
        mag = subset_df["mag"].values
        seq = Sequence(
            inter_times=torch.as_tensor(inter_times, dtype=torch.float32),
            mag=torch.as_tensor(mag, dtype=torch.float32),
        )
        dataset = InMemoryDataset(sequences=[seq])
        dataset.save_to_disk(self.root_dir / "full_sequence.pt")