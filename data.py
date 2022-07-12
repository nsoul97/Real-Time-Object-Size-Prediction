import numpy as np
import pandas as pd
import torch as th
import torch.utils.data as data_utils
import re
import os
from typing import Literal, Optional


def read_dataset(dataset_path):
    """ Read all the movements of the dataset and store them in a dictionary. Choose which joints to read for each
        movement and which attributes to read for each joint.

    Args:
        path (str, optional): The path of the movement dataset.

    Returns:
        data (dictionary): A dictionary with the movement filename as the key and a pd.DataFrame containing the
                           corresponding skeletal data of the movement as the value.
    """

    dataset_data_path = os.path.join(dataset_path, "data")
    dataset_info_path = os.path.join(dataset_path, "RtG_onset_end.csv")
    info = pd.read_csv(dataset_info_path, index_col='Movement')

    joints = ["RThumb4FingerTip", "RIndex4FingerTip", "RMiddle4FingerTip"]
    attr = ['x', 'y']
    cols = ["Time"] + [f"{j}.{at}" for j in joints for at in attr]

    data = dict()
    participants = os.listdir(dataset_data_path)
    for p in participants:
        movements = os.listdir(os.path.join(dataset_data_path, p))
        for m in movements:
            if not re.match(f'{p}_[S|M|L]_(0[1-9]|[1-2][0-9]|30).csv', m): continue

            mov_info = info.loc[m]
            df = pd.read_csv(os.path.join(dataset_data_path, p, m), usecols=cols)
            onset = mov_info['RtG onset']
            end = mov_info['RtG end']

            data[m] = df[(onset <= df["Time"]) & (df["Time"] < end)]
    return data


def calculate_features(data):
    features = dict()
    for mov_name, mov_data in data.items():
        features[mov_name] = dict()

        # Time
        features[mov_name]['elapsed_time'] = mov_data['Time'] - mov_data['Time'].iloc[0]

        # Thumb - Index Aperture
        features[mov_name]['TI-Ap'] = ((mov_data["RThumb4FingerTip.x"] - mov_data["RIndex4FingerTip.x"]) ** 2 +
                                       (mov_data["RThumb4FingerTip.y"] - mov_data["RIndex4FingerTip.y"]) ** 2) ** 0.5

        # Thumb - Middle Aperture
        features[mov_name]['TM-Ap'] = ((mov_data["RThumb4FingerTip.x"] - mov_data["RMiddle4FingerTip.x"]) ** 2 +
                                       (mov_data["RThumb4FingerTip.y"] - mov_data["RMiddle4FingerTip.y"]) ** 2) ** 0.5

        # Index - Middle Aperture
        features[mov_name]['IM-Ap'] = ((mov_data["RIndex4FingerTip.x"] - mov_data["RMiddle4FingerTip.x"]) ** 2 +
                                       (mov_data["RIndex4FingerTip.y"] - mov_data["RMiddle4FingerTip.y"]) ** 2) ** 0.5
    return features


def all_in_kfold(data, rng):
    """ Split the movements into 10 partitions of approximately equal size (+-1) according to the "all-in" strategy.
        The movements of each partition are selected randomly. The partitions have approximately the same number of
        movements (+-1) for a given (participant, object) pair.

    Args:
        data: A dictionary with the movement filename as the key and a pd.DataFrame containing the
                               corresponding data of the movement as the value. This dictionary contains all the
                               movements of the dataset.
        rng:  A numpy random generator to reproduce results if a seed is given.

    Returns: A list containing 10 lists, one for each partition. The nested lists contain the movement filenames of the
             partition.

    """

    part_movements = {f"P{p}": {'S': [], 'M': [], 'L': []} for p in range(1, 9)}
    for mov in data.keys():
        part_id = mov[:2]
        obj_id = mov[3:4]
        part_movements[part_id][obj_id].append(mov)

    partitions = [[] for p in range(10)]
    for part_id in sorted(part_movements.keys()):
        for plfiles in sorted(part_movements[part_id].values()):

            rng.shuffle(plfiles)

            part_div = len(plfiles) // 10
            part_mod = len(plfiles) % 10
            for i in range(10):
                partitions[i] = partitions[i] + [filename for filename in
                                                 plfiles[i * part_div:(i + 1) * part_div]]

            partitions = sorted(partitions, key=lambda x: len(x))
            for i, f in enumerate(range(10 * part_div, 10 * part_div + part_mod)):
                partitions[i].append(plfiles[f])
    return partitions


def one_out_kfold(data):
    """ Split the movements into 8 partitions of approximately equal size according to the "one-out" strategy.
        The movements of the i-th partition are selected as the movements of the i-th participant.

    Args:
        data: A dictionary with the movement filename as the key and a pd.DataFrame containing the corresponding data of
              the movement as the value. This dictionary contains all the movements of the dataset.

    Returns: A list containing 8 lists, one for each partition. The nested lists contain the movement filenames of the
             partition.
    """

    part_movements = {f"P{p}": [] for p in range(1, 9)}
    for mov in sorted(data.keys()):
        part_movements[mov[:2]].append(mov)
    partitions = sorted(part_movements.values())
    return partitions


class Data:
    LABELS_2_IND = {'S': 0,
                    'M': 1,
                    'L': 2}

    def __init__(self,
                 dataset_path: str,
                 strategy: Literal['all-in', 'one-out'],
                 max_pad_length: int,
                 seed: Optional[int] = None) -> None:

        data = read_dataset(dataset_path)
        features = calculate_features(data)

        rng = np.random.default_rng(seed)
        if strategy == 'all-in':
            partitioned_fnames = all_in_kfold(data, rng)
        else:
            partitioned_fnames = one_out_kfold(data)

        self.partitions = [DatasetPartition(fnames_in_partition, features, max_pad_length)
                           for fnames_in_partition in partitioned_fnames]

    def get_train_test_datasets(self, p):
        train_dataset = data_utils.ConcatDataset([self.partitions[i] for i in range(len(self.partitions)) if i != p])
        test_dataset = self.partitions[p]

        return train_dataset, test_dataset


class DatasetPartition(data_utils.Dataset):

    def __init__(self, fnames_in_partition, features, max_pad_length):
        n = len(fnames_in_partition)

        self.time = th.zeros(n, max_pad_length)
        self.aperture_features = th.zeros(n, 3, max_pad_length)
        self.padding_mask = th.zeros(n, max_pad_length)
        self.labels = th.zeros(n)

        for i, fname in enumerate(fnames_in_partition):
            mov_features = features[fname]
            mov_time = th.tensor(mov_features['elapsed_time'].values)
            mov_ti_ap = th.tensor(mov_features['TI-Ap'].values)
            mov_tm_ap = th.tensor(mov_features['TM-Ap'].values)
            mov_im_ap = th.tensor(mov_features['IM-Ap'].values)
            mov_label = Data.LABELS_2_IND[fname[3:4]]
            t = mov_time.shape[0]

            self.time[i, :t] = mov_time
            self.aperture_features[i, 0, :t] = mov_ti_ap
            self.aperture_features[i, 1, :t] = mov_tm_ap
            self.aperture_features[i, 2, :t] = mov_im_ap
            self.padding_mask[i, :t] = 1
            self.labels[i] = mov_label

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        return (self.time[item],
                self.aperture_features[item],
                self.padding_mask[item],
                self.labels[item])

