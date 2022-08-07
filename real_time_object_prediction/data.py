import numpy as np
import pandas as pd
import torch as th
import torch.utils.data as data_utils
import re
import os
from typing import Literal, Tuple, List, Optional
import reproducibility


def get_dataset_statistics(dataset: data_utils.Dataset) -> Tuple[th.Tensor, th.Tensor]:
    feature_sum = 0
    feature_sq_sum = 0
    total_frames = 0
    for train_sample in dataset:
        _, features, mask, _ = train_sample
        feature_sum += th.sum(features, dim=0, keepdim=True)
        feature_sq_sum += th.sum(features ** 2, dim=0, keepdim=True)
        total_frames += th.sum(mask)

    mean = feature_sum / total_frames
    std_dev = feature_sq_sum / total_frames - mean

    return mean, std_dev


def read_dataset(dataset_path,
                 std_dev_window):
    """ Read all the movements of the dataset and store them in a dictionary. Choose which joints to read for each
        movement and which attributes to read for each joint.

    Args:
        dataset_path (str, optional): The path of the movement dataset.

    Returns:
        data (dictionary): A dictionary with the movement filename as the key and a pd.DataFrame containing the
                           corresponding skeletal data of the movement as the value.
    """

    dataset_data_path = os.path.join(dataset_path, "data")
    dataset_info_path = os.path.join(dataset_path, "RtG_onset_end.csv")
    info = pd.read_csv(dataset_info_path, index_col='Movement')

    joints = ["RThumb4FingerTip", "RIndex4FingerTip", "RMiddle4FingerTip"]
    attr = ['x', 'y']
    cols = ["Time"] + [f"{j}.{at}" for j in joints for at in attr] + ["RWrist.y"]

    data = dict()
    participants = os.listdir(dataset_data_path)
    for p in participants:
        movements = os.listdir(os.path.join(dataset_data_path, p))
        for m in movements:
            if not re.match(f'{p}_[S|M|L]_(0[1-9]|[1-2][0-9]|30).csv', m): continue

            mov_info = info.loc[m]
            df = pd.read_csv(os.path.join(dataset_data_path, p, m), usecols=cols)

            start_index = df[df["Time"] >= mov_info['RtG onset']].index[0] - std_dev_window + 1
            end_index = df[df["Time"] >= mov_info['RtG end']].index[0]
            data[m] = df[start_index: end_index]

    return data


def calculate_features(data: pd.DataFrame,
                       std_dev_window: int) -> dict:
    features = dict()
    for mov_name, mov_data in data.items():

        features[mov_name] = dict()

        # Wrist x-standard deviation (over the given window)
        coords = mov_data[f"RWrist.y"].to_numpy()
        wrist_std_dev = []
        for f in range(std_dev_window - 1, coords.shape[0]):
            m = np.average(coords[f - std_dev_window + 1: f + 1])
            frame_std_dev = np.sqrt(np.average((coords[f - std_dev_window + 1: f + 1] - m) ** 2))
            wrist_std_dev.append(frame_std_dev)
        features[mov_name]['wrist_y_stddev'] = wrist_std_dev

        # Ignore the frames before the movement onset
        mov_data = mov_data.iloc[std_dev_window - 1:]

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


def all_in_kfold(data):
    """ Split the movements into 10 partitions of approximately equal size (+-1) according to the "all-in" strategy.
        The movements of each partition are selected randomly. The partitions have approximately the same number of
        movements (+-1) for a given (participant, object) pair.

    Args:
        data: A dictionary with the movement filename as the key and a pd.DataFrame containing the
                               corresponding data of the movement as the value. This dictionary contains all the
                               movements of the dataset.

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

            reproducibility.rng['np_rng'].shuffle(plfiles)

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
                 std_dev_window: int,
                 d_input: int) -> None:

        data = read_dataset(dataset_path, std_dev_window)
        features = calculate_features(data, std_dev_window)

        if strategy == 'all-in':
            partitioned_fnames = all_in_kfold(data)
        else:
            partitioned_fnames = one_out_kfold(data)

        self.partitions = [DatasetPartition(fnames_in_partition, features, max_pad_length, d_input)
                           for fnames_in_partition in partitioned_fnames]

    def __iter__(self):
        self.p = 0
        return self

    def __next__(self):
        k = len(self.partitions)
        if self.p < k:
            train_dataset = data_utils.ConcatDataset([self.partitions[i] for i in range(k) if i != self.p])
            test_dataset = self.partitions[self.p]
            self.p += 1
            return train_dataset, test_dataset
        else:
            raise StopIteration


class DatasetPartition(data_utils.Dataset):

    def __init__(self,
                 fnames_in_partition: List[str],
                 features: dict,
                 max_pad_length: int,
                 d_input: int) -> None:
        super(DatasetPartition, self).__init__()

        self.max_pad_length = max_pad_length
        self.d_input = d_input

        self.time = []
        self.features = []
        self.labels = []

        for i, fname in enumerate(fnames_in_partition):
            mov_features = features[fname]
            self.time.append(th.tensor(mov_features['elapsed_time'].values))
            self.features.append(th.stack([th.tensor(mov_features['TI-Ap'].values),
                                           th.tensor(mov_features['TM-Ap'].values),
                                           th.tensor(mov_features['IM-Ap'].values),
                                           th.tensor(mov_features['wrist_y_stddev'])],
                                          dim=1))
            self.labels.append(Data.LABELS_2_IND[fname[3:4]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        mov_time = self.time[item]
        mov_features = self.features[item]
        mov_label = self.labels[item]

        t = mov_time.shape[0]

        # Pad the input features (time and aperture features)
        pad_mov_time = th.zeros(self.max_pad_length)
        pad_mov_time[:t] = mov_time

        pad_mov_features = th.zeros(self.max_pad_length, self.d_input)
        pad_mov_features[:t, :] = mov_features

        # Set the padding mask (0: mask, 1: keep)
        pad_mask = th.zeros(self.max_pad_length, dtype=th.bool)
        pad_mask[:t] = 1

        return (pad_mov_time,
                pad_mov_features,
                pad_mask,
                mov_label)


class PretrainDataset(data_utils.Dataset):
    def __init__(self,
                 dataset: data_utils.Dataset,
                 task_mode: Literal['forecast', 'value_imputation', 'mixed'],
                 mask_mode: Literal['independent', 'dependent'],
                 value_imp_mask_length: Optional[float] = None,
                 value_imp_ratio: Optional[float] = None,
                 forecast_mask_length: Optional[float] = None
                 ) -> None:
        super(PretrainDataset, self).__init__()

        self.mask_mode = mask_mode
        self.task_mode = task_mode

        self.labeled_dataset = dataset

        self.value_imp_mask_length = value_imp_mask_length
        self.value_imp_ratio = value_imp_ratio
        self.forecast_mask_length = forecast_mask_length

        assert task_mode in ['forecast', 'value_imputation', 'mixed'], "The task that will be used for pretrain may " \
                                                                       "be 'forecast', 'value_imputation' or 'mixed'."

        assert mask_mode in ['independent', 'dependent'], "The value imputation task can be applied to either " \
                                                          "'independent' or 'dependent' variables."

        if task_mode in ['forecast', 'mixed']:
            assert forecast_mask_length is not None, "When the task is 'forecast' or 'mixed', the " \
                                                     "'forecast_imp_mask_length' must be specified."

            self.forecast_pm = 1 / self.forecast_mask_length

        if task_mode in ['value_imputation', 'mixed']:
            assert value_imp_mask_length is not None and value_imp_ratio is not None, \
                "When the task is 'value_imputation' or 'mixed', the 'value_imp_mask_length' and the " \
                "'value_imp_ratio' must be specified."

            self.value_imp_pm = 1 / self.value_imp_mask_length
            self.value_imp_pu = self.value_imp_ratio / ((1 - self.value_imp_ratio) * self.value_imp_mask_length)
            self.value_imp_p_transition = np.array([self.value_imp_pm, self.value_imp_pu])

    def __len__(self) -> int:
        return len(self.labeled_dataset)

    def __getitem__(self, item: int) -> Tuple[th.Tensor, th.Tensor, int]:
        mov_time, mov_features, pad_mask, _ = self.labeled_dataset[item]
        move_length = int(th.sum(pad_mask))
        num_samples = mov_features.shape[1] if self.mask_mode == 'independent' else 1
        max_pad_length = pad_mask.shape[0]

        if self.task_mode == 'value_imputation':
            value_mask = np.zeros((max_pad_length, num_samples))
            value_mask[0] = reproducibility.rng['np_rng'].random(num_samples) > self.value_imp_ratio
            for i in range(1, move_length):
                state = value_mask[i - 1].astype(int)
                p_state = self.value_imp_p_transition[state]
                change_state = reproducibility.rng['np_rng'].random(num_samples) < p_state
                value_mask[i] = (1 - value_mask[i - 1]) * change_state + value_mask[i - 1] * (1 - change_state)

        elif self.task_mode == 'forecast':
            value_mask = np.zeros((max_pad_length, num_samples))
            stop = np.zeros(num_samples, dtype=np.bool)
            for i in range(move_length - 1, -1, -1):
                stop += reproducibility.rng['np_rng'].random(num_samples) < self.forecast_pm
                value_mask[i] = stop

        else:
            forecast_mask = np.zeros((max_pad_length, num_samples))
            stop = np.zeros(num_samples, dtype=np.bool)
            for i in range(move_length - 1, -1, -1):
                stop += reproducibility.rng['np_rng'].random(num_samples) < self.forecast_pm
                forecast_mask[i] = stop

            value_imputation_mask = np.zeros((max_pad_length, num_samples))
            value_imputation_mask[0] = reproducibility.rng['np_rng'].random(num_samples) > self.value_imp_ratio
            for i in range(1, move_length):
                state = value_imputation_mask[i - 1].astype(int)
                p_state = self.value_imp_p_transition[state]
                change_state = reproducibility.rng['np_rng'].random(num_samples) < p_state
                value_imputation_mask[i] = (1 - value_imputation_mask[i - 1]) * change_state + \
                                           value_imputation_mask[i - 1] * (1 - change_state)
            value_mask = np.logical_and(forecast_mask, value_imputation_mask)

        value_mask = th.from_numpy(value_mask).bool()
        return mov_features, pad_mask, value_mask


class PartiallyObservableDataset(data_utils.Dataset):
    def __init__(self,
                 dataset: data_utils.Dataset,
                 keep_frames: Optional[int] = None,
                 mov_completion_perc: Optional[float] = None
                 ) -> None:
        super(PartiallyObservableDataset, self).__init__()

        self.dataset = dataset
        self.keep_frames = keep_frames
        self.mov_completion_perc = mov_completion_perc

        if keep_frames is not None:
            self.mode = 'keep_frames'
            assert mov_completion_perc is None, "Only 'keep_frames' or 'mov_completion_perc' can be specified."
        else:
            self.mode = 'mov_completion_perc'
            assert mov_completion_perc is not None, "Only 'keep_frames' or 'mov_completion_perc' can be specified."

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item: int) -> Tuple[th.Tensor, th.Tensor, int]:

        mov_time, mov_features, pad_mask, label = self.dataset[item]

        if self.mode == 'keep_frames':
            seq_len = th.sum(pad_mask.int()).item()
            keep_frames = min(self.keep_frames, seq_len)
        else:
            total_elapsed_time = mov_time.masked_select(pad_mask)[-1]
            t_perc = mov_time / total_elapsed_time
            keep_frames_mask = th.logical_and(t_perc <= self.mov_completion_perc, pad_mask).int()
            keep_frames = th.sum(keep_frames_mask)

        mov_features[keep_frames:] = 0
        pad_mask[keep_frames:] = 0

        return mov_features, pad_mask, label
