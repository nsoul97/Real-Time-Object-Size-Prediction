import torch as th
import torch.utils.data as data
import pandas as pd
import re
import os


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
        features[mov_name]['time'] = mov_data['Time'] - mov_data['Time'].iloc[0]

        # Thumb - Index Aperture
        features[mov_name]['ti_ap'] = ((mov_data["RThumb4FingerTip.x"] - mov_data["RIndex4FingerTip.x"]) ** 2 +
                                       (mov_data["RThumb4FingerTip.y"] - mov_data["RIndex4FingerTip.y"]) ** 2) ** 0.5

        # Thumb - Middle Aperture
        features[mov_name]['tm_ap'] = ((mov_data["RThumb4FingerTip.x"] - mov_data["RMiddle4FingerTip.x"]) ** 2 +
                                       (mov_data["RThumb4FingerTip.y"] - mov_data["RMiddle4FingerTip.y"]) ** 2) ** 0.5

        # Index - Middle Aperture
        features[mov_name]['im_ap'] = ((mov_data["RIndex4FingerTip.x"] - mov_data["RMiddle4FingerTip.x"]) ** 2 +
                                       (mov_data["RIndex4FingerTip.y"] - mov_data["RMiddle4FingerTip.y"]) ** 2) ** 0.5
    return features


data = read_dataset("/home/soul/Development/Object Size Prediction/dataset/")
features = calculate_features(data)
print(features)
