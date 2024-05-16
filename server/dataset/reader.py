#
# reader.py
# Bart Trzynadlowski
#
# Reads observations from a dataset.
#

from dataclasses import dataclass

import h5py
import numpy as np


@dataclass
class Dataset:
    frames: np.ndarray
    observed_motor_radians: np.ndarray
    target_motor_radians: np.ndarray

def read_dataset(filepath: str) -> Dataset:
    with h5py.File(name=filepath, mode="r") as fp:
        actions = np.array(fp["/action"][...])
        qpos = np.array(fp["/observations/qpos"][...])
        frames = np.array(fp["/observations/images/top"][...])
        assert len(actions) == len(qpos)
        assert len(actions) == len(frames)
        return Dataset(frames=frames, observed_motor_radians=qpos, target_motor_radians=actions)

