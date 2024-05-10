#
# writer.py
# Bart Trzynadlowski
#
# Writes observations to a dataset.
#

import os
from typing import List

import cv2
import h5py
import numpy as np

from ..util import get_next_numbered_dirname


class DatasetWriter:
    def __init__(self, recording_dir: str, dataset_prefix: str):
        self._recording_dir = recording_dir
        self._prefix = dataset_prefix
        self._video_writer = None

    def __del__(self):
        self.finish()

    def record_observation(self, frame: np.ndarray, observed_motor_radians: List[float], target_motor_radians: List[float]):
        if self._video_writer is None:
            # Lazily start new training example recording session when we actually get an 
            # observation
            dir = get_next_numbered_dirname(prefix=self._prefix, root_dir=self._recording_dir)
            os.makedirs(name=dir, exist_ok=True)
            video_filepath = os.path.join(dir, "video.mp4")
            self._video_writer = cv2.VideoWriter(filename=video_filepath, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=30.0, frameSize=(640,480))
            print(f"Writing observations to {dir}...")
        self._video_writer.write(image=frame)   # time taken by this call is ~2ms
    
    def finish(self):
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None



