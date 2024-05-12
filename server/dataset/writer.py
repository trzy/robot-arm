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
    def __init__(self, recording_dir: str | None, dataset_prefix: str):
        self._recording_dir = recording_dir
        self._prefix = dataset_prefix
        self._dataset_dir = None
        self._video_writer = None
        self._frame_samples: List[np.ndarray] = []
        self._observed_motor_radians_samples: List[np.ndarray] = []
        self._target_motor_radians_samples: List[np.ndarray] = []

    def __del__(self):
        self.finish()

    def record_observation(self, frame: np.ndarray, observed_motor_radians: List[float], target_motor_radians: List[float]):
        if self._recording_dir is None:
            # No recording directory specified, do not record
            return

        height, width, _ = frame.shape
        assert frame.dtype == np.uint8

        if self._video_writer is None:
            # Lazily start new training example recording session when we actually get an 
            # observation
            self._dataset_dir = get_next_numbered_dirname(prefix=self._prefix, root_dir=self._recording_dir)
            os.makedirs(name=self._dataset_dir, exist_ok=True)
            video_filepath = os.path.join(self._dataset_dir, "video.mp4")
            self._video_writer = cv2.VideoWriter(filename=video_filepath, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=30.0, frameSize=(width,height))
            print(f"Writing observations to {self._dataset_dir}...")

        # Write to mp4
        self._video_writer.write(image=frame)   # time taken by this call is ~2ms

        # Append data in memory
        self._frame_samples.append(frame)
        self._observed_motor_radians_samples.append(np.array(observed_motor_radians))
        self._target_motor_radians_samples.append(np.array(target_motor_radians))

        # Ensure all data has consistent dimensions
        assert len(observed_motor_radians) == len(target_motor_radians)
        if len(self._observed_motor_radians_samples) >= 2:
            assert len(observed_motor_radians) == len(self._observed_motor_radians_samples[-2])
            assert frame.shape == self._frame_samples[-2].shape
    
    def finish(self):
        if self._video_writer is not None:
            # Finish video
            self._video_writer.release()
            self._video_writer = None

            # Write h5 file
            if len(self._frame_samples) <= 0:
                return
            num_motors = len(self._observed_motor_radians_samples[0])
            num_observations = len(self._observed_motor_radians_samples)
            height, width, channels = self._frame_samples[0].shape
            filepath = os.path.join(self._dataset_dir, "data.h5")
            with h5py.File(name=filepath, mode="w", rdcc_nbytes=1024**2*2) as root:
                root.attrs['sim'] = False   # TODO: is this needed?

                # Follower data, including RGB images from camera, goes under /observations
                follower = root.create_group("observations")
                qpos = follower.create_dataset(name="qpos", shape=(num_observations, num_motors))
                qvel = follower.create_dataset(name="qvel", shape=(num_observations, num_motors))
                camera_images = follower.create_group("images")
                camera_images.create_dataset(name="top", shape=(num_observations, height, width, channels), dtype="uint8", chunks=(1, height, width, channels))

                # Leader data, the motor commands at each time step
                action = root.create_dataset(name="action", shape=(num_observations, num_motors))

                # Store the data we have accumulated
                root["/observations/qpos"][...] = self._observed_motor_radians_samples
                root["/observations/qvel"][...] = np.zeros((num_observations, num_motors))
                root["/observations/images/top"][...] = self._frame_samples
                root["/action"][...] = self._target_motor_radians_samples
            print(f"Dataset written to {self._dataset_dir}")


