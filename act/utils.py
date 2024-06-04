from dataclasses import dataclass
import os
from typing import List

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import IPython
e = IPython.embed


####################################################################################################
# Dataset Loading
#
# Routines for detecting and loading the number of example episodes in the dataset directory.
####################################################################################################

@dataclass
class ExampleEpisode:
    filepath: str
    start_idx: int

@dataclass
class ExampleEpisodes:
    episode_length: int
    episodes: List[ExampleEpisode]

    def num_episodes(self) -> int:
        return len(self.episodes)

def get_example_dirs(dataset_dir: str) -> List[str]:
    return [ os.path.join(dataset_dir, dir) for dir in os.listdir(dataset_dir) if dir.startswith("example-") and os.path.isdir(os.path.join(dataset_dir, dir)) ]

#
# Given observations of length total_data_length, produces N starting indices from which
# episode_length can safely be taken. E.g., if episode_length=135:
#
# total_data_length=135, indices=[0]
# total_data_length=239, indices=[0, 104]
# total_data_length=359, indices=[0, 135, 224]
#
def produce_episode_start_indices(total_data_length: int, episode_length: int):
    indices = []
    idx = -episode_length
    while True:
        # Try to advance but if the episode we woudl produce exceeds total data length, need to shift
        # back to accommodate and produce a final, overlapping chunk
        new_idx = min(idx + episode_length, total_data_length - episode_length)
        if new_idx == idx:
            break
        idx = new_idx
        indices.append(idx)
    return indices

def get_example_episodes(dataset_dir: str, chunk_size: int) -> ExampleEpisodes:
    # Detect all episode files and find the minimum episode length
    min_length = 1e9
    dirs = get_example_dirs(dataset_dir=dataset_dir)
    if len(dirs) == 0:
        raise RuntimeError(f"No episode subdirectories (e.g., 'example-0', ...) found in {dataset_dir}")
    filepaths = []
    examples_too_short = []
    for i in range(len(dirs)):
        dir = dirs[i]
        filepath = os.path.join(dir, "data.hdf5")
        filepaths.append(filepath)
        with h5py.File(name=filepath, mode="r") as fp:
            length = len(fp["/action"])
            min_length = min(min_length, length)
            if length < chunk_size:
                examples_too_short.append(dir)
    print(f"Minimum example episode length: {min_length}")
    if len(examples_too_short) > 0:
        raise RuntimeError(f"Example episode length must be greater than or equal to chunk_size {chunk_size}. Remove the following episodes that are too small and try again: {', '.join(examples_too_short)}")
    
    # Chop up episodes into sub-episodes, each having min_length
    example_episodes: List[ExampleEpisode] = []
    i = 0
    for filepath in filepaths:
        with h5py.File(name=filepath, mode="r") as fp:
            # Determine how many sub-episodes we need to create and what their start indices are
            current_length = fp["/action"].shape[0]
            start_indices = produce_episode_start_indices(total_data_length=current_length, episode_length=min_length)

            # Create a sub-episode for each
            for idx in start_indices:
                example_episodes.append(ExampleEpisode(filepath=filepath, start_idx=idx))
                print(f"Episode {i}: {filepath} at [{idx}:{idx+min_length}]")
                i += 1
    
    # Return the example episodes we will train and validate on
    print(f"Found {len(example_episodes)} example episodes in {len(filepaths)} files")
    return ExampleEpisodes(episode_length=min_length, episodes=example_episodes)


####################################################################################################
# Torch Dataloader
####################################################################################################

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_idxs, episodes: ExampleEpisodes, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_idxs = episode_idxs
        self.episodes = episodes
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_idxs)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_idx = self.episode_idxs[index]
        episode = self.episodes.episodes[episode_idx]
        with h5py.File(episode.filepath, 'r') as root:
            is_sim = root.attrs['sim']

            # Get the sub-episode, not the whole episode. Episodes are variable length but we
            # chopped them up into a series of sub-episodes virtually (without actually modifying
            # the files, just retaining indices to their starting points).
            action_samples = root['/action'][episode.start_idx : episode.start_idx + self.episodes.episode_length]
            qpos_samples = root['/observations/qpos'][episode.start_idx : episode.start_idx + self.episodes.episode_length]
            image_samples_by_cam_name = { cam_name: root[f'/observations/images/{cam_name}'][episode.start_idx : episode.start_idx + self.episodes.episode_length] for cam_name in self.camera_names }

            # Sample within our sub-episode randomly
            original_action_shape = action_samples.shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = qpos_samples[start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_samples = image_samples_by_cam_name[cam_name]
                image_dict[cam_name] = image_samples[start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = action_samples[start_ts:]
                action_len = episode_len - start_ts
            else:
                action = action_samples[max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(examples: ExampleEpisodes):
    all_qpos_data = []
    all_action_data = []
    for episode in examples.episodes:
        with h5py.File(episode.filepath, 'r') as root:
            qpos = root['/observations/qpos'][episode.start_idx : episode.start_idx + examples.episode_length]
            qvel = root['/observations/qvel'][episode.start_idx : episode.start_idx + examples.episode_length]
            action = root['/action'][episode.start_idx : episode.start_idx + examples.episode_length]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir: str, chunk_size: int, camera_names: List[str], batch_size_train: int, batch_size_val: int):
    # Detect examples
    print(f'\nData from: {dataset_dir}\n')
    examples = get_example_episodes(dataset_dir=dataset_dir, chunk_size=chunk_size)

    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(examples.num_episodes())
    train_indices = shuffled_indices[:int(train_ratio * examples.num_episodes())]
    val_indices = shuffled_indices[int(train_ratio * examples.num_episodes()):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(examples=examples)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, examples, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, examples, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


####################################################################################################
# Helper Functions
####################################################################################################

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)