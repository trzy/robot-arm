from glob import glob
import os
import timeit
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

import IPython
e = IPython.embed


####################################################################################################
# Dataset Loading
#
# Routines for detecting and loading the example episodes in the dataset directory.
####################################################################################################

def get_dirs(dataset_dir: str) -> List[str]:
    # First, split on ,
    specified_dirs = dataset_dir.split(",")

    # Expand wildcards for each and retain only the directories. Then, extract the episode
    # directories from each.
    episode_dirs = []
    for specified_dir in specified_dirs:
        dataset_dirs = [ dir for dir in glob(specified_dir) if os.path.isdir(dir) ]
        for dir in dataset_dirs:
            for episode_dir in os.listdir(dir):
                fully_resolved_episode_dir = os.path.join(dir, episode_dir)
                if episode_dir.startswith("example-") and os.path.isdir(fully_resolved_episode_dir):
                    episode_dirs.append(fully_resolved_episode_dir)
    
    return list(set(episode_dirs))

def get_episode_filepaths_and_camera_names(dataset_dir: str) -> Tuple[List[str], List[str]]:
    dirs = get_dirs(dataset_dir=dataset_dir)
    filepaths: List[str] = []
    camera_names: List[str] = []
    for dir in dirs:
        # Validate each file
        filepath = os.path.join(dir, "data.hdf5")
        filepaths.append(filepath)
        with h5py.File(name=filepath, mode="r") as fp:
            num_actions = len(fp["/action"])
            num_qpos = len(fp["/observations/qpos"])
            if num_actions != num_qpos:
                raise ValueError(f"{filepath}: /observations/qpos ({num_qpos}) does not have the same number of samples as /action ({num_actions})")
            camera_names_this_file = sorted(list(fp["/observations/images"].keys()))
            if len(camera_names_this_file) == 0:
                raise ValueError(f"{filepath}: no image data")
            expected_camera_names = [ f"cam{i}" for i in range(len(camera_names_this_file)) ]
            if camera_names_this_file != expected_camera_names:
                raise ValueError(f"{filepath}: camera names must be [cam0,...,camN] but found: {camera_names_this_file}")
            if len(camera_names) == 0:
                camera_names = expected_camera_names
            elif camera_names != camera_names_this_file:
                raise ValueError(f"{filepath}: camera names in this file ({camera_names_this_file}) do not match names in other files ({camera_names})")
    return filepaths, camera_names


####################################################################################################
# Torch Dataloader
####################################################################################################

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, chunk_size, episode_idxs, filepaths: List[str], camera_names: List[str], norm_stats):
        super(EpisodicDataset).__init__()
        self.chunk_size = chunk_size
        self.episode_idxs = episode_idxs
        self.filepaths = filepaths
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_idxs)

    def __getitem__(self, index):
        t0 = timeit.default_timer()

        episode_idx = self.episode_idxs[index]
        filepath = self.filepaths[episode_idx]
        with h5py.File(filepath, 'r') as root:
            sample_full_episode = False # hardcode
            is_sim = root.attrs['sim']

            # Sample within episode randomly
            action_shape = root['/action'].shape
            episode_len = action_shape[0]
            start_ts = 0 if sample_full_episode else np.random.choice(episode_len)

            # Get observations at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            
            # Get all actions after and including start_ts, up to chunk_size in length
            if is_sim:
                raise NotImplemented()
            else:
                start_idx = max(0, start_ts - 1)    # hack, to make timesteps more aligned
                end_idx = min(start_idx + self.chunk_size, episode_len)
                action = root['/action'][start_idx : end_idx]
                action_len = end_idx - start_idx

        self.is_sim = is_sim
        padded_action = np.zeros(shape=(self.chunk_size, *action_shape[1:]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
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

        t1 = timeit.default_timer()
        return image_data, qpos_data, action_data, is_pad, (t1 - t0)


def get_norm_stats(filepaths: List[str]):
    all_qpos_data = []
    all_action_data = []
    for filepath in filepaths:
        with h5py.File(filepath, 'r') as root:
            for qpos in root["/observations/qpos"][:]:
                all_qpos_data.append(torch.from_numpy(qpos))
            for action in root['/action'][:]:
                all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir: str, chunk_size: int, batch_size_train: int, batch_size_val: int):
    # Detect examples
    print(f"\nData from: {dataset_dir}")
    filepaths, camera_names = get_episode_filepaths_and_camera_names(dataset_dir=dataset_dir)
    num_episodes = len(filepaths)
    print(f"{num_episodes} episodes\n")

    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(filepaths=filepaths)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(chunk_size, train_indices, filepaths, camera_names, norm_stats)
    val_dataset = EpisodicDataset(chunk_size, val_indices, filepaths, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim, camera_names


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