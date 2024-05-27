#
# extract_episodes_2.py
# Bart Trzynadlowski
#
# Given recorded episodes produced by the robot-arm server, this script copies out the hdf5 files
# into a sequentially numbered series of episode files (episode_1.hdf5, episode_2.hdf5, ...) as
# expected by imitate_episodes.py.
#
# The shortest episode length is used as the length for all episodes. Episodes that are longer are
# split into multiple episodes with as little overlap as possible.
#

import argparse
import os

import h5py
import numpy as np

# Given observations of length total_data_length, produces N starting indices from which
# episode_length can safely be taken. E.g., if episode_length=135:
#
# total_data_length=135, indices=[0]
# total_data_length=239, indices=[0, 104]
# total_data_length=359, indices=[0, 135, 224]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("extract_episodes")
    parser.add_argument("--dir", metavar="path", action="store", required=True, type=str, help="Directory of robot-arm data recording")
    parser.add_argument("--pad", metavar="strategy", action="store", type=str, help="Pad each episode to length of longest episode ('zero' or 'last')")
    parser.add_argument("--truncate", action="store_true", help="Truncate each episode to the length of the smallest episodes")
    parser.add_argument("--fixed-length", action="store", type=int, help="Length to force each episode to, truncating those above and padding those above")
    options = parser.parse_args()

    if options.pad:
        assert options.pad in [ "zero", "last" ]

    print("Analyzing episodes...")
    dirs = [ os.path.join(options.dir, dir) for dir in os.listdir(options.dir) if dir.startswith("example-") and os.path.isdir(os.path.join(options.dir, dir)) ]
    files = []
    max_length = 0
    min_length = 1e9
    for i in range(len(dirs)):
        dir = dirs[i]
        file = os.path.join(dir, "data.h5")
        files.append(file)
        with h5py.File(name=file, mode="r") as fp:
            min_length = min(min_length, len(fp["/action"]))
            max_length = max(max_length, len(fp["/action"]))
    print(f"  - Minimum episode length: {min_length}")
    print(f"  - Maximum episode length: {max_length}")

    print(f"Extracting episodes of length {min_length}...")
    i = 0
    for file in files:
        with h5py.File(name=file, mode="r") as fp:
            # Determine how many sub-episodes we need to create and what their start indices are
            current_length = fp["/action"].shape[0]
            start_indices = produce_episode_start_indices(total_data_length=current_length, episode_length=min_length)

            # Create a file for each
            for idx in start_indices:
                dest_file = os.path.join(options.dir, f"episode_{i}.hdf5")
                i += 1

                actions = fp["/action"][idx:idx+min_length]
                qpos = fp["/observations/qpos"][idx:idx+min_length]
                qvel = fp["/observations/qvel"][idx:idx+min_length]
                images = fp["/observations/images/top"][idx:idx+min_length]

                with h5py.File(name=dest_file, mode="w", rdcc_nbytes=1024**2*2) as root:
                    root.attrs['sim'] = False   # TODO: is this needed?
                    follower = root.create_group("observations")
                    follower.create_dataset(name="qpos", shape=qpos.shape)
                    follower.create_dataset(name="qvel", shape=qvel.shape)
                    camera_images = follower.create_group("images")
                    camera_images.create_dataset(name="top", shape=images.shape, dtype="uint8", chunks=(1, *images.shape[1:]))
                    action = root.create_dataset(name="action", shape=actions.shape)
                    root["/observations/qpos"][...] = qpos
                    root["/observations/qvel"][...] = qvel
                    root["/observations/images/top"][...] = images
                    root["/action"][...] = actions
                    new_length = len(actions)
                    print(f"  - Copied: {file} [{idx}:{idx+min_length}] -> {dest_file}")

            
      