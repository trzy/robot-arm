#
# extract_episodes.py
# Bart Trzynadlowski
#
# Given recorded episodes produced by the robot-arm server, this script copies out the hdf5 files
# into a sequentially numbered series of episode files (episode_1.hdf5, episode_2.hdf5, ...) as
# expected by imitate_episodes.py.
#

import argparse
import os

import h5py
import numpy as np

def pad(dataset, new_length, strategy: str):
  length = dataset.shape[0]
  if strategy == "last":
    pad_element = dataset[-1]
  else:
    pad_element = np.zeros(dataset[-1].shape)
  pad_values = np.repeat(pad_element[np.newaxis, :], new_length - length, axis=0)
  dataset = dataset[:]
  dataset.resize((new_length, *dataset.shape[1:]))
  dataset[length:] = pad_values
  return dataset

def truncate(dataset, new_length):
  dataset = dataset[:]
  return dataset[:new_length]

def force_length(dataset, new_length, pad_strategy: str):
  length = dataset.shape[0]
  if length > new_length:
    return truncate(dataset=dataset, new_length=new_length)
  else:
    return pad(dataset=dataset, new_length=new_length, strategy=pad_strategy)

if __name__ == "__main__":
  parser = argparse.ArgumentParser("extract_episodes")
  parser.add_argument("--dir", metavar="path", action="store", required=True, type=str, help="Directory of robot-arm data recording")
  parser.add_argument("--pad", metavar="strategy", action="store", default="zero", type=str, help="Pad each episode to length of longest episode ('zero' or 'last')")
  parser.add_argument("--truncate", action="store_true", help="Truncate each episode to the length of the smallest episodes")
  parser.add_argument("--fixed-length", action="store", type=int, help="Length to force each episode to, truncating those above and padding those above")
  options = parser.parse_args()

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

  print(f"Extracting and padding episodes to {max_length} observations...")
  for i in range(len(files)):
    file = files[i]
    dest_file = os.path.join(options.dir, f"episode_{i}.hdf5")
    with h5py.File(name=file, mode="r") as fp:
      current_length = fp["/action"].shape[0]
      if options.fixed_length:
        # Set to fixed length
        actions = force_length(dataset=fp["/action"], new_length=options.fixed_length, pad_strategy=options.pad)
        qpos = force_length(dataset=fp["/observations/qpos"], new_length=options.fixed_length, pad_strategy=options.pad)
        qvel = force_length(dataset=fp["/observations/qvel"], new_length=options.fixed_length, pad_strategy=options.pad)
        images = force_length(dataset=fp["/observations/images/top"], new_length=options.fixed_length, pad_strategy=options.pad)
      elif options.truncate:
        # Truncate to min
        actions = truncate(dataset=fp["/action"], new_length=min_length)
        qpos = truncate(dataset=fp["/observations/qpos"], new_length=min_length)
        qvel = truncate(dataset=fp["/observations/qvel"], new_length=min_length)
        images = truncate(dataset=fp["/observations/images/top"], new_length=min_length)
      else:
        # Pad to max
        actions = pad(dataset=fp["/action"], new_length=max_length, strategy=options.pad)
        qpos = pad(dataset=fp["/observations/qpos"], new_length=max_length, strategy=options.pad)
        qvel = pad(dataset=fp["/observations/qvel"], new_length=max_length, strategy=options.pad)
        images = pad(dataset=fp["/observations/images/top"], new_length=max_length, strategy=options.pad)
      with h5py.File(name=dest_file, mode="w", rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = False   # TODO: is this needed?
        follower = root.create_group("observations")
        qpos = follower.create_dataset(name="qpos", shape=qpos.shape)
        qvel = follower.create_dataset(name="qvel", shape=qvel.shape)
        camera_images = follower.create_group("images")
        camera_images.create_dataset(name="top", shape=images.shape, dtype="uint8", chunks=(1, *images.shape[1:]))
        action = root.create_dataset(name="action", shape=actions.shape)
        root["/observations/qpos"][...] = qpos
        root["/observations/qvel"][...] = qvel
        root["/observations/images/top"][...] = images
        root["/action"][...] = actions
        new_length = len(actions)
        print(f"  - Copied: {file} -> {dest_file} and extended from {current_length} -> {new_length} observations")