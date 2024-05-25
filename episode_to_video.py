import argparse

import cv2
import h5py

parser = argparse.ArgumentParser("episode_to_video")
parser.add_argument("file", nargs=1)
options = parser.parse_args()

with h5py.File(name=options.file[0], mode="r") as fp:
    images = fp["/observations/images/top"]
    height, width, _ = images[0].shape
    video_writer = cv2.VideoWriter(filename="out.mp4", fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=30.0, frameSize=(width,height))
    for image in images:
        video_writer.write(image=image)
    video_writer.release()