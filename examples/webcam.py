import argparse
import time
from typing import List

import cv2
import numpy as np

from server.util import SquareTiling


class FrameRateCalculator:
    _last_frame_time: float = None
    fps: float = 0

    def record_frame(self) -> float:
        timestamp = time.perf_counter()
        if self._last_frame_time is not None:
            self.fps = 1.0 / (timestamp - self._last_frame_time)
        self._last_frame_time = timestamp

def main(camera_idxs: List[int]):
    # Initialize webcams
    caps: List[cv2.VideoCapture] = []
    for camera_idx in camera_idxs:
        cap = cv2.VideoCapture(index=camera_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print(f"Error: Could not open webcam {camera_idx}")
            return
        caps.append(cap)

    # Set the window name
    window_name = "Webcam" if len(camera_idxs) == 1 else "Webcams"

    # Create a frame buffer large enough for all cameras
    tiling = SquareTiling(num_tiles = len(camera_idxs))
    all_frames = np.zeros((480 * tiling.height, 640 * tiling.width, 3), dtype=np.uint8)

    # Acquire and display frames
    i = 0
    fps = FrameRateCalculator()
    while True:
        # Capture each camera's current frame and write it to the tiled view buffer
        for i in range(len(caps)):
            cap = caps[i]
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Cannot receive frame (stream end?) for camera {i}. Exiting...")
                break
            tile_coord = tiling.index_to_coordinate(idx=i)
            x = tile_coord[0] * 640
            y = tile_coord[1] * 480
            all_frames[y : y + 480, x : x + 640, :] = frame

        # Display the captured frames
        cv2.putText(all_frames, "%1.1f" % fps.fps, org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow(window_name, all_frames)

        # Update frame rate
        fps.record_frame()
        i += 1
        if i % 60 == 0:
            print(fps.fps)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the webcams and close OpenCV window
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("robotest")
    parser.add_argument("--camera", metavar="index", action="store", type=str, required=True, help="Cameras to use")
    options = parser.parse_args()

    camera_idxs = [ int(camera_idx) for camera_idx in options.camera.split(",") ]
    main(camera_idxs=camera_idxs)