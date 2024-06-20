#
# camera_process.py
# Bart Trzynadlowski
#
# Camera process. Acquires frames as fast as possible, hopefully at a steady rate, and writes them
# to shared memory for readers.
#

from multiprocessing import Process, shared_memory, Value
from multiprocessing.sharedctypes import Synchronized
import time
from typing import List

import cv2
import numpy as np

from ..util import FrameRateCalculator, SquareTiling


class CameraFrameProvider:
    def __init__(self, num_cameras: int):
        self._num_cameras = num_cameras
        self._memory: shared_memory.SharedMemory | None = None

    def __del__(self):
        if self._memory is not None:
            self._memory.close()

    def get_frame_buffer(self) -> np.ndarray:
        if self._memory is None:
            # Lazy instantiate only when this provider is used
            self._memory = shared_memory.SharedMemory(name="camera_frame_buffer", create=False)
        return np.ndarray(shape=(self._num_cameras,480,640,3), dtype=np.uint8, buffer=self._memory.buf, order="C")

class CameraProcess:
    def __init__(self, camera_idxs: List[int]):
        # Synchronized flag to terminate sub-processes
        self._terminate = Value("b", False)

        # Synchronized frame rate, written by acquisition process
        self._fps = Value("d", 0.0)

        # Shared memory for frame buffer
        self.num_cameras = len(camera_idxs)
        try:
            self._memory = shared_memory.SharedMemory(name="camera_frame_buffer", create=True, size=640 * 480 * 3 * self.num_cameras)
        except FileExistsError:
            self._memory = shared_memory.SharedMemory(name="camera_frame_buffer")
        self._memory_buffer = self._memory.buf

        # Camera frame acquisition process
        print("Starting camera frame acquisition process...")
        frame_acquisition_process_args = (camera_idxs, self._fps, self._terminate)
        self._frame_acquisition_process = Process(target=CameraProcess._run_frame_acquisition, args=frame_acquisition_process_args)
        self._frame_acquisition_process.start()

        # Camera display window process
        print("Starting camera display window process...")
        display_process_args = (self.num_cameras, self._fps, self._terminate)
        self._display_process = Process(target=CameraProcess._run_display_window, args=display_process_args)
        self._display_process.start()

    def __del__(self):
        # Signal termination using last byte in buffer
        self._terminate_subprocesses()
        self._frame_acquisition_process.join()
        self._display_process.join()
        self._memory.close()
        self._memory.unlink()
        print("Terminated camera process")

    def create_frame_provider(self) -> CameraFrameProvider:
        return CameraFrameProvider(num_cameras=self.num_cameras)

    def _terminate_subprocesses(self):
        self._terminate.value = True

    @staticmethod
    def _run_frame_acquisition(camera_idxs: List[int], fps: Synchronized, terminate: Synchronized):
        num_cameras = len(camera_idxs)
        single_frame_size = 640 * 480 * 3
        fps_calculator = FrameRateCalculator()
        memory = shared_memory.SharedMemory(name="camera_frame_buffer")
        captures: List[cv2.VideoCapture] = []
        for i in range(num_cameras):
            capture = cv2.VideoCapture(index=camera_idxs[i])
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            captures.append(capture)
        while not terminate.value:
            all_succeeded = True
            for i in range(num_cameras):
                success, frame = captures[i].read()
                all_succeeded = all_succeeded and success
                memory.buf[i * single_frame_size : (i + 1) * single_frame_size] = frame.flatten()[:]
            fps_calculator.record_frame()
            fps.value = fps_calculator.fps
            if not all_succeeded:
                time.sleep(1.0 / 30.0)
                continue
        memory.close()
        capture.release()

    @staticmethod
    def _run_display_window(num_cameras: int, fps: Synchronized, terminate: Synchronized):
        window_name = "Camera" if num_cameras == 1 else "Cameras"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse)
        frame_provider = CameraFrameProvider(num_cameras=num_cameras)
        tiling = SquareTiling(num_tiles=num_cameras)
        all_cameras = np.zeros((480 * tiling.height, 640 * tiling.width, 3), dtype=np.uint8)
        while not terminate.value:
            frames = frame_provider.get_frame_buffer().copy()   # (N,480,640,3)
            for i in range(num_cameras):
                tile_coord = tiling.index_to_coordinate(idx=i)
                x = tile_coord[0] * 640
                y = tile_coord[1] * 480
                all_cameras[y : y + 480, x : x + 640, :] = frames[i, :, :, :]
            cv2.putText(all_cameras, "%1.0f" % fps.value, org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            draw_calibration_target(frame=all_cameras)
            cv2.imshow(window_name, all_cameras)
            cv2.waitKey(1)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Calibration point: (x,y)=({x},{y})")

def draw_calibration_target(frame: np.ndarray):
    # These points were obtained by clicking on the camera window at corners of the robot arm base
    robot_l_path = [
        (299,275),
        (343,296),
        (373,270)
    ]
    robot_r_path = [
        (872,306),
        (932,299),
        (951,319),
        (1002,312)
    ]
    coaster_l_path = [
        (335,409),
        (417,332),
        (547,392),
        (494,478)
    ]
    coaster_r_path = [
        (1066,301),
        (1158,285),
        (1227,322),
        (1129,348)
    ]
    paths = [ robot_l_path, robot_r_path, coaster_l_path, coaster_r_path ]
    for path in paths:
        num_points = len(path)
        for i in range(num_points - 1):
            cv2.line(img=frame, pt1=path[i], pt2=path[i+1], color=(0, 255, 0), thickness=1)