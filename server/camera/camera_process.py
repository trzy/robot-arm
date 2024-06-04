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

import cv2
import numpy as np

from ..util import FrameRateCalculator


class CameraFrameProvider:
    def __init__(self):
        self._memory: shared_memory.SharedMemory | None = None
    
    def __del__(self):
        if self._memory is not None:
            self._memory.close()
    
    def get_frame_buffer(self) -> np.ndarray:
        if self._memory is None:
            # Lazy instantiate only when this provider is used
            self._memory = shared_memory.SharedMemory(name="camera_frame_buffer", create=False)
        return np.ndarray(shape=(480,640,3), dtype=np.uint8, buffer=self._memory.buf, order="C")

class CameraProcess:
    def __init__(self, camera_idx: int):
        # Synchronized flag to terminate sub-processes
        self._terminate = Value("b", False)

        # Synchronized frame rate, written by acquisition process
        self._fps = Value("d", 0.0)

        # Shared memory for frame buffer
        try:
            self._memory = shared_memory.SharedMemory(name="camera_frame_buffer", create=True, size=640*480*3)
        except FileExistsError:
            self._memory = shared_memory.SharedMemory(name="camera_frame_buffer")
        self._memory_buffer = self._memory.buf

        # Camera frame acquisition process
        print("Starting camera frame acquisition process...")
        frame_acquisition_process_args = (camera_idx, self._fps, self._terminate)
        self._frame_acquisition_process = Process(target=CameraProcess._run_frame_acquisition, args=frame_acquisition_process_args)
        self._frame_acquisition_process.start()

        # Camera display window process
        print("Starting camera display window process...")
        display_process_args = (self._fps, self._terminate)
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

    def get_frame_buffer(self) -> np.ndarray:
        return np.ndarray(shape=(480,640,3), dtype=np.uint8, buffer=self._memory_buffer, order="C")

    def _terminate_subprocesses(self):
        self._terminate.value = True

    @staticmethod
    def _run_frame_acquisition(camera_idx: int, fps: Synchronized, terminate: Synchronized):
        fps_calculator = FrameRateCalculator()
        memory = shared_memory.SharedMemory(name="camera_frame_buffer")
        capture = cv2.VideoCapture(index=camera_idx)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while not terminate.value:
            success, frame = capture.read()
            fps_calculator.record_frame()
            fps.value = fps_calculator.fps
            if not success:
                time.sleep(1.0/30.0)
                continue
            memory.buf[0:640*480*3] = frame.flatten()[:]
        memory.close()
        capture.release()
    
    @staticmethod
    def _run_display_window(fps: Synchronized, terminate: Synchronized):
        cv2.namedWindow("Camera")
        cv2.setMouseCallback("Camera", on_mouse)
        frame_provider = CameraFrameProvider()
        while not terminate.value:
            frame = frame_provider.get_frame_buffer().copy()
            cv2.putText(frame, "%1.0f" % fps.value, org = (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 255, 255), thickness = 2, lineType = cv2.LINE_AA)
            draw_calibration_target(frame=frame)
            cv2.imshow("Camera", frame)
            cv2.waitKey(1)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Calibration point: (x,y)=({x},{y})")

def draw_calibration_target(frame: np.ndarray):
    # These points were obtained by clicking on the camera window at corners of the robot arm base
    robot_path = [
        # F.inc
        # (211,327),
        # (274,302),
        # (317,330),
        # (364,310)

        # Apartment
        # (260,206),
        # (320,209),
        # (320,244),
        # (370,248),
        # (371,236),
        # (372,211),
        # (429,214)

        # F.inc w/ Windows laptop, demo day
        (272,342),
        (322,343),
        (320,370),
        (367,371),
        (371,342),
        (422,344),
    ]
    orange_bin_path = [
        # F.inc w/ Windows laptop, demo day
        (10,398),
        (151,422)
    ]
    blue_bin_path = [
        # F.inc w/ Windows laptop, demo day
        (564,451),
        (639,441)
    ]
    paths = [ robot_path, orange_bin_path, blue_bin_path ]
    for path in paths:
        num_points = len(path)
        for i in range(num_points - 1):
            cv2.line(img=frame, pt1=path[i], pt2=path[i+1], color=(0, 255, 0), thickness=1)