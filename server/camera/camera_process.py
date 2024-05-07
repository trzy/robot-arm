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
        capture = cv2.VideoCapture(index=camera_idx, apiPreference=cv2.CAP_AVFOUNDATION)
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
    
    @staticmethod
    def _run_display_window(fps: Synchronized, terminate: Synchronized):
        frame_provider = CameraFrameProvider()
        while not terminate.value:
            frame = frame_provider.get_frame_buffer()
            cv2.putText(frame, "%1.0f" % fps.value, org = (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 255, 255), thickness = 2, lineType = cv2.LINE_AA)
            cv2.imshow("Camera", frame)
            cv2.waitKey(1)

