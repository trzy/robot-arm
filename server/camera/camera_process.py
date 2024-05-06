#
# camera_process.py
# Bart Trzynadlowski
#
# Camera process. Acquires frames as fast as possible, hopefully at a steady rate, and writes them
# to shared memory for readers.
#

from multiprocessing import Process, shared_memory
import time

import cv2
import numpy as np

from ..util import FrameRateCalculator


class CameraProcess:
    def __init__(self, camera_idx: int):
        try:
            self._memory = shared_memory.SharedMemory(name="camera_frame_buffer", create=True, size=640*480*3 + 1)
        except FileExistsError:
            self._memory = shared_memory.SharedMemory(name="camera_frame_buffer")
        self._memory_buffer = self._memory.buf
        self._set_termination_flag(terminate=False)
        print("Starting camera process...")
        process_args = (camera_idx,)
        self._process = Process(target=CameraProcess._run, args=process_args)
        self._process.start()

    def __del__(self):
        # Signal termination using last byte in buffer
        self._set_termination_flag(terminate=True)
        self._process.join()
        self._memory.close()
        self._memory.unlink()
        print("Terminated camera process")

    def get_frame_buffer(self) -> np.ndarray:
        return np.ndarray(shape=(480,640,3), dtype=np.uint8, buffer=self._memory_buffer, order="C")

    def _set_termination_flag(self, terminate: bool):
        self._memory_buffer[640*480*3 + 0] = 1 if terminate else 0

    @staticmethod
    def _terminate(memory: memoryview):
        return memory[640*480*3+0] != 0

    @staticmethod
    def _run(camera_idx: int):
        fps = FrameRateCalculator()
        memory = shared_memory.SharedMemory(name="camera_frame_buffer")
        capture = cv2.VideoCapture(index=camera_idx, apiPreference=cv2.CAP_AVFOUNDATION)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while not CameraProcess._terminate(memory=memory.buf):
            success, frame = capture.read()
            fps.record_frame()
            if not success:
                time.sleep(1.0/30.0)
                continue
            cv2.putText(frame, "%1.1f" % fps.fps, org = (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 255, 255), thickness = 2, lineType = cv2.LINE_AA)
            memory.buf[0:640*480*3] = frame.flatten()[:]
        memory.close()