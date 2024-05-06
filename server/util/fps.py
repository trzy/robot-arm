#
# fps.py
#
# Frame rate estimator.
#

import time


class FrameRateCalculator:
    _last_frame_time: float = None
    fps: float = 0

    def record_frame(self) -> float:
        timestamp = time.perf_counter()
        if self._last_frame_time is not None:
            self.fps = 1.0 / (timestamp - self._last_frame_time)
        self._last_frame_time = timestamp