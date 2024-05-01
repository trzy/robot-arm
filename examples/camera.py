import time

import cv2
import depthai as dai

class FrameRateCalculator:
    _last_frame_time: float = None
    fps: float = 0

    def record_frame(self) -> float:
        timestamp = time.perf_counter()
        if self._last_frame_time is not None:
            self.fps = 1.0 / (timestamp - self._last_frame_time)
        self._last_frame_time = timestamp

fps = FrameRateCalculator()

# Create pipeline
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setPreviewSize(640,480)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("rgb")
cam.preview.link(xout.input)
cam.setFps(fps=30)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    while True:
        frame = q.get().getFrame()
        fps.record_frame()
        cv2.putText(frame, "%1.1f" % fps.fps, org = (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 255, 255), thickness = 2, lineType = cv2.LINE_AA)
        cv2.imshow("RGB Camera", frame)
        cv2.waitKey(delay=1)    # this 1ms delay is necessary to process the window on macOS (see https://github.com/justadudewhohacks/opencv4nodejs/issues/412)