import argparse
import time

import cv2


class FrameRateCalculator:
    _last_frame_time: float = None
    fps: float = 0

    def record_frame(self) -> float:
        timestamp = time.perf_counter()
        if self._last_frame_time is not None:
            self.fps = 1.0 / (timestamp - self._last_frame_time)
        self._last_frame_time = timestamp

def main(camera_idx: int):
    # Initialize the webcam
    cap = cv2.VideoCapture(index=camera_idx, apiPreference=cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set the window name
    window_name = "Webcam"

    fps = FrameRateCalculator()

    i = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        fps.record_frame()
        i += 1
        if i % 60 == 0:
            print(fps.fps)

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Cannot receive frame (stream end?). Exiting...")
            break

        # Display the captured frame
        cv2.putText(frame, "%1.1f" % fps.fps, org = (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 255, 255), thickness = 2, lineType = cv2.LINE_AA)
        cv2.imshow(window_name, frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("robotest")
    parser.add_argument("--camera", metavar="index", action="store", type=int, required=True, help="Camera to use")
    options = parser.parse_args()
    main(camera_idx=options.camera)