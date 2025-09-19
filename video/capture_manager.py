import cv2
import time
from typing import Tuple, Any

class CaptureManager:
    """
    Thin wrapper over cv2.VideoCapture that enforces an approximate target FPS
    by pacing reads with sleep. Use set_target_fps to adjust at runtime.
    """
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480, target_fps: float = 30.0):
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera")
        # Try to set properties (not all cams honor these)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        self._target_fps = max(1.0, float(target_fps))
        self._min_interval = 1.0 / self._target_fps
        self._last_return_time = 0.0

    def set_target_fps(self, fps: float):
        fps = max(1.0, float(fps))
        self._target_fps = fps
        self._min_interval = 1.0 / fps
        # also hint camera driver
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self) -> Tuple[bool, Any]:
        # Pace reads to not exceed target FPS
        now = time.time()
        elapsed = now - self._last_return_time
        if self._last_return_time > 0 and elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        ret, frame = self.cap.read()
        self._last_return_time = time.time()
        return ret, frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass
