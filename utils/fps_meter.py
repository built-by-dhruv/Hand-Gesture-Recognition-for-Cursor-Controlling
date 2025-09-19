import time
from collections import deque

class FPSMeter:
    """
    Simple FPS meter with both instantaneous and smoothed FPS.
    Use tick() once per frame; access fps and smoothed_fps properties.
    """
    def __init__(self, window: int = 30, ema_alpha: float = 0.9):
        self.prev_time = None
        self.fps = 0.0
        self.smoothed_fps = 0.0
        self.ema_alpha = max(0.0, min(ema_alpha, 0.99))
        self.samples = deque(maxlen=max(5, window))

    def tick(self) -> float:
        now = time.time()
        if self.prev_time is None:
            self.prev_time = now
            return 0.0
        dt = now - self.prev_time
        self.prev_time = now
        if dt <= 0:
            return self.smoothed_fps
        inst_fps = 1.0 / dt
        self.fps = inst_fps
        # update rolling average
        self.samples.append(inst_fps)
        avg = sum(self.samples) / len(self.samples)
        # update EMA towards average for stability
        if self.smoothed_fps == 0.0:
            self.smoothed_fps = avg
        else:
            self.smoothed_fps = self.ema_alpha * self.smoothed_fps + (1 - self.ema_alpha) * avg
        return self.smoothed_fps

    def get_int(self) -> int:
        return int(self.smoothed_fps or self.fps or 0)
