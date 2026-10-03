"""Video decoding and metadata, separate from scientific processing."""
from pathlib import Path
import cv2
import numpy as np


class Video:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            self.cap.release()
            raise ValueError(f"Cannot open video: {self.path}")
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        value = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = value if np.isfinite(value) and value > 0 else None
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.n_frames <= 0:
            self.cap.release()
            raise ValueError("Video has no reported frames")
        self._next_index = 0

    def read(self, index):
        if not 0 <= index < self.n_frames:
            raise IndexError(f"Frame {index} outside video")
        if index != self._next_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            self._next_index = -1
            raise ValueError(f"Could not decode frame {index}")
        self._next_index = index + 1
        return frame

    def metadata(self):
        return {"path": str(self.path), "n_frames": self.n_frames,
                "fps_reported": self.fps, "width": self.width, "height": self.height,
                "time_basis": "frame / reported FPS; assumes constant frame rate",
                "spatial_calibration_um_per_px": None}

    def close(self):
        self.cap.release()
