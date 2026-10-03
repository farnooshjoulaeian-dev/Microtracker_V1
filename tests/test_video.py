import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
from video import Video


class VideoTests(unittest.TestCase):
    def capture(self, fps=25):
        cap = Mock()
        cap.isOpened.return_value = True
        props = {cv2.CAP_PROP_FRAME_COUNT: 3, cv2.CAP_PROP_FPS: fps,
                 cv2.CAP_PROP_FRAME_WIDTH: 20, cv2.CAP_PROP_FRAME_HEIGHT: 10}
        cap.get.side_effect = lambda key: props.get(key, 0)
        cap.read.return_value = (True, np.zeros((10, 20, 3), dtype=np.uint8))
        return cap

    def test_sequential_reads_do_not_seek_but_navigation_does(self):
        cap = self.capture()
        with patch("video.cv2.VideoCapture", return_value=cap):
            video = Video("fixture.mp4")
            video.read(0); video.read(1)
            cap.set.assert_not_called()
            video.read(0)
            cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 0)
            with self.assertRaises(IndexError):
                video.read(3)
            video.close(); cap.release.assert_called_once()

    def test_unknown_fps_does_not_invent_measurement_time(self):
        with patch("video.cv2.VideoCapture", return_value=self.capture(fps=0)):
            video = Video("fixture.mp4")
            self.assertIsNone(video.fps)
            self.assertIsNone(video.metadata()["spatial_calibration_um_per_px"])

    def test_decode_failure_is_reported(self):
        cap = self.capture(); cap.read.return_value = (False, None)
        with patch("video.cv2.VideoCapture", return_value=cap):
            with self.assertRaises(ValueError):
                Video("fixture.mp4").read(0)


if __name__ == "__main__":
    unittest.main()
