from __future__ import annotations

import glob
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CameraSensor:
    def __init__(self, camera_index: int | str = 0):
        self.camera_index = camera_index
        self._capture: Any | None = None

    @staticmethod
    def _available_camera_devices() -> list[str]:
        return sorted(glob.glob("/dev/video*"))

    def _get_capture(self):
        if self._capture is None:
            try:
                import cv2
            except ImportError as exc:  # pragma: no cover - requires optional dependency
                raise RuntimeError(
                    "OpenCV (cv2) is required for camera capture but is not installed."
                ) from exc
            self._capture = cv2.VideoCapture(self.camera_index)
            if not self._capture.isOpened():  # pragma: no cover - hardware dependent
                # Reset cached capture so we can retry if the configuration changes
                self._capture.release()
                self._capture = None
                devices = self._available_camera_devices()
                device_hint = (
                    f"Available video devices: {', '.join(devices)}."
                    if devices
                    else "No /dev/video* devices were detected."
                )
                raise RuntimeError(
                    "Unable to open camera index "
                    f"{self.camera_index}. Ensure the camera is connected or set "
                    "PI_CAMERA_INDEX to the correct device. "
                    + device_hint
                )
        return self._capture

    def capture_frame(self) -> np.ndarray:
        capture = self._get_capture()
        ret, frame = capture.read()
        if not ret:  # pragma: no cover - hardware dependent
            raise RuntimeError("Failed to read frame from camera")
        try:
            import cv2
        except ImportError:  # pragma: no cover - optional dependency
            cv2 = None
        if cv2 is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame.astype(np.uint8)


class MicrophoneSensor:
    def __init__(self, device: str | int | None = None, sample_rate: int = 16000):
        self.device = device
        self.sample_rate = sample_rate

    def capture_window(self, duration_seconds: float = 1.0) -> np.ndarray:
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - requires optional dependency
            raise RuntimeError(
                "sounddevice is required for microphone capture but is not installed."
            ) from exc

        num_samples = int(self.sample_rate * duration_seconds)
        waveform = sd.rec(
            num_samples,
            samplerate=self.sample_rate,
            channels=1,
            device=self.device,
            dtype="float32",
            blocking=True,
        )
        if waveform.ndim > 1:
            waveform = waveform[:, 0]
        return waveform.astype(np.float32)


@dataclass
class MockSensorPair:
    camera_frame: np.ndarray
    audio_waveform: np.ndarray
    sample_rate: int = 16000

    def build_camera(self) -> CameraSensor:
        class _MockCamera(CameraSensor):
            def __init__(self, frame: np.ndarray):
                self.frame = frame

            def capture_frame(self) -> np.ndarray:
                return np.array(self.frame, copy=True)

        return _MockCamera(self.camera_frame)

    def build_microphone(self) -> MicrophoneSensor:
        class _MockMicrophone(MicrophoneSensor):
            def __init__(self, waveform: np.ndarray, sample_rate: int):
                self.waveform = waveform
                self.sample_rate = sample_rate

            def capture_window(self, duration_seconds: float = 1.0) -> np.ndarray:
                return np.array(self.waveform, copy=True)

        return _MockMicrophone(self.audio_waveform, self.sample_rate)
