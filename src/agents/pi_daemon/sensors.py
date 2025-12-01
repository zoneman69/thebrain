from __future__ import annotations

import glob
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CameraSensor:
    def __init__(
        self,
        camera_index: int | str = 0,
        retry_attempts: int = 3,
        retry_delay_seconds: float = 2.0,
    ):
        self.camera_index = camera_index
        self.retry_attempts = max(1, int(retry_attempts))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self._capture: Any | None = None

    def close(self) -> None:
        if self._capture is not None:  # pragma: no cover - hardware dependent
            self._capture.release()
            self._capture = None

    @staticmethod
    def _available_camera_devices() -> list[str]:
        return sorted(glob.glob("/dev/video*"))

    @staticmethod
    def _open_capture(cv2: Any, index: int | str):
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            return capture
        capture.release()
        return None

    def _get_capture(self):
        if self._capture is None:
            try:
                import cv2
            except ImportError as exc:  # pragma: no cover - requires optional dependency
                raise RuntimeError(
                    "OpenCV (cv2) is required for camera capture but is not installed."
                ) from exc
            self._capture = self._open_capture(cv2, self.camera_index)
            if self._capture is None:  # pragma: no cover - hardware dependent
                devices = self._available_camera_devices()
                for device in devices:
                    if device == self.camera_index:
                        continue
                    self._capture = self._open_capture(cv2, device)
                    if self._capture is not None:
                        logger.warning(
                            "Unable to open camera %s; falling back to %s",
                            self.camera_index,
                            device,
                        )
                        self.camera_index = device
                        break

                if self._capture is None:
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
        attempts_remaining = self.retry_attempts
        while not ret and attempts_remaining > 0:  # pragma: no cover - hardware dependent
            attempts_remaining -= 1
            logger.warning(
                "Failed to read frame from camera %s; attempting to reinitialize (%d retries left)",
                self.camera_index,
                attempts_remaining,
            )
            self.close()
            if self.retry_delay_seconds:
                time.sleep(self.retry_delay_seconds)
            capture = self._get_capture()
            ret, frame = capture.read()

        if not ret:
            raise RuntimeError("Failed to read frame from camera after reinitialization")
        try:
            import cv2
        except ImportError:  # pragma: no cover - optional dependency
            cv2 = None
        if cv2 is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame.astype(np.uint8)


class FrameFileSensor:
    def __init__(
        self,
        frame_path: str | os.PathLike[str],
        retry_attempts: int = 3,
        retry_delay_seconds: float = 2.0,
    ):
        self.frame_path = Path(frame_path)
        self.retry_attempts = max(1, int(retry_attempts))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))

    def close(self) -> None:
        # No resources to free, but we keep this method for API compatibility
        # with CameraSensor so callers can always call cam.close().
        return None

    def capture_frame(self) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - requires optional dependency
            raise RuntimeError(
                "OpenCV (cv2) is required for frame decoding but is not installed."
            ) from exc

        attempts_remaining = self.retry_attempts
        while attempts_remaining >= 0:  # pragma: no cover - hardware/filesystem dependent
            if not self.frame_path.exists():
                logger.warning("Frame file %s is missing; waiting for writer", self.frame_path)
            else:
                frame = cv2.imread(str(self.frame_path))
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return frame.astype(np.uint8)
                logger.warning("Failed to decode frame file %s; retrying", self.frame_path)

            attempts_remaining -= 1
            if attempts_remaining < 0:
                break
            if self.retry_delay_seconds:
                time.sleep(self.retry_delay_seconds)

        raise RuntimeError(
            f"Failed to read frame from {self.frame_path}. Ensure the producer is writing a valid image."
        )


class SpectrogramFileSensor:
    """
    Reads a spectrogram image (e.g., HIPPO_SPEC PNG) and returns it as a
    2D float32 array for audio encoding.
    """

    def __init__(
        self,
        spec_path: str | os.PathLike[str],
        retry_attempts: int = 3,
        retry_delay_seconds: float = 2.0,
    ):
        self.spec_path = Path(spec_path)
        self.retry_attempts = max(1, int(retry_attempts))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))

    def close(self) -> None:
        # No hardware resources to free, but keep API shape consistent.
        return None

    def capture_spectrogram(self) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - requires optional dependency
            raise RuntimeError(
                "OpenCV (cv2) is required for spectrogram decoding but is not installed."
            ) from exc

        attempts_remaining = self.retry_attempts
        while attempts_remaining >= 0:  # pragma: no cover - filesystem dependent
            if not self.spec_path.exists():
                logger.warning("Spectrogram file %s is missing; waiting for writer", self.spec_path)
            else:
                img = cv2.imread(str(self.spec_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return img.astype(np.float32)
                logger.warning(
                    "Failed to decode spectrogram file %s; retrying", self.spec_path
                )

            attempts_remaining -= 1
            if attempts_remaining < 0:
                break
            if self.retry_delay_seconds:
                time.sleep(self.retry_delay_seconds)

        raise RuntimeError(
            f"Failed to read spectrogram from {self.spec_path}. "
            "Ensure the producer is writing a valid spectrogram image."
        )


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
                self._capture = None

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
