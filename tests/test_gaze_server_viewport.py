import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
EYETRAX_SRC = ROOT_DIR / "eyetrax" / "src"

sys.path.insert(0, str(EYETRAX_SRC))

import eyetrax.server as server_module


class FakeGazeEstimator:
    def __init__(self, *args, **kwargs):
        self.last_train_targets = None

    def reset_movement_center(self):
        return None

    def train(self, X, y):
        self.last_train_targets = y

    def close(self):
        return None


class GazeServerViewportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        gaze_patcher = patch.object(server_module, "GazeEstimator", FakeGazeEstimator)
        self.addCleanup(gaze_patcher.stop)
        gaze_patcher.start()

        sleep_patcher = patch.object(server_module.asyncio, "sleep", AsyncMock())
        self.addCleanup(sleep_patcher.stop)
        sleep_patcher.start()

        self.server = server_module.GazeServer(filter_method="none")
        self.server._broadcast = AsyncMock()
        self.server._wait_for_face = AsyncMock(return_value=True)
        self.server._capture_point = AsyncMock(
            return_value=[np.array([0.1, 0.2, 0.3], dtype=np.float32)]
        )
        self.server._build_calibration_quality = Mock(
            return_value={
                "ok": True,
                "score": 100,
                "message": "ok",
                "horizontal_spread": 1.0,
                "vertical_spread": 1.0,
                "mean_error_px": 0.0,
                "sample_count": 1,
            }
        )

    async def test_calibration_uses_viewport_from_starting_client(self):
        patient_client = object()
        stale_client = object()

        await self.server._handle_message(
            stale_client,
            json.dumps(
                {
                    "type": "set_screen_size",
                    "width": 1920,
                    "height": 1080,
                }
            ),
        )
        await self.server._handle_message(
            patient_client,
            json.dumps(
                {
                    "type": "set_screen_size",
                    "width": 1280,
                    "height": 720,
                }
            ),
        )

        await self.server._run_calibration(patient_client, method="5p")

        self.assertEqual((self.server.screen_width, self.server.screen_height), (1280, 720))
        self.assertIs(self.server._calibration_owner, patient_client)

        targets = self.server.gaze_estimator.last_train_targets
        self.assertIsNotNone(targets)
        self.assertTrue(np.array_equal(targets[0], np.array([640, 360])))


class GazeServerCameraListTests(unittest.TestCase):
    def setUp(self):
        gaze_patcher = patch.object(server_module, "GazeEstimator", FakeGazeEstimator)
        self.addCleanup(gaze_patcher.stop)
        gaze_patcher.start()

        self.server = server_module.GazeServer(filter_method="none")

    def test_refresh_available_cameras_preserves_existing_sources_when_active_camera_is_open(self):
        self.server.available_camera_indices = [0, 1, 2]
        self.server.camera_index = 1
        self.server.cap = Mock()
        self.server.cap.isOpened.return_value = True

        with patch.object(server_module, "list_available_cameras", return_value=[1]):
            self.server._refresh_available_cameras()

        self.assertEqual(self.server.available_camera_indices, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
