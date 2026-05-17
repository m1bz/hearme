"""
HearMe Eye-Tracking WebSocket Bridge Server

Bridges eyetrax (Python-based gaze estimation) with the HearMe React frontend
via WebSocket. Handles calibration, real-time gaze streaming, and recalibration.

Protocol:
  Server → Client:
    { type: "gaze", x, y, blink, moved }
    { type: "calibration_status", status, point_index, total_points, point_x, point_y, phase, progress }
    { type: "face_status", detected, blink }
    { type: "connection", status: "ready" }
    { type: "error", message }

  Client → Server:
    { type: "start_calibration", method: "9p"|"5p" }
    { type: "recalibrate" }
    { type: "set_screen_size", width, height }
"""

import asyncio
import base64
import json
import time
import traceback
from typing import Optional

import cv2
import numpy as np

try:
    import websockets
except ImportError:
    raise ImportError(
        "websockets is required. Install with: pip install websockets"
    )

from eyetrax.gaze import GazeEstimator
from eyetrax.calibration.common import compute_grid_points
from eyetrax.filters import KalmanEMASmoother, KalmanSmoother, NoSmoother, make_kalman
from eyetrax.utils.video import list_available_cameras, open_camera_with_index


class GazeServer:
    """WebSocket server bridging eyetrax gaze estimation to frontend."""

    DEFAULT_SCREEN_WIDTH = 1920
    DEFAULT_SCREEN_HEIGHT = 1080

    def __init__(
        self,
        camera_index: int = 0,
        host: str = "localhost",
        port: int = 8765,
        filter_method: str = "kalman",
        ema_alpha: float = 0.25,
    ):
        self.camera_index = camera_index
        self.host = host
        self.port = port
        self.filter_method = filter_method
        self.ema_alpha = ema_alpha

        self.gaze_estimator = GazeEstimator(model_name="ridge")
        self.smoother = None
        self.cap = None
        self.calibrated = False
        self.running = False
        self.clients = set()
        self.client_viewports = {}

        # Screen size (overridden by frontend)
        self.screen_width = self.DEFAULT_SCREEN_WIDTH
        self.screen_height = self.DEFAULT_SCREEN_HEIGHT

        # Calibration state
        self._calibrating = False
        self._last_gaze_position = None
        self.available_camera_indices = []
        self._calibration_owner = None

    def _make_smoother(self):
        if self.filter_method == "kalman":
            return KalmanSmoother(make_kalman())
        if self.filter_method == "kalman_ema":
            return KalmanEMASmoother(make_kalman(), ema_alpha=self.ema_alpha)
        return NoSmoother()

    async def start(self):
        """Start the WebSocket server."""
        self._refresh_available_cameras()
        self._open_camera(self.camera_index, allow_fallback=True)
        print(f"[gaze-server] Starting WebSocket server on ws://{self.host}:{self.port}")

        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=60,
        ):
            self.running = True
            await asyncio.Future()  # Run forever

    def _refresh_available_cameras(self):
        detected_cameras = list_available_cameras()
        cameras = set(self.available_camera_indices)
        cameras.update(detected_cameras)

        # Keep the active camera in the selector even if probing while it is
        # already open makes OpenCV miss it temporarily.
        if self.cap and self.cap.isOpened():
            cameras.add(self.camera_index)

        if not cameras:
            cameras.add(self.camera_index)

        self.available_camera_indices = sorted(cameras)

    def _connection_payload(self):
        return {
            "type": "connection",
            "status": "ready",
            "calibrated": self.calibrated,
            "camera_index": self.camera_index,
            "available_cameras": self.available_camera_indices,
        }

    def _reset_tracking_state(self):
        self.calibrated = False
        self.gaze_estimator.reset_movement_center()
        self.smoother = None
        self._last_gaze_position = None
        self._calibration_owner = None

    def _sanitize_viewport(self, width, height):
        width = max(1, int(width or self.DEFAULT_SCREEN_WIDTH))
        height = max(1, int(height or self.DEFAULT_SCREEN_HEIGHT))
        return width, height

    def _set_client_viewport(self, websocket, width, height):
        viewport = self._sanitize_viewport(width, height)
        self.client_viewports[websocket] = viewport
        return viewport

    def _get_client_viewport(self, websocket):
        return self.client_viewports.get(
            websocket,
            (self.DEFAULT_SCREEN_WIDTH, self.DEFAULT_SCREEN_HEIGHT),
        )

    def _activate_client_viewport(self, websocket):
        width, height = self._get_client_viewport(websocket)
        self.screen_width = width
        self.screen_height = height
        self._calibration_owner = websocket
        return width, height

    def _open_camera(self, camera_index: int, *, allow_fallback: bool):
        previous_cap = self.cap
        cap, actual_index = open_camera_with_index(
            camera_index,
            allow_fallback=allow_fallback,
        )
        self.cap = cap
        self.camera_index = actual_index
        if previous_cap and previous_cap.isOpened():
            previous_cap.release()
        print(f"[gaze-server] Camera opened (index {self.camera_index})")

    async def _send_connection_state(self, websocket):
        await websocket.send(json.dumps(self._connection_payload()))

    async def _broadcast_connection_state(self):
        await self._broadcast(self._connection_payload())

    async def _handle_client(self, websocket):
        """Handle a new client connection."""
        self.clients.add(websocket)
        self._set_client_viewport(
            websocket,
            self.DEFAULT_SCREEN_WIDTH,
            self.DEFAULT_SCREEN_HEIGHT,
        )
        print(f"[gaze-server] Client connected ({len(self.clients)} total)")

        try:
            self._refresh_available_cameras()
            await self._send_connection_state(websocket)

            # Start gaze loop if calibrated
            if self.calibrated:
                asyncio.ensure_future(self._gaze_loop(websocket))

            async for message in websocket:
                await self._handle_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            self.client_viewports.pop(websocket, None)
            if self._calibration_owner is websocket:
                self._calibration_owner = None
            print(f"[gaze-server] Client disconnected ({len(self.clients)} total)")

    async def _handle_message(self, websocket, raw_message: str):
        """Process incoming client messages."""
        try:
            msg = json.loads(raw_message)
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Invalid JSON",
            }))
            return

        msg_type = msg.get("type")

        if msg_type == "set_screen_size":
            width, height = self._set_client_viewport(
                websocket,
                msg.get("width"),
                msg.get("height"),
            )
            print(
                f"[gaze-server] Viewport for client updated to {width}x{height}"
            )

        elif msg_type == "start_calibration":
            method = msg.get("method", "9p")
            await self._run_calibration(websocket, method)

        elif msg_type == "recalibrate":
            self._reset_tracking_state()
            method = msg.get("method", "9p")
            await self._run_calibration(websocket, method)

        elif msg_type == "reset_calibration":
            # Drop calibration and stop streaming, but don't auto-start a new one.
            # The frontend will return to the pre-calibration screen and the user
            # presses Start to begin a fresh calibration.
            self._reset_tracking_state()
            print("[gaze-server] Calibration reset, awaiting new calibration")
            self._refresh_available_cameras()
            await self._send_connection_state(websocket)

        elif msg_type == "check_face":
            await self._check_face(websocket)

        elif msg_type == "set_camera":
            await self._set_camera(websocket, msg.get("camera_index"))

        else:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Unknown message type: {msg_type}",
            }))

    def _encode_preview(self, frame):
        """Return a small mirrored JPEG preview for the setup screen."""
        preview = cv2.flip(frame, 1)
        preview = cv2.resize(preview, (360, 270), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(
            ".jpg",
            preview,
            [int(cv2.IMWRITE_JPEG_QUALITY), 70],
        )
        if not ok:
            return None
        return base64.b64encode(encoded).decode("ascii")

    def _build_setup_checks(self, metrics, blink: bool):
        if not metrics:
            checks = {
                "face": False,
                "webcam_level": False,
                "centered": False,
                "head_pose": False,
                "eye_line": False,
                "distance": False,
                "lighting": False,
                "eyes_open": False,
            }
            return {
                "ready": False,
                "checks": checks,
                "metrics": None,
            }

        yaw = abs(metrics["yaw_degrees"])
        pitch = abs(metrics["pitch_degrees"])
        roll = abs(metrics["roll_degrees"])
        inter_eye_ratio = metrics["inter_eye_ratio"]
        brightness = metrics["brightness"]

        checks = {
            "face": True,
            "webcam_level": abs(metrics["center_offset_y"]) <= 0.10,
            "centered": (
                abs(metrics["center_offset_x"]) <= 0.12
            ),
            "head_pose": yaw <= 20.0 and pitch <= 20.0,
            "eye_line": roll <= 18.0,
            "distance": 0.13 <= inter_eye_ratio <= 0.34,
            "lighting": 55.0 <= brightness <= 220.0,
            "eyes_open": not blink,
        }

        return {
            "ready": all(checks.values()),
            "checks": checks,
            "metrics": metrics,
        }

    def _build_calibration_quality(self, features, targets):
        """Summarize whether calibration samples create separated gaze outputs."""
        if not features or not targets:
            return {
                "ok": False,
                "score": 0,
                "message": "No calibration samples were captured.",
                "horizontal_spread": 0.0,
                "vertical_spread": 0.0,
                "mean_error_px": None,
                "sample_count": 0,
            }

        try:
            x_values = np.asarray(features)
            target_values = np.asarray(targets, dtype=np.float32)
            predictions = self.gaze_estimator.predict(x_values)

            grouped_predictions = {}
            grouped_targets = {}
            for prediction, target in zip(predictions, target_values):
                key = (int(target[0]), int(target[1]))
                grouped_predictions.setdefault(key, []).append(prediction)
                grouped_targets[key] = target

            predicted_centers = []
            target_centers = []
            for key, group in grouped_predictions.items():
                predicted_centers.append(np.mean(np.asarray(group), axis=0))
                target_centers.append(grouped_targets[key])

            predicted_centers = np.asarray(predicted_centers, dtype=np.float32)
            target_centers = np.asarray(target_centers, dtype=np.float32)
            errors = np.linalg.norm(predicted_centers - target_centers, axis=1)

            horizontal_spread = float(
                (np.max(predicted_centers[:, 0]) - np.min(predicted_centers[:, 0]))
                / max(self.screen_width, 1)
            )
            vertical_spread = float(
                (np.max(predicted_centers[:, 1]) - np.min(predicted_centers[:, 1]))
                / max(self.screen_height, 1)
            )
            mean_error = float(np.mean(errors))

            spread_score = (
                0.45 * min(horizontal_spread / 0.65, 1.0)
                + 0.45 * min(vertical_spread / 0.65, 1.0)
            )
            error_limit = max(max(self.screen_width, self.screen_height) * 0.35, 1)
            error_score = 0.10 * max(0.0, 1.0 - (mean_error / error_limit))
            score = int(round(100 * max(0.0, min(spread_score + error_score, 1.0))))

            ok = (
                score >= 60
                and horizontal_spread >= 0.45
                and vertical_spread >= 0.45
            )
            if ok:
                message = "Corners produced separated gaze data. Continue holding steady for Kalman tuning."
            elif horizontal_spread < 0.45 or vertical_spread < 0.45:
                message = "Corner looks are too similar. Recalibrate with eyes in the blue band and avoid head movement."
            else:
                message = "Calibration is usable but noisy. Recalibrate if gaze feels jumpy."

            return {
                "ok": ok,
                "score": score,
                "message": message,
                "horizontal_spread": round(horizontal_spread, 3),
                "vertical_spread": round(vertical_spread, 3),
                "mean_error_px": round(mean_error, 1),
                "sample_count": int(len(features)),
            }
        except Exception as exc:
            return {
                "ok": False,
                "score": 0,
                "message": f"Calibration quality check failed: {exc}",
                "horizontal_spread": 0.0,
                "vertical_spread": 0.0,
                "mean_error_px": None,
                "sample_count": int(len(features)),
            }

    async def _check_face(self, websocket):
        """Check if a face is currently detected."""
        ret, frame = self.cap.read()
        if not ret:
            await websocket.send(json.dumps({
                "type": "face_status",
                "detected": False,
                "blink": False,
                "ready": False,
                "checks": self._build_setup_checks(None, False)["checks"],
            }))
            return

        features, blink = self.gaze_estimator.extract_features(frame)
        setup = self._build_setup_checks(
            self.gaze_estimator.last_face_metrics,
            bool(blink),
        )
        await websocket.send(json.dumps({
            "type": "face_status",
            "detected": features is not None,
            "blink": bool(blink),
            "ready": setup["ready"],
            "checks": setup["checks"],
            "metrics": setup["metrics"],
            "preview": self._encode_preview(frame),
        }))

    async def _set_camera(self, websocket, requested_camera_index):
        """Switch the active camera while on the setup screen."""
        if self._calibrating:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Cannot switch cameras during calibration",
            }))
            return

        try:
            requested_camera_index = int(requested_camera_index)
        except (TypeError, ValueError):
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Invalid camera index",
            }))
            return

        self._refresh_available_cameras()
        if requested_camera_index not in self.available_camera_indices:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Camera {requested_camera_index} is not available",
            }))
            await self._send_connection_state(websocket)
            return

        if requested_camera_index == self.camera_index and self.cap and self.cap.isOpened():
            await self._send_connection_state(websocket)
            return

        try:
            self._open_camera(requested_camera_index, allow_fallback=False)
        except RuntimeError as exc:
            await websocket.send(json.dumps({
                "type": "error",
                "message": str(exc),
            }))
            return

        self._reset_tracking_state()
        self._refresh_available_cameras()
        print(f"[gaze-server] Switched to camera {self.camera_index}")
        await self._broadcast_connection_state()

    async def _run_calibration(self, websocket, method: str = "9p"):
        """Run calibration flow, sending point data to the frontend."""
        if self._calibrating:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Calibration already in progress",
            }))
            return

        self._calibrating = True
        self.calibrated = False
        self._last_gaze_position = None
        viewport_width, viewport_height = self._activate_client_viewport(websocket)

        try:
            print(
                "[gaze-server] Starting calibration using viewport "
                f"{viewport_width}x{viewport_height}"
            )

            # Notify calibration starting
            await self._broadcast({
                "type": "calibration_status",
                "status": "starting",
            })

            # Wait for face detection
            await self._broadcast({
                "type": "calibration_status",
                "status": "waiting_face",
            })

            face_found = await self._wait_for_face(websocket)
            if not face_found:
                await self._broadcast({
                    "type": "calibration_status",
                    "status": "failed",
                    "reason": "No face detected",
                })
                return

            # Countdown
            await self._broadcast({
                "type": "calibration_status",
                "status": "countdown",
            })
            await asyncio.sleep(2.0)

            # Step 1: model calibration points.
            if method == "5p":
                order = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2)]
            else:  # 9p default
                order = [
                    (1, 1), (0, 0), (2, 0), (0, 2), (2, 2),
                    (1, 0), (0, 1), (2, 1), (1, 2),
                ]

            pts = compute_grid_points(order, viewport_width, viewport_height)
            total_points = len(pts)

            # Collect calibration data
            all_features = []
            all_targets = []

            for idx, (px, py) in enumerate(pts):
                self.gaze_estimator.reset_movement_center()

                # Pulse phase - attract attention
                await self._broadcast({
                    "type": "calibration_status",
                    "status": "pulse",
                    "phase": "model_calibration",
                    "point_index": idx,
                    "total_points": total_points,
                    "point_x": px,
                    "point_y": py,
                })

                await asyncio.sleep(1.0)

                # Capture phase - collect gaze data
                await self._broadcast({
                    "type": "calibration_status",
                    "status": "capturing",
                    "phase": "model_calibration",
                    "point_index": idx,
                    "total_points": total_points,
                    "point_x": px,
                    "point_y": py,
                })

                point_features = await self._capture_point(
                    websocket,
                    px,
                    py,
                    point_index=idx,
                    total_points=total_points,
                    phase="model_calibration",
                )
                if point_features is None:
                    # Calibration was cancelled
                    await self._broadcast({
                        "type": "calibration_status",
                        "status": "failed",
                        "reason": "Calibration cancelled",
                    })
                    return

                for feat in point_features:
                    all_features.append(feat)
                    all_targets.append([px, py])

            # Train model
            if all_features:
                self.gaze_estimator.train(
                    np.array(all_features),
                    np.array(all_targets),
                )
                quality = self._build_calibration_quality(all_features, all_targets)
                await self._broadcast({
                    "type": "calibration_status",
                    "status": "quality_check",
                    "phase": "model_calibration",
                    "quality": quality,
                })
                await asyncio.sleep(1.4)

                self.smoother = self._make_smoother()
                if self.filter_method in {"kalman", "kalman_ema"}:
                    # Step 2: tune the Kalman filter with the original
                    # three-point fine-tuning pass before tracking starts.
                    await self._run_kalman_tuning(websocket, self.smoother)

                self.calibrated = True

                await self._broadcast({
                    "type": "calibration_status",
                    "status": "complete",
                })
                print("[gaze-server] Calibration complete, starting gaze stream")

                # Start gaze streaming for all connected clients
                for client in self.clients:
                    asyncio.ensure_future(self._gaze_loop(client))
            else:
                await self._broadcast({
                    "type": "calibration_status",
                    "status": "failed",
                    "reason": "No features captured",
                })

        except Exception as e:
            traceback.print_exc()
            await self._broadcast({
                "type": "calibration_status",
                "status": "failed",
                "reason": str(e),
            })
        finally:
            self._calibrating = False

    async def _wait_for_face(self, websocket, timeout: float = 30.0) -> bool:
        """Wait for a stable face detection (not blinking)."""
        start = time.time()
        stable_start = None
        required_stable = 0.5  # Need 0.5s of stable face

        while time.time() - start < timeout:
            ret, frame = self.cap.read()
            if not ret:
                await asyncio.sleep(0.03)
                continue

            features, blink = self.gaze_estimator.extract_features(frame)
            face_ok = features is not None and not blink

            await self._broadcast({
                "type": "face_status",
                "detected": features is not None,
                "blink": bool(blink),
            })

            if face_ok:
                if stable_start is None:
                    stable_start = time.time()
                elif time.time() - stable_start >= required_stable:
                    return True
            else:
                stable_start = None

            await asyncio.sleep(0.03)

        return False

    async def _capture_point(
        self,
        websocket,
        px: int,
        py: int,
        *,
        point_index: Optional[int] = None,
        total_points: Optional[int] = None,
        phase: str = "model_calibration",
        duration: float = 1.0,
    ):
        """Capture gaze features while user looks at a calibration point."""
        features_collected = []
        start = time.time()

        while time.time() - start < duration:
            ret, frame = self.cap.read()
            if not ret:
                await asyncio.sleep(0.03)
                continue

            feat, blink = self.gaze_estimator.extract_features(frame)

            # Movement check — reset timer if patient moved
            if self.gaze_estimator.patient_moved:
                await self._broadcast({
                    "type": "calibration_status",
                    "status": "movement_warning",
                    "phase": phase,
                    "point_index": point_index,
                    "total_points": total_points,
                    "point_x": px,
                    "point_y": py,
                })
                start = time.time()  # Reset capture timer
                await asyncio.sleep(0.5)
                self.gaze_estimator.reset_movement_center()
                continue

            if feat is not None and not blink:
                features_collected.append(feat)

            # Send progress
            elapsed = time.time() - start
            progress = min(elapsed / duration, 1.0)
            await self._broadcast({
                "type": "calibration_status",
                "status": "capturing",
                "phase": phase,
                "point_index": point_index,
                "total_points": total_points,
                "point_x": px,
                "point_y": py,
                "progress": progress,
            })

            await asyncio.sleep(0.03)

        return features_collected if features_collected else None

    async def _run_kalman_tuning(self, websocket, smoother) -> bool:
        """Tune Kalman measurement noise using the original three-point pass."""
        points = [
            (self.screen_width // 2, self.screen_height // 4),
            (self.screen_width // 4, 3 * self.screen_height // 4),
            (3 * self.screen_width // 4, 3 * self.screen_height // 4),
        ]

        proximity_threshold = max(80, self.screen_width / 5)
        initial_delay = 0.5
        data_collection_duration = 0.5
        point_timeout = 18.0
        gaze_positions = []

        await self._broadcast({
            "type": "calibration_status",
            "status": "kalman_starting",
            "phase": "kalman_tuning",
            "total_points": len(points),
        })

        for idx, (px, py) in enumerate(points):
            self.gaze_estimator.reset_movement_center()
            point_start = time.time()
            settle_start = None
            collection_start = None
            point_samples = []

            while True:
                now = time.time()
                if now - point_start > point_timeout:
                    print(
                        "[gaze-server] Kalman tuning point timed out; "
                        "continuing with collected samples"
                    )
                    if point_samples:
                        gaze_positions.extend(point_samples)
                    break

                ret, frame = self.cap.read()
                if not ret:
                    await asyncio.sleep(0.03)
                    continue

                features, blink = self.gaze_estimator.extract_features(frame)

                if self.gaze_estimator.patient_moved:
                    settle_start = None
                    collection_start = None
                    point_samples = []
                    await self._broadcast({
                        "type": "calibration_status",
                        "status": "movement_warning",
                        "phase": "kalman_tuning",
                        "point_index": idx,
                        "total_points": len(points),
                        "point_x": px,
                        "point_y": py,
                    })
                    await asyncio.sleep(0.5)
                    self.gaze_estimator.reset_movement_center()
                    continue

                status = "kalman_waiting"
                progress = 0.0

                if features is not None and not blink:
                    gaze_point = self.gaze_estimator.predict(np.array([features]))[0]
                    gaze_x, gaze_y = map(int, gaze_point)
                    distance = np.hypot(gaze_x - px, gaze_y - py)

                    if distance <= proximity_threshold:
                        if settle_start is None:
                            settle_start = now

                        settled_for = now - settle_start
                        progress = min(settled_for / initial_delay, 1.0)

                        if settled_for >= initial_delay:
                            if collection_start is None:
                                collection_start = now
                                point_samples = []

                            status = "kalman_capturing"
                            point_samples.append([gaze_x, gaze_y])
                            collected_for = now - collection_start
                            progress = min(
                                collected_for / data_collection_duration,
                                1.0,
                            )

                            if collected_for >= data_collection_duration:
                                gaze_positions.extend(point_samples)
                                break
                    else:
                        settle_start = None
                        collection_start = None
                        point_samples = []

                else:
                    settle_start = None
                    collection_start = None
                    point_samples = []

                await self._broadcast({
                    "type": "calibration_status",
                    "status": status,
                    "phase": "kalman_tuning",
                    "point_index": idx,
                    "total_points": len(points),
                    "point_x": px,
                    "point_y": py,
                    "progress": progress,
                })
                await asyncio.sleep(0.03)

        if len(gaze_positions) < 2:
            print("[gaze-server] Kalman tuning skipped: insufficient samples")
            return False

        gaze_positions = np.array(gaze_positions)
        variance = np.var(gaze_positions, axis=0)
        variance[variance == 0] = 1e-4
        smoother.kf.measurementNoiseCov = np.array(
            [[variance[0], 0], [0, variance[1]]],
            dtype=np.float32,
        )
        print(
            "[gaze-server] Kalman tuned with measurement variance "
            f"x={variance[0]:.3f}, y={variance[1]:.3f}"
        )
        return True

    async def _gaze_loop(self, websocket):
        """Stream real-time gaze predictions to a client."""
        while self.calibrated and websocket in self.clients:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    await asyncio.sleep(0.03)
                    continue

                features, blink = self.gaze_estimator.extract_features(frame)

                if features is not None and not blink:
                    prediction = self.gaze_estimator.predict(np.array([features]))[0]
                    x, y = map(int, prediction)

                    if self.smoother:
                        x, y = self.smoother.step(x, y)

                    x = int(np.clip(x, 0, self.screen_width - 1))
                    y = int(np.clip(y, 0, self.screen_height - 1))
                    self._last_gaze_position = (x, y)

                    await websocket.send(json.dumps({
                        "type": "gaze",
                        "x": x,
                        "y": y,
                        "valid": True,
                        "blink": False,
                        "moved": bool(self.gaze_estimator.patient_moved),
                    }))
                else:
                    if self._last_gaze_position is None:
                        x, y = -1, -1
                    else:
                        x, y = self._last_gaze_position

                    await websocket.send(json.dumps({
                        "type": "gaze",
                        "x": int(x),
                        "y": int(y),
                        "valid": False,
                        "blink": bool(blink),
                        "moved": bool(self.gaze_estimator.patient_moved),
                    }))

                await asyncio.sleep(0.033)  # ~30fps

            except websockets.exceptions.ConnectionClosed:
                break
            except Exception:
                await asyncio.sleep(0.1)

    async def _broadcast(self, message: dict):
        """Send a message to all connected clients."""
        data = json.dumps(message)
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(data)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        self.clients -= disconnected

    def cleanup(self):
        """Release resources."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.gaze_estimator.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="HearMe Gaze WebSocket Server")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--host", default="localhost", help="WebSocket host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument(
        "--filter",
        choices=["kalman", "kalman_ema", "none"],
        default="kalman",
        help="Gaze smoothing filter; default matches `eyetrax-demo --filter kalman`",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.25,
        help="EMA smoothing strength for kalman_ema; matches eyetrax CLI default",
    )
    args = parser.parse_args()

    server = GazeServer(
        camera_index=args.camera,
        host=args.host,
        port=args.port,
        filter_method=args.filter,
        ema_alpha=args.ema_alpha,
    )

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n[gaze-server] Shutting down...")
    finally:
        server.cleanup()


if __name__ == "__main__":
    main()
