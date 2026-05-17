import { useCallback, useEffect } from 'react';
import { useEyeTracking, TrackingState } from '../../context/EyeTrackingContext';
import './Calibration.css';

const CHECK_ITEMS = [
  {
    key: 'face',
    label: 'Face visible',
    fail: 'Move into the camera view',
  },
  {
    key: 'webcam_level',
    label: 'Eyes level with webcam',
    fail: 'Raise/lower camera or seat until eyes sit inside the blue band',
  },
  {
    key: 'centered',
    label: 'Horizontally centered',
    fail: 'Move left or right toward the guide',
  },
  {
    key: 'head_pose',
    label: 'Facing webcam',
    fail: 'Look straight at the webcam',
  },
  {
    key: 'eye_line',
    label: 'Head tilt',
    fail: 'Small tilt is OK; gently level your eyes',
  },
  {
    key: 'distance',
    label: 'Good distance',
    fail: 'Move slightly closer or farther',
  },
  {
    key: 'lighting',
    label: 'Lighting',
    fail: 'Use even front lighting',
  },
  {
    key: 'eyes_open',
    label: 'Eyes open',
    fail: 'Avoid blinking while starting',
  },
];

function SetupCheck({ item, ok }) {
  return (
    <div className={`setup-check ${ok ? 'ok' : 'needs-work'}`}>
      <span className="setup-check-dot" />
      <span className="setup-check-label">{item.label}</span>
      <span className="setup-check-detail">{ok ? 'OK' : item.fail}</span>
    </div>
  );
}

export default function PreCalibration() {
  const {
    state,
    faceDetected,
    cameraPreview,
    availableCameras,
    activeCameraIndex,
    cameraSwitching,
    setupChecks,
    setupReady,
    faceMetrics,
    error,
    startCalibration,
    checkFace,
    setCamera,
  } = useEyeTracking();

  useEffect(() => {
    if (state !== TrackingState.PRE_CALIBRATION) return;

    const interval = setInterval(() => {
      checkFace();
    }, 500);

    checkFace();
    return () => clearInterval(interval);
  }, [state, checkFace]);

  const handleStart = useCallback(() => {
    startCalibration('9p');
  }, [startCalibration]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (state !== TrackingState.PRE_CALIBRATION) return;
      if (e.code === 'Space' || e.code === 'Enter') {
        e.preventDefault();
        handleStart();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [state, handleStart]);

  if (state !== TrackingState.PRE_CALIBRATION && state !== TrackingState.DISCONNECTED) {
    return null;
  }

  const formatAngle = (value) => `${Math.round(value || 0)} deg`;
  const metricsSummary = faceMetrics
    ? `Turn ${formatAngle(faceMetrics.yaw_degrees)} | Nod ${formatAngle(faceMetrics.pitch_degrees)} | Tilt ${formatAngle(faceMetrics.roll_degrees)}`
    : 'Waiting for face measurements';
  const showCameraSwitcher =
    state === TrackingState.PRE_CALIBRATION &&
    availableCameras.length >= 2;

  return (
    <div className="calibration-overlay">
      <div className="precalib-container">
        <div className="precalib-header">
          <h1 className="precalib-title">Eye Tracking Setup</h1>
          <p className="precalib-subtitle">
            Sit naturally, keep your face visible, and use the guide to make small adjustments before calibration.
          </p>
        </div>

        <div className="precalib-layout">
          <section className="camera-panel" aria-label="Camera preview">
            <div className={`camera-frame ${setupReady ? 'ready' : ''}`}>
              {cameraPreview ? (
                <img className="camera-preview" src={cameraPreview} alt="Camera preview" />
              ) : (
                <div className="camera-placeholder">
                  {state === TrackingState.DISCONNECTED ? 'Connecting to gaze server...' : 'Waiting for camera preview...'}
                </div>
              )}
              <div className="camera-guide">
                <div className="camera-guide-eye-band" />
                <div className="camera-guide-face" />
                <div className="camera-guide-line horizontal" />
                <div className="camera-guide-line vertical" />
                <div className="camera-guide-label eye-label">eyes here</div>
                <div className="camera-guide-label center-label">center face</div>
              </div>
            </div>

            <div className={`setup-ready-banner ${setupReady ? 'ready' : 'not-ready'}`}>
              {setupReady ? 'Ready to calibrate' : 'Guidance only - start when this feels natural'}
            </div>
            <div className="setup-metrics">{metricsSummary}</div>
            <div className="eye-level-note">
              Best accuracy usually comes when your eyes sit inside the blue band, roughly level with the webcam.
            </div>
            {showCameraSwitcher ? (
              <div className="camera-switcher" aria-label="Camera selection">
                <div className="camera-switcher-label">Gaze camera</div>
                <div className="camera-switcher-buttons">
                  {availableCameras.map((cameraIndex) => (
                    <button
                      key={cameraIndex}
                      type="button"
                      className={`camera-switcher-btn ${cameraIndex === activeCameraIndex ? 'active' : ''}`}
                      onClick={() => setCamera(cameraIndex)}
                      disabled={cameraSwitching || cameraIndex === activeCameraIndex}
                    >
                      Camera {cameraIndex + 1}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
          </section>

          <section className="setup-panel" aria-label="Setup checks">
            <div className="precalib-rules">
              <div className="rule-item">
                <span className="rule-text">
                  <strong>Lighting:</strong> use even front light and avoid a bright window behind you.
                </span>
              </div>
              <div className="rule-item">
                <span className="rule-text">
                  <strong>Eye level:</strong> keep your eyes inside the blue band; adjust chair/screen height before moving your head.
                </span>
              </div>
              <div className="rule-item">
                <span className="rule-text">
                  <strong>Corner separation:</strong> after the first pass, the app checks whether corner looks produced different gaze data.
                </span>
              </div>
              <div className="rule-item">
                <span className="rule-text">
                  <strong>Head angle:</strong> face the webcam roughly straight on; do not force a rigid posture.
                </span>
              </div>
              <div className="rule-item">
                <span className="rule-text">
                  <strong>Movement:</strong> stay still during both calibration steps.
                </span>
              </div>
            </div>

            <div className="setup-checks">
              {CHECK_ITEMS.map((item) => (
                <SetupCheck
                  key={item.key}
                  item={item}
                  ok={Boolean(setupChecks?.[item.key])}
                />
              ))}
            </div>
          </section>
        </div>

        <div className="precalib-status">
          <div className={`face-indicator ${faceDetected ? 'detected' : 'not-detected'}`}>
            <span className="indicator-dot" />
            <span>{faceDetected ? 'Face detected' : 'No face detected'}</span>
          </div>
        </div>

        {error && <div className="precalib-error">{error}</div>}

        {state === TrackingState.DISCONNECTED && (
          <div className="precalib-error">
            Connecting to gaze server. Make sure the Python server is running.
          </div>
        )}

        <button
          className="precalib-start-btn"
          onClick={handleStart}
          disabled={state === TrackingState.DISCONNECTED}
          aria-label="Start calibration"
        >
          Start Calibration
        </button>
        <p className="precalib-hint">Press <kbd>Space</kbd> or <kbd>Enter</kbd> to start</p>
      </div>
    </div>
  );
}
