import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';

const EyeTrackingContext = createContext(null);

const WS_URL = 'ws://localhost:8765';
const RECONNECT_DELAY = 2000;

// eslint-disable-next-line react-refresh/only-export-components
export const TrackingState = {
  DISCONNECTED: 'disconnected',
  CONNECTED: 'connected',
  PRE_CALIBRATION: 'pre_calibration',
  CALIBRATING: 'calibrating',
  TRACKING: 'tracking',
};

function readViewportSize() {
  const viewport = window.visualViewport;
  return {
    width: Math.max(1, Math.round(viewport?.width ?? window.innerWidth)),
    height: Math.max(1, Math.round(viewport?.height ?? window.innerHeight)),
  };
}

export function EyeTrackingProvider({ children }) {
  const [state, setState] = useState(TrackingState.DISCONNECTED);
  const [gazePosition, setGazePosition] = useState({ x: -1, y: -1 });
  const [gazeValid, setGazeValid] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [isBlinking, setIsBlinking] = useState(false);
  const [patientMoved, setPatientMoved] = useState(false);
  const [calibrationData, setCalibrationData] = useState(null);
  const [cameraPreview, setCameraPreview] = useState(null);
  const [availableCameras, setAvailableCameras] = useState([]);
  const [activeCameraIndex, setActiveCameraIndex] = useState(0);
  const [cameraSwitching, setCameraSwitching] = useState(false);
  const [setupChecks, setSetupChecks] = useState(null);
  const [setupReady, setSetupReady] = useState(false);
  const [faceMetrics, setFaceMetrics] = useState(null);
  const [error, setError] = useState(null);

  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const connectRef = useRef(null);

  const handleCalibrationStatus = useCallback((msg) => {
    switch (msg.status) {
      case 'starting':
      case 'waiting_face':
      case 'countdown':
        setState(TrackingState.CALIBRATING);
        setCalibrationData({ status: msg.status });
        break;

      case 'pulse':
      case 'capturing':
      case 'kalman_starting':
      case 'kalman_waiting':
      case 'kalman_capturing':
        setState(TrackingState.CALIBRATING);
        setCalibrationData({
          status: msg.status,
          phase: msg.phase,
          pointIndex: msg.point_index,
          totalPoints: msg.total_points,
          pointX: msg.point_x,
          pointY: msg.point_y,
          progress: msg.progress || 0,
        });
        break;

      case 'quality_check':
        setState(TrackingState.CALIBRATING);
        setCalibrationData({
          status: msg.status,
          phase: msg.phase,
          quality: msg.quality,
        });
        break;

      case 'movement_warning':
        setCalibrationData((prev) => ({
          ...prev,
          status: 'movement_warning',
          phase: msg.phase || prev?.phase,
          pointIndex: msg.point_index ?? prev?.pointIndex,
          totalPoints: msg.total_points ?? prev?.totalPoints,
          pointX: msg.point_x ?? prev?.pointX,
          pointY: msg.point_y ?? prev?.pointY,
        }));
        break;

      case 'complete':
        setState(TrackingState.TRACKING);
        setCalibrationData(null);
        break;

      case 'failed':
        setState(TrackingState.PRE_CALIBRATION);
        setCalibrationData(null);
        setError(msg.reason || 'Calibration failed');
        break;

      default:
        break;
    }
  }, []);

  const handleMessage = useCallback((msg) => {
    switch (msg.type) {
      case 'connection':
        setState(msg.calibrated ? TrackingState.TRACKING : TrackingState.PRE_CALIBRATION);
        setAvailableCameras(Array.isArray(msg.available_cameras) ? msg.available_cameras : []);
        setActiveCameraIndex(Number.isInteger(msg.camera_index) ? msg.camera_index : 0);
        setCameraSwitching(false);
        break;

      case 'gaze':
        if (msg.valid !== false && msg.x >= 0 && msg.y >= 0) {
          setGazePosition({ x: msg.x, y: msg.y });
          setGazeValid(true);
        } else {
          setGazeValid(false);
        }
        setIsBlinking(msg.blink);
        setPatientMoved(msg.moved);
        break;

      case 'face_status':
        setFaceDetected(msg.detected);
        setIsBlinking(msg.blink);
        setSetupReady(Boolean(msg.ready));
        setSetupChecks(msg.checks || null);
        setFaceMetrics(msg.metrics || null);
        setCameraPreview(msg.preview ? `data:image/jpeg;base64,${msg.preview}` : null);
        break;

      case 'calibration_status':
        handleCalibrationStatus(msg);
        break;

      case 'error':
        setError(msg.message);
        setCameraSwitching(false);
        break;

      default:
        break;
    }
  }, [handleCalibrationStatus]);

  const sendScreenSize = useCallback((ws = wsRef.current) => {
    if (ws?.readyState !== WebSocket.OPEN) return;

    const { width, height } = readViewportSize();
    ws.send(JSON.stringify({
      type: 'set_screen_size',
      width,
      height,
    }));
  }, []);

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimerRef.current) return;

    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null;
      connectRef.current?.();
    }, RECONNECT_DELAY);
  }, []);

  const connect = useCallback(() => {
    const currentSocket = wsRef.current;
    if (
      currentSocket?.readyState === WebSocket.OPEN ||
      currentSocket?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setError(null);
        sendScreenSize(ws);
      };

      ws.onmessage = (event) => {
        handleMessage(JSON.parse(event.data));
      };

      ws.onclose = () => {
        wsRef.current = null;
        setState(TrackingState.DISCONNECTED);
        scheduleReconnect();
      };

      ws.onerror = () => {
        setError('WebSocket connection failed. Is the gaze server running?');
        ws.close();
      };
    } catch {
      setError('Cannot connect to gaze server');
      scheduleReconnect();
    }
  }, [handleMessage, scheduleReconnect, sendScreenSize]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  const startCalibration = useCallback((method = '9p') => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setError('Not connected to gaze server');
      return;
    }

    setError(null);
    sendScreenSize(ws);
    ws.send(JSON.stringify({ type: 'start_calibration', method }));
  }, [sendScreenSize]);

  const recalibrate = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setError('Not connected to gaze server');
      return;
    }

    setError(null);
    setGazePosition({ x: -1, y: -1 });
    setGazeValid(false);
    setIsBlinking(false);
    setPatientMoved(false);
    setCalibrationData(null);
    ws.send(JSON.stringify({ type: 'reset_calibration' }));
    setState(TrackingState.PRE_CALIBRATION);
  }, []);

  const checkFace = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: 'check_face' }));
  }, []);

  const setCamera = useCallback((cameraIndex) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setError('Not connected to gaze server');
      return;
    }

    setError(null);
    setCameraSwitching(true);
    ws.send(JSON.stringify({
      type: 'set_camera',
      camera_index: cameraIndex,
    }));
  }, []);

  useEffect(() => {
    const connectTimer = window.setTimeout(() => {
      connectRef.current?.();
    }, 0);

    return () => {
      window.clearTimeout(connectTimer);

      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
      }

      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    window.addEventListener('resize', sendScreenSize);
    return () => window.removeEventListener('resize', sendScreenSize);
  }, [sendScreenSize]);

  const value = {
    state,
    gazePosition,
    gazeValid,
    faceDetected,
    isBlinking,
    patientMoved,
    calibrationData,
    cameraPreview,
    availableCameras,
    activeCameraIndex,
    cameraSwitching,
    setupChecks,
    setupReady,
    faceMetrics,
    error,
    startCalibration,
    recalibrate,
    checkFace,
    setCamera,
  };

  return (
    <EyeTrackingContext.Provider value={value}>
      {children}
    </EyeTrackingContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useEyeTracking() {
  const context = useContext(EyeTrackingContext);
  if (!context) {
    throw new Error('useEyeTracking must be used inside EyeTrackingProvider');
  }
  return context;
}
