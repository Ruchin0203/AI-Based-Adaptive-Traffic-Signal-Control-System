# =============================================================================
#  detector.py  —  Vehicle Detection (YOLOv8 + demo fallback)
#
#  Key addition: get_weighted_count() returns both raw count and a
#  weighted count that accounts for vehicle class (motorcycle=0.5, car=1.0,
#  bus=2.5, truck=3.0) so the signal controller can make smarter decisions.
#
#  Author : Ruchin Patel | Adani University | B.Tech CSE 2024-25
# =============================================================================

import threading
import time
import cv2
import config
from utils.demo_traffic import FourWayDemoSource

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False


class VehicleDetector:

    def __init__(self, road: str, source, model=None):
        self.road    = road
        self._src    = source
        self._model  = model
        self._count  = 0
        self._weighted = 0.0   # weighted vehicle count
        self._frame  = None
        self._inf_ms = 0.0
        self._lock   = threading.Lock()
        self._run    = False

        use_demo = (source is None or source == "demo"
                    or not _YOLO_OK or model is None)
        self._demo = FourWayDemoSource(road) if use_demo else None

    def start(self):
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._run = False

    def set_phase(self, phase: str):
        """Inform demo source of current signal phase for animation."""        
        if self._demo is not None:
            self._demo.set_phase(phase)

    def get_count(self) -> int:
        with self._lock:
            return self._count

    def get_weighted_count(self) -> tuple:
        """Returns (raw_count: int, weighted_count: float)."""
        with self._lock:
            return self._count, self._weighted

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    # ── Thread loops ──────────────────────────────────────────────────────────

    def _loop(self):
        if self._demo:
            self._loop_demo()
        else:
            self._loop_yolo()

    def _loop_demo(self):
        delay = 1.0 / config.FPS_CAP
        while self._run:
            frame, count = self._demo.get_frame_and_count()
            frame = cv2.resize(frame, (config.FEED_W, config.FEED_H))
            with self._lock:
                self._count    = count
                self._weighted = float(count)   # demo: all treated as cars (weight=1)
                self._frame    = frame
            time.sleep(delay)

    def _loop_yolo(self):
        cap = cv2.VideoCapture(self._src)
        if not cap.isOpened():
            print(f"[{self.road}] Cannot open '{self._src}' → demo mode.")
            self._demo = FourWayDemoSource(self.road)
            self._loop_demo()
            return

        idx, last_c, last_w, last_f = 0, 0, 0.0, None
        delay = 1.0 / config.FPS_CAP

        while self._run:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (config.FEED_W, config.FEED_H))

            if idx % config.DETECTION_INTERVAL == 0:
                t0 = time.perf_counter()
                count, weighted, ann = self._detect(frame)
                elapsed = (time.perf_counter() - t0) * 1000
                last_c, last_w, last_f = count, weighted, ann
                with self._lock:
                    self._count    = count
                    self._weighted = weighted
                    self._frame    = ann
                    self._inf_ms   = elapsed
            else:
                with self._lock:
                    self._count    = last_c
                    self._weighted = last_w
                    if last_f is not None:
                        self._frame = last_f

            idx += 1
            time.sleep(delay)

        cap.release()

    def _detect(self, frame):
        """
        Run YOLOv8 inference.
        Returns (raw_count, weighted_count, annotated_frame).
        """
        results   = self._model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
        annotated = frame.copy()
        count     = 0
        weighted  = 0.0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in config.VEHICLE_CLASSES:
                    continue
                conf = float(box.conf[0])

                # Weighted count
                w = config.VEHICLE_WEIGHT.get(cls, config.VEHICLE_WEIGHT_DEFAULT)
                weighted += w
                count    += 1

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{r.names[cls]} {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 0), -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)

        # HUD overlay
        cv2.putText(annotated, f"Detected: {count}  Weighted: {weighted:.1f}",
                    (8, config.FEED_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, config.C_GOOD, 2)
        return count, weighted, annotated


def create_detectors() -> dict:
    model = None
    if _YOLO_OK:
        print("[detector] Loading YOLOv8-nano …")
        try:
            model = YOLO("yolov8n.pt")
            print("[detector] YOLOv8-nano ready.")
        except Exception as e:
            print(f"[detector] Load failed ({e}) → demo mode.")
    else:
        print("[detector] ultralytics not installed → demo mode.")

    return {
        road: VehicleDetector(road, config.VIDEO_SOURCES.get(road, "demo"), model)
        for road in config.ROADS
    }
