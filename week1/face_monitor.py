import base64
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

mp_face_mesh_module = mp.solutions.face_mesh

# Landmark index groups (MediaPipe's 468-point face mesh) used to check that
# each region of the face is actually visible and inside the cropped face box.
REGIONS = {
    "left_eye": [33, 133, 160, 159, 158, 144, 145, 153],
    "right_eye": [362, 263, 387, 386, 385, 373, 374, 380],
    "nose": [1, 2, 98, 327],
    "mouth": [61, 291, 13, 14, 78, 308],
    "left_jaw": [234, 93, 132],
    "right_jaw": [454, 323, 361],
}

EDGE_MARGIN = 0.03          # how close to the CROP edge counts as "cut off"
CROP_PADDING_RATIO = 0.25   # extra margin added around YOLO's box before cropping
YOLO_CONFIDENCE = 0.5       # minimum YOLO detection confidence to accept a box

#dataclass
@dataclass
class FrameResult:
    status: str  # "full" | "partial" | "absent"
    face_count: int
    missing_regions: List[str] = field(default_factory=list)


class FaceMonitor:
    """
    Hybrid pipeline:
      YOLO-face detects the face bounding box (fast).
      MediaPipe Face Mesh runs only on that cropped region (accurate, and
      cheap precisely because the input to it is small).
    """

    def __init__(self, yolo_weights_path: str = "models/yolo26s-face.pt"):
        self.yolo = YOLO(yolo_weights_path)
        self.face_mesh = mp_face_mesh_module.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def decode_base64_frame(b64_data: str) -> Optional[np.ndarray]:
        """Decode a base64 (data URL or raw) JPEG/PNG string into a BGR numpy image."""
        try:
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    # ---------- Step 2: YOLO-face detection ----------
    def _detect_faces(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Runs YOLO-face and returns a list of (x1, y1, x2, y2) boxes above threshold."""
        results = self.yolo(frame_bgr, verbose=False)[0]
        boxes: List[Tuple[int, int, int, int]] = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            for box, conf in zip(xyxy, confs):
                if conf >= YOLO_CONFIDENCE:
                    x1, y1, x2, y2 = box
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes

    @staticmethod
    def _largest_box(boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        def area(b):
            x1, y1, x2, y2 = b
            return max(0, x2 - x1) * max(0, y2 - y1)
        return max(boxes, key=area)

    # ---------- Step 3: crop with padding ----------
    @staticmethod
    def _crop_with_padding(frame_bgr: np.ndarray, box: Tuple[int, int, int, int],
                            padding_ratio: float = CROP_PADDING_RATIO) -> np.ndarray:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        pad_x, pad_y = int(w * padding_ratio), int(h * padding_ratio)

        frame_h, frame_w = frame_bgr.shape[:2]
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(frame_w, x2 + pad_x)
        cy2 = min(frame_h, y2 + pad_y)

        return frame_bgr[cy1:cy2, cx1:cx2]

    # ---------- Step 4: Face Mesh on the crop ----------
    def _landmarks_on_crop(self, cropped_bgr: np.ndarray):
        if cropped_bgr is None or cropped_bgr.size == 0:
            return None
        rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark

    # ---------- Step 5: classify full / partial ----------
    @staticmethod
    def _classify(landmarks) -> Tuple[str, List[str]]:
        if landmarks is None:
            # YOLO found a box, but Face Mesh couldn't confirm facial structure
            # inside it -> treat conservatively as partial, not full.
            return "partial", ["undetermined"]

        missing = []
        for region, idxs in REGIONS.items():
            region_hidden = False
            for idx in idxs:
                lm = landmarks[idx]
                if (lm.x < EDGE_MARGIN or lm.x > 1 - EDGE_MARGIN or
                        lm.y < EDGE_MARGIN or lm.y > 1 - EDGE_MARGIN):
                    region_hidden = True
                    break
            if region_hidden:
                missing.append(region)

        if missing:
            return "partial", missing
        return "full", []

    # ---------- Public entry point ----------
    def analyze(self, frame_bgr: Optional[np.ndarray]) -> FrameResult:
        if frame_bgr is None:
            return FrameResult(status="absent", face_count=0)

        boxes = self._detect_faces(frame_bgr)
        if not boxes:
            return FrameResult(status="absent", face_count=0)

        face_count = len(boxes)
        primary_box = self._largest_box(boxes)
        cropped = self._crop_with_padding(frame_bgr, primary_box)
        landmarks = self._landmarks_on_crop(cropped)
        status, missing_regions = self._classify(landmarks)

        return FrameResult(status=status, face_count=face_count, missing_regions=missing_regions)

    def close(self):
        self.face_mesh.close()
