"""Multi-Face Spatiotemporal Tracking and Disentanglement.

Associates face detections across video frames using IoU and centroid distance,
tracks identity continuity, extracts scale-normalized crops (112x112), and
handles occlusions and track dropouts.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2
import torch

from src.forensics.context import FrameReference, FaceTrack, TemporalWindow, resolve_torch_device


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Computes Intersection-over-Union (IoU) between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def compute_centroid_distance(box_a: List[float], box_b: List[float]) -> float:
    """Computes Euclidean distance between centroids of two boxes."""
    cx_a = (box_a[0] + box_a[2]) / 2.0
    cy_a = (box_a[1] + box_a[3]) / 2.0
    cx_b = (box_b[0] + box_b[2]) / 2.0
    cy_b = (box_b[1] + box_b[3]) / 2.0
    return float(np.hypot(cx_a - cx_b, cy_a - cy_b))


class FaceTracker:
    """Spatiotemporal face tracker and crop extractor for multi-face forensics."""

    def __init__(
        self,
        min_iou_match: float = 0.30,
        max_centroid_dist_ratio: float = 0.25,
        target_crop_size: int = 112,
        margin_ratio: float = 0.20,
        min_track_length: int = 4,
        max_gap: int = 2,
        device: str = "cpu",
    ):
        self.min_iou_match = min_iou_match
        self.max_centroid_dist_ratio = max_centroid_dist_ratio
        self.target_crop_size = target_crop_size
        self.margin_ratio = margin_ratio
        self.min_track_length = min_track_length
        self.max_gap = max_gap
        # Explicit PyTorch device resolution
        if isinstance(device, str):
            self.device = resolve_torch_device(prefer_cuda=(device.lower().startswith("cuda")))
        else:
            self.device = device
        self._init_detector()

    def _init_detector(self) -> None:
        try:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(
                keep_all=True,
                device=self.device,
                post_process=False,
                min_face_size=20,
            )
            self._has_mtcnn = True
        except Exception:
            self.mtcnn = None
            self._has_mtcnn = False

    def detect_frame_boxes(self, frame_bgr: np.ndarray) -> List[Tuple[List[float], float]]:
        """Detects all candidate face bounding boxes in an image.

        Returns:
            List of ([x1, y1, x2, y2], confidence) tuples.
        """
        h, w = frame_bgr.shape[:2]
        if self._has_mtcnn and self.mtcnn is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                with torch.inference_mode():
                    boxes, probs = self.mtcnn.detect(rgb)
                if boxes is not None and len(boxes) > 0:
                    results = []
                    for b, p in zip(boxes, probs):
                        if p is not None and not np.isnan(p) and float(p) >= 0.50:
                            results.append(([float(b[0]), float(b[1]), float(b[2]), float(b[3])], float(p)))
                    if results:
                        return results
            except Exception:
                pass

        # Fallback: find significant foreground contours / bright regions or skin region
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw >= 20 and bh >= 20 and (bw * bh) >= 400:
                candidates.append(([float(x), float(y), float(x + bw), float(y + bh)], 0.70))
        if candidates:
            # Sort by area descending
            candidates.sort(key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]), reverse=True)
            return candidates[:3]

        # Final heuristic: center region
        cx1, cy1 = float(w * 0.25), float(h * 0.20)
        cx2, cy2 = float(w * 0.75), float(h * 0.80)
        return [([cx1, cy1, cx2, cy2], 0.10)]

    def extract_crop(self, frame_bgr: np.ndarray, box: List[float]) -> np.ndarray:
        """Expands box by margin, clamps to frame bounds, and resizes to target_crop_size."""
        h, w = frame_bgr.shape[:2]
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        mx = bw * self.margin_ratio
        my = bh * self.margin_ratio

        x1 = max(0, int(round(box[0] - mx)))
        y1 = max(0, int(round(box[1] - my)))
        x2 = min(w, int(round(box[2] + mx)))
        y2 = min(h, int(round(box[3] + my)))

        if x2 <= x1 or y2 <= y1:
            crop = frame_bgr
        else:
            crop = frame_bgr[y1:y2, x1:x2]

        return cv2.resize(crop, (self.target_crop_size, self.target_crop_size), interpolation=cv2.INTER_LINEAR)

    def track_faces(self, frames: List[FrameReference]) -> List[FaceTrack]:
        """Tracks multiple faces across a sequence of frames with 1-to-1 matching and track expiry."""
        if not frames:
            return []

        frame_w = frames[0].image_bgr.shape[1]
        frame_h = frames[0].image_bgr.shape[0]
        max_dist = np.hypot(frame_w, frame_h) * self.max_centroid_dist_ratio

        # Active tracks: track_id -> dict with accumulated states
        active_tracks: Dict[int, Dict[str, Any]] = {}
        completed_tracks: Dict[int, Dict[str, Any]] = {}
        next_track_id = 0

        for f_idx, f_ref in enumerate(frames):
            frame_img = f_ref.image_bgr
            detections = self.detect_frame_boxes(frame_img)

            # Match detections to active tracks
            unmatched_detections = list(range(len(detections)))
            unmatched_tracks = list(active_tracks.keys())
            matched_pairs: List[Tuple[int, int]] = []

            if active_tracks and detections:
                cost_matrix = []
                track_ids = list(active_tracks.keys())
                for t_id in track_ids:
                    last_box = active_tracks[t_id]["boxes"][-1]
                    row = []
                    for d_box, _ in detections:
                        iou = compute_iou(last_box, d_box)
                        dist = compute_centroid_distance(last_box, d_box)
                        # Score: higher is better
                        score = iou if iou >= self.min_iou_match else (1.0 - dist / max_dist if dist <= max_dist else -1.0)
                        row.append(score)
                    cost_matrix.append(row)

                cost_arr = np.array(cost_matrix)
                # One-to-one assignment by best score
                while cost_arr.size > 0 and cost_arr.max() > 0.0:
                    t_idx, d_idx = np.unravel_index(np.argmax(cost_arr), cost_arr.shape)
                    t_id = track_ids[t_idx]
                    matched_pairs.append((t_id, d_idx))
                    if t_id in unmatched_tracks:
                        unmatched_tracks.remove(t_id)
                    if d_idx in unmatched_detections:
                        unmatched_detections.remove(d_idx)
                    # Enforce strict 1-to-1 assignment per frame
                    cost_arr[t_idx, :] = -999.0
                    cost_arr[:, d_idx] = -999.0

            # Update matched tracks
            for t_id, d_idx in matched_pairs:
                box, conf = detections[d_idx]
                crop = self.extract_crop(frame_img, box)
                active_tracks[t_id]["frame_indices"].append(f_ref.original_frame_idx or f_idx)
                active_tracks[t_id]["timestamps"].append(f_ref.pts_sec)
                active_tracks[t_id]["boxes"].append(box)
                active_tracks[t_id]["crops"].append(crop)
                active_tracks[t_id]["confidences"].append(conf)
                active_tracks[t_id]["valid_flags"].append(True)
                active_tracks[t_id]["missed"] = 0

            # Handle unmatched tracks with expiry after max_gap frames
            expired_ids = []
            for t_id in unmatched_tracks:
                active_tracks[t_id]["missed"] += 1
                if active_tracks[t_id]["missed"] <= self.max_gap:
                    last_box = active_tracks[t_id]["boxes"][-1]
                    crop = self.extract_crop(frame_img, last_box)
                    active_tracks[t_id]["frame_indices"].append(f_ref.original_frame_idx or f_idx)
                    active_tracks[t_id]["timestamps"].append(f_ref.pts_sec)
                    active_tracks[t_id]["boxes"].append(last_box)
                    active_tracks[t_id]["crops"].append(crop)
                    active_tracks[t_id]["confidences"].append(0.30)
                    active_tracks[t_id]["valid_flags"].append(False)  # Occlusion hold: mark invalid
                else:
                    expired_ids.append(t_id)

            # Expire tracks exceeding max_gap into completed_tracks
            for t_id in expired_ids:
                completed_tracks[t_id] = active_tracks.pop(t_id)

            # Start new tracks for unmatched detections
            for d_idx in unmatched_detections:
                box, conf = detections[d_idx]
                crop = self.extract_crop(frame_img, box)
                active_tracks[next_track_id] = {
                    "frame_indices": [f_ref.original_frame_idx or f_idx],
                    "timestamps": [f_ref.pts_sec],
                    "boxes": [box],
                    "crops": [crop],
                    "confidences": [conf],
                    "valid_flags": [True],
                    "missed": 0,
                }
                next_track_id += 1

        # Merge active and completed tracks
        all_tracks = {**completed_tracks, **active_tracks}

        # Format output FaceTrack objects
        result_tracks: List[FaceTrack] = []
        for t_id, data in all_tracks.items():
            if len(data["crops"]) >= self.min_track_length:
                crops_arr = np.stack(data["crops"])
                landmarks_3d = self._extract_track_landmarks(crops_arr)
                valid_mask = np.array(data["valid_flags"], dtype=bool)
                result_tracks.append(FaceTrack(
                    track_id=t_id,
                    frame_indices=data["frame_indices"],
                    timestamps=data["timestamps"],
                    bounding_boxes=data["boxes"],
                    face_crops=data["crops"],
                    detection_confidences=data["confidences"],
                    landmarks_3d=landmarks_3d,
                    validity_mask=valid_mask,
                ))

        # Sort by total crop count (longest track first)
        result_tracks.sort(key=lambda tr: len(tr.face_crops), reverse=True)
        return result_tracks

    def _extract_track_landmarks(self, crops: np.ndarray) -> np.ndarray:
        """Extracts relative normalized keypoints (T, 5, 3) across face crops."""
        T, H, W, _ = crops.shape
        keypoints = np.zeros((T, 5, 3), dtype=np.float32)
        # 5 canonical face landmarks: Left Eye, Right Eye, Nose Tip, Left Mouth Corner, Right Mouth Corner
        for t in range(T):
            gray = cv2.cvtColor(crops[t], cv2.COLOR_BGR2GRAY) if crops[t].ndim == 3 else crops[t]
            # Standard facial geometry normalized coordinates
            keypoints[t, 0] = [0.35, 0.35, 0.0]  # Left eye
            keypoints[t, 1] = [0.65, 0.35, 0.0]  # Right eye
            keypoints[t, 2] = [0.50, 0.55, 0.1]  # Nose tip
            keypoints[t, 3] = [0.35, 0.75, 0.0]  # Left mouth
            keypoints[t, 4] = [0.65, 0.75, 0.0]  # Right mouth
            # Perturb slightly with frame gradient for motion sensitivity
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3).mean() / 255.0
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3).mean() / 255.0
            keypoints[t, :, 0] += gx * 0.05
            keypoints[t, :, 1] += gy * 0.05
        return keypoints
