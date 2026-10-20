"""Tests for FaceTracker spatiotemporal face tracking and crop extraction."""
import numpy as np
import cv2
import pytest

from src.forensics.context import FrameReference
from src.forensics.tracking import FaceTracker, compute_iou, compute_centroid_distance


def test_iou_and_centroid_calculations():
    box_a = [10.0, 10.0, 50.0, 50.0]  # area 1600
    box_b = [10.0, 10.0, 50.0, 50.0]
    assert compute_iou(box_a, box_b) == 1.0

    box_c = [50.0, 50.0, 90.0, 90.0]
    assert compute_iou(box_a, box_c) == 0.0

    dist = compute_centroid_distance(box_a, box_c)
    assert pytest.approx(dist, 0.01) == np.hypot(40.0, 40.0)


def test_face_tracker_on_synthetic_frames():
    tracker = FaceTracker(target_crop_size=112, min_track_length=3)

    # Create synthetic frames with a simulated moving white square representing a face
    frames = []
    for i in range(10):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        # Draw moving face box
        x = 50 + i * 2
        y = 60
        cv2.rectangle(img, (x, y), (x + 80, y + 80), (255, 255, 255), -1)
        frames.append(FrameReference(
            sample_idx=i,
            original_frame_idx=i,
            pts_sec=i * 0.04,
            image_bgr=img,
        ))

    tracks = tracker.track_faces(frames)
    assert len(tracks) >= 1
    primary_track = tracks[0]
    assert len(primary_track.face_crops) == 10
    assert primary_track.face_crops[0].shape == (112, 112, 3)
    assert primary_track.landmarks_3d.shape == (10, 5, 3)
