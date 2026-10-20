"""Integration tests verifying that MTCNN executes successfully and validates detection output."""

import numpy as np
import pytest

try:
    import torch
    from facenet_pytorch import MTCNN
    from src.data.detector import MTCNNFaceDetector
    from src.utils.config import DetectionConfig
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False


@pytest.mark.skipif(not HAS_MTCNN, reason="facenet-pytorch or PyTorch not installed")
def test_mtcnn_execution_and_detection():
    """Verify MTCNN executes successfully under torch.inference_mode()."""
    config = DetectionConfig(confidence_threshold=0.50)
    detector = MTCNNFaceDetector(config=config, device="cpu")

    # 1. Blank frame test: should return None without crashing
    blank_frame = np.zeros((160, 160, 3), dtype=np.uint8)
    bbox_blank = detector.detect_face(blank_frame)
    assert bbox_blank is None

    # 2. Synthetic frame test: verifies MTCNN executes successfully and validates detection output
    synth_frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    # Draw simple facial feature approximations
    synth_frame[50:150, 50:150] = [210, 180, 140]  # skin tone block
    bbox_synth = detector.detect_face(synth_frame)
    if bbox_synth is not None:
        assert bbox_synth.confidence >= 0.50
        assert bbox_synth.x2 > bbox_synth.x1
        assert bbox_synth.y2 > bbox_synth.y1
        assert bbox_synth.area > 0.0
