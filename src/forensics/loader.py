"""Forensic Video Loader with container inspection and temporal window frame extraction."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np

from src.forensics.context import FrameReference, VideoMetadata


class ForensicVideoLoader:
    """Robust video loader supporting container metadata inspection and windowed frame extraction."""

    def __init__(self) -> None:
        self.ffprobe_path = shutil.which("ffprobe")

    def inspect_container(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """Inspects digital video container properties via FFprobe or OpenCV fallback."""
        p = Path(video_path)
        if not p.is_file():
            raise FileNotFoundError(f"Video file does not exist: {p}")

        if self.ffprobe_path:
            try:
                cmd = [
                    self.ffprobe_path,
                    "-v", "error",
                    "-show_entries",
                    "format=duration,bit_rate,format_name:stream=codec_name,codec_type,width,height,r_frame_rate,avg_frame_rate,nb_frames,sample_rate",
                    "-of", "json",
                    str(p),
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
                data = json.loads(res.stdout)

                streams = data.get("streams", [])
                v_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
                audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

                r_fps_str = v_stream.get("r_frame_rate", "25/1")
                avg_fps_str = v_stream.get("avg_frame_rate", r_fps_str)
                try:
                    num, den = map(float, r_fps_str.split("/"))
                    fps = num / den if den > 0 else 25.0
                except Exception:
                    fps = 25.0

                duration = float(data.get("format", {}).get("duration", 0.0))
                bitrate = data.get("format", {}).get("bit_rate")
                bitrate_kbps = float(bitrate) / 1000.0 if bitrate else 5000.0

                total_frames = int(v_stream.get("nb_frames") or (duration * fps if duration > 0 else 0))
                is_vfr = r_fps_str != avg_fps_str

                return {
                    "duration_sec": duration,
                    "fps": fps,
                    "total_frames": total_frames,
                    "width": int(v_stream.get("width", 0)),
                    "height": int(v_stream.get("height", 0)),
                    "codec": str(v_stream.get("codec_name", "unknown")),
                    "bitrate_kbps": bitrate_kbps,
                    "is_vfr": is_vfr,
                    "has_audio": len(audio_streams) > 0,
                }
            except Exception:
                pass

        # Fallback to OpenCV
        return self._inspect_opencv(p)

    def _inspect_opencv(self, video_path: Path) -> Dict[str, Any]:
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        duration = total_frames / fps if fps > 0 else 0.0
        return {
            "duration_sec": duration,
            "fps": fps,
            "total_frames": total_frames,
            "width": w,
            "height": h,
            "codec": "unknown",
            "bitrate_kbps": 5000.0,
            "is_vfr": False,
            "has_audio": False,
        }

    def load_temporal_window_frames(
        self,
        video_path: Union[str, Path],
        start_sec: float,
        duration_sec: float,
        target_frame_count: int = 16,
    ) -> List[FrameReference]:
        """Loads dense consecutive frames within a temporal window [start_sec, start_sec + duration_sec]."""
        p = Path(video_path)
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            return []

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Calculate starting frame index
        start_frame = max(0, int(start_sec * fps))
        if total_frames > 0 and start_frame >= total_frames:
            start_frame = max(0, total_frames - target_frame_count)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        raw_frames: List[FrameReference] = []
        frame_idx = start_frame
        max_duration_frames = max(target_frame_count, int(duration_sec * fps) + 2)

        while cap.isOpened() and len(raw_frames) < max_duration_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            pts_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            pts_sec = pts_msec / 1000.0 if pts_msec > 0 else (frame_idx / fps if fps > 0 else 0.0)

            ref = FrameReference(
                sample_idx=len(raw_frames),
                original_frame_idx=frame_idx,
                pts_sec=pts_sec,
                is_pts_exact=pts_msec > 0,
                image_bgr=frame,
            )
            raw_frames.append(ref)
            frame_idx += 1

        cap.release()

        if not raw_frames:
            return []

        # If we have more frames than target_frame_count, sample evenly or take initial block
        if len(raw_frames) > target_frame_count:
            step_indices = np.linspace(0, len(raw_frames) - 1, target_frame_count, dtype=int)
            return [raw_frames[i] for i in step_indices]

        return raw_frames
