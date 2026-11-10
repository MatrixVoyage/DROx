"""FastAPI application for Deepfake Video Detection inference and technical dashboard."""

import os
import shutil
import tempfile
import re
import base64
import cv2
import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, List
from pydantic import BaseModel

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.inference.predictor import VideoPredictor, PredictionResult, EnsemblePredictor
from src.forensics.spatial_temporal_pipeline import SpatialTemporalForensicEngine
from src.forensics.context import ForensicAnalysisConfig
from src.utils.logger import setup_logger

logger = setup_logger("api_service")

AVAILABLE_ENGINES: Dict[str, str] = {
    "v1-baseline": "checkpoints/ffpp_c23_baseline_soft_lstm/best.pth",
    "v2-evolved": "checkpoints/m10_selected_candidate/best.pth",
    "v3-robust": "checkpoints/m12_full/best.pth",
    "v4-generalization": "checkpoints/m13_2b_stage5/best.pth",
    "v5-family-invariant": "checkpoints/m13_3_f2f_seed42/best.pth",
    "m17-sensor-residuals": "checkpoints/m17_track_a_seed42/best.pth",
    "m18-dual-stream": "checkpoints/m18_dual_stream_seed42/best.pth",
    "m22-ensemble": "checkpoints/m22_ensemble/fusion_config.json",
}

# Global predictor registry supporting multiple engines
_predictors: Dict[str, Any] = {}
_predictor: Optional[Any] = None


def get_predictor(engine: str = "v1-baseline") -> Any:
    """Get or initialize the predictor instance for the specified engine."""
    global _predictor, _predictors

    # If mock predictor injected by test fixture, return it
    if _predictor is not None:
        return _predictor

    if engine not in AVAILABLE_ENGINES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported engine '{engine}'. Available engines: {list(AVAILABLE_ENGINES.keys())}",
        )

    if engine in _predictors:
        return _predictors[engine]

    if engine == "m22-ensemble":
        pred_m13 = get_predictor("v5-family-invariant")
        pred_m18 = get_predictor("m18-dual-stream")
        cfg_path = os.environ.get("DEEPFAKE_M22_CONFIG_PATH", AVAILABLE_ENGINES["m22-ensemble"])
        predictor = EnsemblePredictor(
            predictor_m13=pred_m13,
            predictor_m18=pred_m18,
            fusion_config_path=cfg_path,
        )
        _predictors[engine] = predictor
        return predictor

    if engine == "v1-baseline":
        checkpoint_path = os.environ.get("DEEPFAKE_CHECKPOINT_PATH", AVAILABLE_ENGINES["v1-baseline"])
    elif engine == "v2-evolved":
        checkpoint_path = os.environ.get("DEEPFAKE_V2_CHECKPOINT_PATH", AVAILABLE_ENGINES["v2-evolved"])
    elif engine == "v3-robust":
        checkpoint_path = os.environ.get("DEEPFAKE_V3_CHECKPOINT_PATH", AVAILABLE_ENGINES["v3-robust"])
    elif engine == "v4-generalization":
        checkpoint_path = os.environ.get("DEEPFAKE_V4_CHECKPOINT_PATH", AVAILABLE_ENGINES["v4-generalization"])
    elif engine == "v5-family-invariant":
        checkpoint_path = os.environ.get("DEEPFAKE_V5_CHECKPOINT_PATH", AVAILABLE_ENGINES["v5-family-invariant"])
    elif engine == "m17-sensor-residuals":
        checkpoint_path = os.environ.get("DEEPFAKE_M17_CHECKPOINT_PATH", AVAILABLE_ENGINES["m17-sensor-residuals"])
    elif engine == "m18-dual-stream":
        checkpoint_path = os.environ.get("DEEPFAKE_M18_CHECKPOINT_PATH", AVAILABLE_ENGINES["m18-dual-stream"])
    else:
        checkpoint_path = AVAILABLE_ENGINES[engine]

    device = os.environ.get("DEEPFAKE_DEVICE", None)
    logger.info(f"Initializing VideoPredictor [{engine}] from checkpoint: {checkpoint_path} (device={device})")
    if not Path(checkpoint_path).is_file():
        logger.warning(f"Checkpoint not found at {checkpoint_path}; running in uninitialized mode")
        raise RuntimeError(f"Checkpoint not found at {checkpoint_path}")

    predictor = VideoPredictor(checkpoint_path=checkpoint_path, device=device)
    _predictors[engine] = predictor
    return predictor


def set_predictor(predictor: Optional[VideoPredictor]) -> None:
    """Set or override the global predictor (useful for testing)."""
    global _predictor
    _predictor = predictor


_forensic_engine: Optional[Any] = None


def get_forensic_engine() -> Any:
    """Get or initialize the SpatialTemporalForensicEngine instance."""
    global _forensic_engine
    if _forensic_engine is not None:
        return _forensic_engine
    device = os.environ.get("DEEPFAKE_DEVICE", None)
    logger.info(f"Initializing SpatialTemporalForensicEngine (device={device})...")
    _forensic_engine = SpatialTemporalForensicEngine(device=device)
    return _forensic_engine


def set_forensic_engine(engine: Optional[Any]) -> None:
    """Set or override the global forensic engine (useful for testing)."""
    global _forensic_engine
    _forensic_engine = engine


def saliency_to_base64_heatmap(saliency: np.ndarray) -> str:
    """Encodes 2D float32 saliency map [0, 1] into a base64 PNG JET heatmap."""
    if saliency is None or saliency.size == 0:
        return ""
    try:
        norm = np.clip(saliency, 0.0, 1.0)
        uint8_map = (norm * 255.0).astype(np.uint8)
        color_map = cv2.applyColorMap(uint8_map, cv2.COLORMAP_JET)
        ret, buf = cv2.imencode(".png", color_map)
        if not ret:
            return ""
        return base64.b64encode(buf).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to generate saliency heatmap base64: {e}")
        return ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle context manager to optionally preload default model on startup."""
    checkpoint_path = os.environ.get(
        "DEEPFAKE_CHECKPOINT_PATH",
        AVAILABLE_ENGINES["v1-baseline"],
    )
    if Path(checkpoint_path).is_file():
        try:
            get_predictor("v1-baseline")
            logger.info("Deepfake VideoPredictor [v1-baseline] successfully warmed up.")
        except Exception as e:
            logger.error(f"Failed to warm up baseline predictor: {e}")
    else:
        logger.warning(f"Baseline checkpoint '{checkpoint_path}' not present at startup.")
    yield
    # Cleanup if necessary


app = FastAPI(
    title="Deepfake Video Detection API",
    description="REST API and interactive dashboard for deepfake video detection supporting v1-baseline and v2-evolved architectures",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for local dashboards and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="Health Check and Service Status")
async def health_check() -> Dict[str, Any]:
    """Return service health, model status, device, and version."""
    model_loaded = False
    device = "unknown"
    checkpoint_name = "none"

    if _predictor is not None:
        model_loaded = True
        device = str(_predictor.device)
        checkpoint_name = str(_predictor.checkpoint_path.name)
    elif "v1-baseline" in _predictors:
        model_loaded = True
        device = str(_predictors["v1-baseline"].device)
        checkpoint_name = str(_predictors["v1-baseline"].checkpoint_path.name)
    else:
        ckpt = os.environ.get("DEEPFAKE_CHECKPOINT_PATH", AVAILABLE_ENGINES["v1-baseline"])
        if Path(ckpt).is_file():
            checkpoint_name = Path(ckpt).name

    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": device,
        "checkpoint": checkpoint_name,
        "default_engine": "v5-family-invariant",
        "champion_engine": "v5-family-invariant",
        "ensemble_engine": "m22-ensemble",
        "available_engines": list(AVAILABLE_ENGINES.keys()),
        "version": "2.1.0",
    }


@app.get("/api/v1/model-info", summary="Model Architecture and Provenance")
async def get_model_info(engine: str = "v1-baseline") -> Dict[str, Any]:
    """Return model architecture details, research constraints, and citation."""
    pred = None
    try:
        pred = get_predictor(engine=engine)
        backbone_name = pred.config.model.backbone.name
        attn_type = pred.config.model.attention.type
        temp_type = pred.config.model.temporal.type
        num_frames = getattr(pred.config.model, "input_frames", 16)
        img_size = getattr(pred.config.model, "input_size", 112)
        device = str(pred.device)
        checkpoint_p = str(pred.checkpoint_path)
    except Exception:
        if engine in ("v5-family-invariant", "v4-generalization", "v3-robust", "v2-evolved"):
            backbone_name = "efficientnet_b0"
            attn_type = "identity"
            temp_type = "transformer"
        else:
            backbone_name = "resnext50_32x4d"
            attn_type = "soft"
            temp_type = "lstm"
        num_frames = 16
        img_size = 112
        device = "cuda" if os.environ.get("DEEPFAKE_DEVICE") != "cpu" else "cpu"
        checkpoint_p = AVAILABLE_ENGINES.get(engine, AVAILABLE_ENGINES["v1-baseline"])

    display_names = {
        "v1-baseline": "v1-baseline",
        "v2-evolved": "v2-evolved (M10)",
        "v3-robust": "v3-robust (M12)",
        "v4-generalization": "v4-generalization (M13.2b)",
        "v5-family-invariant": "v5-family-invariant (M13.4 LOFO-F2F Checkpoint)",
        "m17-sensor-residuals": "m17-sensor-residuals (M17 Track A)",
        "m18-dual-stream": "m18-dual-stream (M18 Kinematics)",
        "m22-ensemble": "m22-ensemble (M22 Log-Odds Fusion)",
    }
    display_name = display_names.get(engine, engine)

    checkpoint_provenance = None
    default_thresh = 0.5
    if engine == "m22-ensemble":
        arch_title = "Calibrated Log-Odds Platt Fusion [Champion M13.5 + Baseline M18]"
        training_dataset = "FF++ C23 + Celeb-DF v2 (Validation Calibrated, Zero Neural Retraining)"
        default_thresh = 0.9191
        checkpoint_provenance = {
            "model_id": "m22-ensemble",
            "status": "Highest-Accuracy Research Ensemble Configuration (All-Time Project Record)",
            "benchmark_results": {
                "face2face_auc": 0.8118,
                "celeb_df_v2_auc": 0.8213,
                "jpeg_q50_auc": 0.8944,
                "h264_crf23_auc": 0.7017,
                "clean_specificity": 0.939,
            },
            "parameters": {
                "w1_m13": 0.4092,
                "w2_m18": 0.3590,
                "bias": 1.4219,
                "tau_95": 0.9191,
            },
            "note": "Combines static frequency/texture signals with 3D differential kinematics without modifying neural weights.",
        }
    elif engine == "m18-dual-stream":
        arch_title = "Frozen M17 Track A Trunk + 3D Differential Kinematics + Dynamic Gating"
        training_dataset = "FaceForensics++ C23 (Targeted Reenactment Representation)"
        checkpoint_provenance = {
            "model_id": "m18-dual-stream",
            "status": "Reenactment Research Baseline (M18)",
            "benchmark_results": {
                "face2face_auc": 0.7858,
                "celeb_df_v2_auc": 0.8166,
            },
        }
    elif engine == "m17-sensor-residuals":
        arch_title = "EfficientNet-B0 + SRM + Bayar Multi-Scale Residuals"
        training_dataset = "FaceForensics++ C23 (Sensor Residual Representation)"
        checkpoint_provenance = {
            "model_id": "m17-sensor-residuals",
            "status": "Sensor Noise Baseline (M17 Track A)",
        }
    elif engine == "v5-family-invariant":
        arch_title = f"{backbone_name} + {temp_type.upper()} + LocalFreq + LocalTexture + MultiDomainFusion"
        training_dataset = "FaceForensics++ C23 (LOFO-F2F) + Celeb-DF v2"
        checkpoint_provenance = {
            "model_id": "v5-family-invariant",
            "status": "Promoted Research Champion for the M13 Family-Invariant Architecture (Production Research Engine)",
            "single_deployed_checkpoint": {
                "checkpoint": "checkpoints/m13_3_f2f_seed42/best.pth",
                "training_fold": "LOFO-F2F (seen: Deepfakes, FaceSwap, NeuralTextures; held-out: Face2Face)",
                "held_out_f2f_test_auc": 0.7903,
                "clean_specificity_tau_0_5": 0.9388,
            },
            "research_family_transfer_study": {
                "protocol": "4-Fold Leave-One-Family-Out Matrix",
                "models": "Four independently trained LOFO checkpoints",
                "macro_mean_auc": 0.9165,
                "note": "0.9165 is the aggregate research-matrix statistic across 4 distinct models, NOT the performance of the single deployed checkpoint.",
            },
            "training_fold": "LOFO-F2F (seen: Deepfakes, FaceSwap, NeuralTextures; held-out: Face2Face)",
            "held_out_f2f_test_auc": 0.7903,
            "lofo_matrix_macro_mean_auc": 0.9165,
            "note": "0.9165 is the aggregate macro-mean across 4 distinct models, NOT the performance of the single deployed checkpoint.",
        }
    elif engine == "v4-generalization":
        arch_title = f"{backbone_name} + {temp_type.upper()} + Stage-5 MidHook + AuxHead"
        training_dataset = "FaceForensics++ C23 + Celeb-DF v2"
    elif engine == "v3-robust":
        arch_title = f"{backbone_name} + Identity Attention + {temp_type.upper()}"
        training_dataset = "FaceForensics++ C23 + Celeb-DF v2"
    elif engine == "v2-evolved":
        arch_title = f"{backbone_name} + Identity Attention + {temp_type.upper()}"
        training_dataset = "FaceForensics++ C23"
    else:
        arch_title = f"{backbone_name} + Soft Attention + LSTM"
        training_dataset = "FaceForensics++ C23"

    calib_meta = getattr(pred, "calibration_metadata", None) if pred else None

    resp: Dict[str, Any] = {
        "engine": engine,
        "display_name": display_name,
        "available_engines": list(AVAILABLE_ENGINES.keys()),
        "architecture": arch_title,
        "backbone": backbone_name,
        "attention_module": attn_type,
        "temporal_module": temp_type,
        "input_resolution": [img_size, img_size] if isinstance(img_size, int) else img_size,
        "sequence_length": num_frames,
        "checkpoint_path": checkpoint_p,
        "training_dataset": training_dataset,
        "reference_paper": "Lal et al., 2025 (IJCNN) — Facial Deepfake Detection Using Spatiotemporal Networks",
        "default_threshold": default_thresh,
        "active_device": device,
        "calibration": calib_meta,
        "research_limitations": [
            "Inference operates on 16 sampled frames per video clip.",
            "Face detection uses MTCNN; videos with obstructed faces or no detectable face may trigger REJECT state.",
            "Facial Reenactment Manifold Boundary: Subtle expression modifications (Face2Face) operate inside the natural human facial subspace without blending seams, yielding ~38-43% sensitivity compared to 89-100% on Deepfakes/NeuralTextures.",
            "Handcrafted Branch Gating: Standalone frequency and texture branches have elevated false-alarm rates (28-47% on clean reals); their noisy anomaly signals are gated/suppressed by the multi-domain fusion head.",
            "Threshold Sensitivity Trade-Off: In the production temperature-scaled probability space (T=2.0502), high-sensitivity operating point (tau*=0.1260, corresponding to tau=0.0190 in raw sigmoid space) derived strictly on validation clean reals increases Face2Face recall to 43.10% (25/58) while maintaining 93.88% (92/98) clean specificity (6 false alarms per 98 clean reals).",
            "Cross-Dataset Generalization: Cross-domain transfer to Celeb-DF v2 yields ~0.796 AUC, showing expected domain-shift degradation relative to seen synthesis families (>0.98 AUC).",
            "Probabilistic Confidence Semantics: Model scores represent statistical confidence within the validated distribution, not objective physical proof or sole legal evidence. Avoid treating high decision margins as epistemic certainty.",
            "OOD Status: Out-of-Distribution feature distance is an experimental diagnostic indicator (Mahalanobis distance on penultimate embeddings against 343 validation clean reals), not a validated production rejection system.",
        ],
    }
    if checkpoint_provenance:
        resp["checkpoint_provenance"] = checkpoint_provenance
    return resp


@app.post("/api/v1/predict", summary="Predict Deepfake Probability for Uploaded Video")
async def predict_video(
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV, MKV)"),
    threshold: Optional[float] = Form(0.5, description="Custom classification threshold"),
    engine: Optional[str] = Form("v1-baseline", description="Inference engine ('v1-baseline', 'v2-evolved', 'v3-robust', 'v4-generalization', or 'v5-family-invariant')"),
    calibrated: Optional[bool] = Form(True, description="Enable calibrated 3-state rejection policy"),
) -> Dict[str, Any]:
    """Upload a video file and evaluate deepfake probability."""
    filename = file.filename or "uploaded_video.mp4"
    ext = Path(filename).suffix.lower()
    valid_exts = {".mp4", ".avi", ".mov", ".mkv"}
    if ext not in valid_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {sorted(list(valid_exts))}",
        )

    try:
        active_engine = engine or "v1-baseline"
        predictor = get_predictor(engine=active_engine)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model predictor service unavailable: {e}",
        )

    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            shutil.copyfileobj(file.file, tmp_file)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read and store uploaded file: {e}",
            )

    try:
        active_threshold = threshold if threshold is not None else 0.5
        
        # Derive a stable, sanitized video_id from the original filename to guarantee deterministic sampling
        safe_stem = re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(filename).stem)
        if not safe_stem:
            safe_stem = "uploaded_video"

        result: PredictionResult = predictor.predict_video(
            video_path=tmp_path,
            threshold=active_threshold,
            video_id=safe_stem,
            calibrated=bool(calibrated),
        )
        if result.verdict == "Error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Video processing failed: {result.error}",
            )
        res_dict = result.to_dict()
        res_dict["original_filename"] = filename
        res_dict["engine"] = active_engine
        if not res_dict.get("decision"):
            res_dict["decision"] = "FAKE" if result.verdict == "Deepfake" else ("REAL" if result.verdict == "Real" else "REJECT")
        return res_dict

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference prediction failed for {filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during video processing: {e}",
        )
    finally:
        # Cleanup temporary uploaded file
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete temporary file {tmp_path}: {e}")


@app.post("/api/v1/compare", summary="Compare uploaded video across Champion M13.5, Baseline M18, and Fused M22 Ensemble")
async def compare_video(
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV, MKV)"),
    threshold: Optional[float] = Form(None, description="Custom classification threshold override"),
    calibrated: Optional[bool] = Form(True, description="Enable calibrated 3-state rejection policy"),
) -> Dict[str, Any]:
    """Upload a video file and evaluate deepfake probability across multiple engines simultaneously.

    Returns side-by-side comparative results and discordance analysis between
    Champion M13.5 (static spatial artifacts), Baseline M18 (kinematic motion),
    and Fused M22 Ensemble.
    """
    filename = file.filename or "uploaded_video.mp4"
    ext = Path(filename).suffix.lower()
    valid_exts = {".mp4", ".avi", ".mov", ".mkv"}
    if ext not in valid_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {sorted(list(valid_exts))}",
        )

    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            shutil.copyfileobj(file.file, tmp_file)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read and store uploaded file: {e}",
            )

    try:
        safe_stem = re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(filename).stem) or "uploaded_video"

        # If a single mock predictor is injected by test fixture, reuse it
        if _predictor is not None:
            pred_m13 = _predictor
            pred_m18 = _predictor
            pred_m22 = _predictor
        else:
            pred_m13 = get_predictor("v5-family-invariant")
            pred_m18 = get_predictor("m18-dual-stream")
            pred_m22 = get_predictor("m22-ensemble")

        # For M22 ensemble, its calibrated operating policy is tau_95 (0.9191).
        # Use 0.9191 unless a custom threshold overriding the standard 0.5 default is specified.
        thresh_m22 = 0.9191 if (threshold is None or abs(float(threshold) - 0.50) < 1e-4) else float(threshold)

        res_m13 = pred_m13.predict_video(video_path=tmp_path, threshold=threshold, video_id=safe_stem, calibrated=bool(calibrated))
        res_m18 = pred_m18.predict_video(video_path=tmp_path, threshold=threshold, video_id=safe_stem, calibrated=bool(calibrated))
        res_m22 = pred_m22.predict_video(video_path=tmp_path, threshold=thresh_m22, video_id=safe_stem, calibrated=bool(calibrated))

        p_m13 = float(res_m13.probability)
        p_m18 = float(res_m18.probability)
        p_m22 = float(res_m22.probability)

        d_m13 = res_m13.decision or ("FAKE" if res_m13.verdict == "Deepfake" else "REAL")
        d_m18 = res_m18.decision or ("FAKE" if res_m18.verdict == "Deepfake" else "REAL")
        d_m22 = res_m22.decision or ("FAKE" if res_m22.verdict == "Deepfake" else "REAL")

        is_reject = any(r.verdict == "Reject" for r in (res_m13, res_m18, res_m22))
        all_fake = all(d == "FAKE" for d in (d_m13, d_m18, d_m22))
        all_real = all(d == "REAL" for d in (d_m13, d_m18, d_m22))

        if is_reject:
            consensus = "REJECT_OR_UNCERTAIN"
        elif all_fake:
            consensus = "CONCORDANT_FAKE"
        elif all_real:
            consensus = "CONCORDANT_REAL"
        else:
            consensus = "DISCORDANT"

        if is_reject:
            summary = "One or more engines rejected the sample due to safety or quality bounds."
        elif all_fake:
            summary = "All engines concordantly detect deepfake manipulation evidence across both spatial and kinematic domains."
        elif all_real:
            summary = "All engines concordantly detect authentic presentation within validated distribution bounds."
        elif d_m13 == "FAKE" and d_m18 == "REAL":
            summary = "Spatial anomaly detected by Champion M13.5, but kinematic differential stream (M18) observes natural rigid motion."
        elif d_m18 == "FAKE" and d_m13 == "REAL":
            summary = "Kinematic differential stream (M18) detects subtle non-rigid reenactment warping, while spatial branch (M13.5) remains below threshold."
        else:
            summary = "Disagreement observed between individual models; resolved via calibrated log-odds fusion (M22)."

        resp = {
            "video_filename": filename,
            "consensus": consensus,
            "discordance_summary": summary,
            "engines": {
                "v5-family-invariant": res_m13.to_dict(),
                "m18-dual-stream": res_m18.to_dict(),
                "m22-ensemble": res_m22.to_dict(),
            },
            "signals": {
                "m13_5_probability": round(p_m13, 4),
                "m18_probability": round(p_m18, 4),
                "m22_fused_probability": round(p_m22, 4),
                "decision_m13": d_m13,
                "decision_m18": d_m18,
                "decision_m22": d_m22,
            },
        }
        return resp
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


@app.post("/api/v1/forensic-audit", summary="Run Spatiotemporal Forensic Audit & Saliency Analysis")
async def forensic_audit_video(
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV, MKV)"),
    threshold: Optional[float] = Form(0.5, description="Decision threshold"),
    windows: Optional[int] = Form(4, description="Maximum temporal windows to evaluate"),
    window_duration: Optional[float] = Form(2.0, description="Duration in seconds per window"),
    frames_per_window: Optional[int] = Form(16, description="Frames sampled per window"),
) -> Dict[str, Any]:
    """Execute end-to-end spatiotemporal multi-window forensic analysis on video."""
    filename = file.filename or "uploaded_video.mp4"
    ext = Path(filename).suffix.lower()
    valid_exts = {".mp4", ".avi", ".mov", ".mkv"}
    if ext not in valid_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {sorted(list(valid_exts))}",
        )

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            shutil.copyfileobj(file.file, tmp_file)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read and store uploaded file: {e}",
            )

    try:
        engine = get_forensic_engine()
        cfg = ForensicAnalysisConfig(
            max_windows=int(windows or 4),
            window_duration_sec=float(window_duration or 2.0),
            frames_per_window=int(frames_per_window or 16),
            confidence_threshold=float(threshold if threshold is not None else 0.50),
        )
        report = engine.analyze_video(tmp_path, config=cfg, threshold=threshold, original_filename=filename)
        rep_dict = report.to_dict()
        rep_dict["video_filename"] = filename
        if rep_dict.get("structured_telemetry") and rep_dict["structured_telemetry"].get("stream"):
            rep_dict["structured_telemetry"]["stream"]["file_name"] = filename
            rep_dict["structured_telemetry"]["stream"]["source_file_name"] = filename
        rep_dict["decision_threshold"] = float(threshold if threshold is not None else 0.50)
        rep_dict["saliency_heatmap_base64"] = saliency_to_base64_heatmap(report.overall_spatial_saliency)

        # AI Forensic Analyst synthesis via Groq (Qwen 3.8 27B / GPT-OSS)
        try:
            from src.forensics.groq_analyst import generate_human_friendly_forensic_analysis, is_groq_available
            if is_groq_available():
                groq_report = generate_human_friendly_forensic_analysis(
                    audit_data=rep_dict,
                    threshold=float(threshold if threshold is not None else 0.50),
                )
                rep_dict["groq_analyst_report"] = groq_report
            else:
                rep_dict["groq_analyst_report"] = None
        except Exception as groq_err:
            logger.warning(f"Groq forensic synthesis error during audit: {groq_err}")
            rep_dict["groq_analyst_report"] = None

        return rep_dict
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forensic audit failed for {filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forensic audit error during video processing: {e}",
        )
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


class GroqSynthesisRequest(BaseModel):
    audit_data: Dict[str, Any]
    prediction_data: Optional[Dict[str, Any]] = None
    threshold: Optional[float] = None
    model: Optional[str] = None


@app.post("/api/v1/groq-analyst-synthesis", summary="Generate human-friendly forensic analysis via Groq")
async def groq_analyst_synthesis(req: GroqSynthesisRequest):
    """Generates an explainable, human-friendly forensic narrative using Groq LLM."""
    try:
        from src.forensics.groq_analyst import (
            generate_human_friendly_forensic_analysis,
            is_groq_available,
            build_canonical_evidence,
            validate_groq_report,
            get_report_validation_details,
        )
        if not is_groq_available():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Groq API key is not configured in .env or environment.",
            )
        evidence = build_canonical_evidence(
            audit_data=req.audit_data,
            pred_data=req.prediction_data,
            threshold=req.threshold,
        )
        report, is_fallback = generate_human_friendly_forensic_analysis(
            audit_data=req.audit_data,
            pred_data=req.prediction_data,
            threshold=req.threshold,
            preferred_model=req.model,
            return_fallback_flag=True,
        )
        is_valid, validation_level, issues = get_report_validation_details(
            report, evidence, is_fallback=is_fallback
        )
        return {
            "success": True,
            "groq_report": report,
            "model": req.model or "openai/gpt-oss-120b",
            "validated": is_valid,
            "validation_level": validation_level,
            "validation_status": validation_level,
            "validation_issues": issues,
            "canonical_evidence": evidence.to_dict(),
            "is_fallback": is_fallback,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Groq synthesis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Groq synthesis failed: {e}",
        )


# Secure Forensic Artifact Serving with Path Containment
ARTIFACT_ROOT = (Path(__file__).resolve().parent.parent.parent / "reports" / "artifacts").resolve()
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


def resolve_artifact(filename: str) -> Path:
    """Safely resolves an artifact filename strictly within ARTIFACT_ROOT, guarding against traversal."""
    if not filename or Path(filename).name != filename:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    candidate = (ARTIFACT_ROOT / filename).resolve()
    try:
        candidate.relative_to(ARTIFACT_ROOT)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    if not candidate.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    return candidate


@app.get("/artifacts/forensics/{filename}", summary="Securely serve forensic visual artifact")
async def get_forensic_artifact(filename: str):
    """Serve visual forensic artifacts (heatmaps, saliency overlays, trajectory plots) with path containment."""
    path = resolve_artifact(filename)
    return FileResponse(str(path))


# Static UI Mount
ui_dir = Path(__file__).resolve().parent.parent / "ui"
if ui_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")

    @app.get("/", summary="Serve Technical Dashboard")
    async def get_dashboard():
        """Serve the technical dashboard HTML."""
        index_html = ui_dir / "index.html"
        if index_html.is_file():
            return FileResponse(str(index_html))
        return JSONResponse({"message": "Technical dashboard assets not found in src/ui."})
