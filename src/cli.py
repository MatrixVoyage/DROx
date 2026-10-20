"""Command-line interface for environment diagnostics, manifest splitting, and video preprocessing."""

import argparse
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.utils.config import AppConfig, load_config

from src.utils.logger import setup_logger
from src.utils.seed import seed_everything
from src.data.manifest import load_manifest, save_manifest, split_manifest_by_group, validate_manifest, validate_split_integrity
from src.data.pipeline import VideoFacePipeline
from src.data.storage import ProcessedDataStorage
from src.data.dataset import ProcessedVideoDataset, create_dataloader
from src.data.adapters.celeb_df import build_celebdf_manifest
from src.data.adapters.dfdc import build_dfdc_manifest_from_metadata
from src.data.adapters.ffpp import build_ffpp_manifest
from src.models.detector import DeepfakeDetector
from src.training.checkpoint import CheckpointManager
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_metrics
from src.evaluation.cross_dataset import CrossDatasetEvaluator
from src.evaluation.reporting import BenchmarkReporter
from src.experiments.runner import ExperimentRunner



def verify_env() -> None:
    """Check and display environment, PyTorch, CUDA, and MTCNN diagnostic details."""
    logger = setup_logger("verify_env")
    logger.info("=== Deepfake Detection Environment Diagnostics ===")
    logger.info(f"Python Version: {sys.version.split()[0]}")

    # PyTorch
    try:
        import torch
        logger.info(f"PyTorch Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA Available: {cuda_available}")
        if cuda_available:
            logger.info(f"CUDA Version: {torch.version.cuda}")
            gpu_name = torch.cuda.get_device_name(0)
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU Name: {gpu_name}")
            logger.info(f"Total VRAM: {total_vram_gb:.2f} GB")
        else:
            logger.warning("CUDA is not available. Execution will fall back to CPU.")
    except ImportError:
        logger.error("PyTorch is NOT installed.")

    # Torchvision
    try:
        import torchvision
        logger.info(f"Torchvision Version: {torchvision.__version__}")
    except ImportError:
        logger.error("Torchvision is NOT installed.")

    # MTCNN / facenet-pytorch
    try:
        from facenet_pytorch import MTCNN
        logger.info("facenet-pytorch MTCNN: Available")
    except ImportError:
        logger.error("facenet-pytorch is NOT installed.")

    # OpenCV
    try:
        import cv2
        logger.info(f"OpenCV Version: {cv2.__version__}")
    except ImportError:
        logger.error("OpenCV is NOT installed.")

    logger.info("==================================================")


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Preprocess single video or manifest."""
    logger = setup_logger("preprocess")
    config_path = args.config if args.config else "configs/default.yaml"
    config = load_config(config_path)

    if args.output_dir:
        config.storage.processed_dir = args.output_dir

    seed_everything(config.seed)
    storage = ProcessedDataStorage(config.storage.processed_dir)
    storage.save_run_config(config)

    pipeline = VideoFacePipeline(config=config)

    if args.video:
        video_path = Path(args.video)
        logger.info(f"Processing single video: {video_path}")
        result = pipeline.process_video(
            video_path=video_path,
            video_id=args.video_id or video_path.stem,
            group_id=args.group_id or video_path.stem,
            split=args.split or "test",
        )
        t_path, m_path = storage.save(result, split=args.split or "test", config_snapshot=config.to_dict())
        if result.status == "success":
            logger.info(f"Success! Tensor saved to: {t_path}, Metadata: {m_path}")
        else:
            logger.warning(f"Video {result.status}: {result.error_message}. Metadata: {m_path}")

    elif args.manifest:
        manifest_path = Path(args.manifest)
        logger.info(f"Processing dataset manifest: {manifest_path}")
        df = load_manifest(manifest_path)
        validate_manifest(df, check_paths=False)

        success_count = 0
        rejected_count = 0
        failed_count = 0

        for idx, row in df.iterrows():
            vid = str(row["video_id"])
            gid = str(row["group_id"])
            vpath = Path(row["video_path"])
            split = str(row["split"]) if "split" in row and pd.notna(row["split"]) and str(row["split"]).strip() else (args.split or "train")

            result = pipeline.process_video(
                video_path=vpath,
                video_id=vid,
                group_id=gid,
                split=split,
            )
            storage.save(result, split=split, config_snapshot=config.to_dict())

            if result.status == "success":
                success_count += 1
            elif result.status.startswith("rejected"):
                rejected_count += 1
            else:
                failed_count += 1

        logger.info(
            f"Preprocess complete. Success: {success_count}, "
            f"Rejected: {rejected_count}, Failed: {failed_count}"
        )
    else:
        logger.error("Must specify either --video or --manifest")
        sys.exit(1)


def cmd_manifest_split(args: argparse.Namespace) -> None:
    """Split manifest by indivisible group stratification."""
    logger = setup_logger("manifest_split")
    input_path = Path(args.input)
    output_path = Path(args.output)

    logger.info(f"Loading manifest from: {input_path}")
    df = load_manifest(input_path)

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    logger.info(f"Splitting with ratios: train={ratios[0]}, val={ratios[1]}, test={ratios[2]}")

    result_df = split_manifest_by_group(df, ratios=ratios, seed=args.seed)
    save_manifest(result_df, output_path)

    logger.info(f"Saved stratified manifest with zero group leakage to: {output_path}")
    for s in ["train", "val", "test"]:
        cnt = (result_df["split"] == s).sum()
        real_cnt = ((result_df["split"] == s) & (result_df["label"] == 0)).sum()
        fake_cnt = ((result_df["split"] == s) & (result_df["label"] == 1)).sum()
        logger.info(f"  Split '{s}': {cnt} total (real: {real_cnt}, fake: {fake_cnt})")


def cmd_inspect_backbone(args: argparse.Namespace) -> None:
    """Inspect ResNeXt-50 backbone parameters, memory footprint, and provenance."""
    import torch
    from src.models.backbones.resnext import ResNeXt50Backbone

    logger = setup_logger("inspect_backbone")
    config_path = args.config if args.config else "configs/default.yaml"
    config = load_config(config_path)

    # Optional overrides from CLI
    if args.freeze_strategy:
        config.model.backbone.freeze_strategy = args.freeze_strategy
    if args.no_pretrained:
        config.model.backbone.weights = None

    logger.info("Initializing ResNeXt-50 backbone...")
    model = ResNeXt50Backbone(config=config.model.backbone)
    model.eval()

    prov = model.get_provenance()
    logger.info("=== Backbone Provenance ===")
    for k, v in prov.items():
        logger.info(f"  {k}: {v}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    logger.info("=== Parameter Counts ===")
    logger.info(f"  Total parameters:     {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    logger.info(f"  Frozen parameters:    {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")

    # Memory / benchmark pass
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Running test forward pass on {device} (batch_size={args.batch_size}, frames={args.num_frames})...")
    model = model.to(device)
    dummy_input = torch.randn(args.batch_size, args.num_frames, 3, 112, 112, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    with torch.inference_mode():
        pooled, spatial = model(dummy_input, return_spatial_map=True)

    logger.info("=== Forward Pass Output ===")
    logger.info(f"  Input shape:   {tuple(dummy_input.shape)}")
    logger.info(f"  Pooled shape:  {tuple(pooled.shape)}")
    logger.info(f"  Spatial shape: {tuple(spatial.shape)}")

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info("=== CUDA Memory Profile ===")
        logger.info(f"  Allocated: {allocated:.2f} MB")
        logger.info(f"  Reserved:  {reserved:.2f} MB")
        logger.info(f"  Peak:      {peak:.2f} MB")


def cmd_inspect_attention(args: argparse.Namespace) -> None:
    """Inspect attention module parameters, output shapes, and memory footprint."""
    import torch
    from src.models.attention.factory import build_attention_module

    logger = setup_logger("inspect_attention")
    config_path = args.config if args.config else "configs/default.yaml"
    config = load_config(config_path)

    if args.attention_type:
        config.model.attention.type = args.attention_type

    attn_type = config.model.attention.type
    classification = (
        "Paper-Aligned Baseline (Lal et al., 2025)"
        if attn_type == "soft"
        else "Project-Level Ablation"
    )

    logger.info(f"Initializing Attention Module (type='{attn_type}')...")
    model = build_attention_module(config.model.attention)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("=== Attention Module Metadata ===")
    logger.info(f"  Module Class:         {model.__class__.__name__}")
    logger.info(f"  Attention Type:       {attn_type}")
    logger.info(f"  Classification:       {classification}")
    logger.info(f"  Total Parameters:     {total_params:,}")
    logger.info(f"  Trainable Parameters: {trainable_params:,}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Running test forward pass on {device} (batch_size={args.batch_size}, frames={args.num_frames})...")
    model = model.to(device)

    dummy_input = torch.randn(args.batch_size, args.num_frames, 2048, 4, 4, device=device)

    # Inference pass with map extraction
    model.eval()
    with torch.no_grad():
        out, maps = model(dummy_input, return_attention_maps=True)

    logger.info("=== Forward Pass Output ===")
    logger.info(f"  Input shape:          {tuple(dummy_input.shape)}")
    logger.info(f"  Context vector shape: {tuple(out.shape)}")
    logger.info(f"  Attention map keys:   {list(maps.keys())}")
    for k, v in maps.items():
        logger.info(f"    Map '{k}': shape {tuple(v.shape)}")

    # VRAM profiling on CUDA
    if device.type == "cuda":
        # 1. FP32 Forward Peak
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        x_fp32 = torch.randn(
            args.batch_size, args.num_frames, 2048, 4, 4, device=device, requires_grad=True
        )
        model.train()
        out_fp32 = model(x_fp32)
        fp32_fwd_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # 2. FP32 Forward + Backward Peak
        loss_fp32 = out_fp32.sum()
        loss_fp32.backward()
        fp32_bwd_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # 3. FP16 AMP Forward Peak
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        x_fp16 = torch.randn(
            args.batch_size, args.num_frames, 2048, 4, 4, device=device, requires_grad=True
        )
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out_fp16 = model(x_fp16)
        fp16_fwd_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # 4. FP16 AMP Forward + Backward Peak
        loss_fp16 = out_fp16.sum()
        loss_fp16.backward()
        fp16_bwd_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        logger.info("=== CUDA Memory Profile ===")
        logger.info(f"  FP32 Forward Peak:            {fp32_fwd_peak:.2f} MB")
        logger.info(f"  FP32 Forward + Backward Peak: {fp32_bwd_peak:.2f} MB")
        logger.info(f"  FP16 AMP Forward Peak:        {fp16_fwd_peak:.2f} MB")
        logger.info(f"  FP16 AMP Fwd + Bwd Peak:      {fp16_bwd_peak:.2f} MB")


def cmd_inspect_model(args: argparse.Namespace) -> None:
    """Inspect complete DeepfakeDetector architecture, parameter distribution, and memory footprint."""
    import torch
    from src.models.detector import DeepfakeDetector

    logger = setup_logger("inspect_model")
    config_path = args.config if args.config else "configs/default.yaml"
    config = load_config(config_path)

    # Apply CLI overrides
    if args.no_pretrained:
        config.model.backbone.weights = None
    if args.freeze_strategy:
        config.model.backbone.freeze_strategy = args.freeze_strategy
    if args.attention_type:
        config.model.attention.type = args.attention_type
    if args.temporal_type:
        config.model.temporal.type = args.temporal_type
        if args.temporal_type == "bilstm":
            config.model.temporal.bidirectional = True
            config.model.temporal.hidden_dim = 1024
        else:
            config.model.temporal.bidirectional = False
            config.model.temporal.hidden_dim = 2048

    logger.info("Initializing complete DeepfakeDetector pipeline...")
    model = DeepfakeDetector(config.model)

    # Parameter accounting by stage
    bb_total = sum(p.numel() for p in model.backbone.parameters())
    bb_train = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    bb_froz = bb_total - bb_train

    attn_total = sum(p.numel() for p in model.attention.parameters())
    attn_train = sum(p.numel() for p in model.attention.parameters() if p.requires_grad)

    temp_total = sum(p.numel() for p in model.temporal.parameters())
    temp_train = sum(p.numel() for p in model.temporal.parameters() if p.requires_grad)

    clf_total = sum(p.numel() for p in model.classifier.parameters())
    clf_train = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    froz_params = total_params - train_params

    logger.info("=== DeepfakeDetector Parameter Breakdown ===")
    logger.info(f"  Backbone ({config.model.backbone.name}, freeze={config.model.backbone.freeze_strategy}):")
    logger.info(f"    Total: {bb_total:,} | Trainable: {bb_train:,} | Frozen: {bb_froz:,}")
    logger.info(f"  Attention ({config.model.attention.type}):")
    logger.info(f"    Total: {attn_total:,} | Trainable: {attn_train:,}")
    logger.info(f"  Temporal ({config.model.temporal.type}):")
    logger.info(f"    Total: {temp_total:,} | Trainable: {temp_train:,}")
    logger.info(f"  Classifier (MLP 2048 -> {config.model.classifier.hidden_dim} -> 1):")
    logger.info(f"    Total: {clf_total:,} | Trainable: {clf_train:,}")
    logger.info(
        f"  Total Model: {total_params:,} | Trainable: {train_params:,} ({train_params/total_params*100:.1f}%) | Frozen: {froz_params:,}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Target device: {gpu_name} ({total_vram_gb:.2f} GB total VRAM)")
    else:
        logger.info("Target device: CPU")

    logger.info(f"Running test forward pass (batch_size={args.batch_size}, frames={args.num_frames})...")
    model = model.to(device)

    dummy_input = torch.randn(args.batch_size, args.num_frames, 3, 112, 112, device=device)

    # Diagnostic forward
    model.eval()
    with torch.no_grad():
        diag_out = model(dummy_input, return_intermediates=True)

    logger.info("=== Forward Output Shapes ===")
    logger.info(f"  Raw Logits:           {tuple(diag_out.logits.shape)}")
    logger.info(f"  Diagnostic Probs:     {tuple(diag_out.probabilities.shape)}")
    logger.info(f"  Frame Features:       {tuple(diag_out.frame_features.shape)}")
    logger.info(f"  Temporal Features:    {tuple(diag_out.temporal_features.shape)}")
    logger.info(f"  Attention Map Keys:   {list(diag_out.attention_maps.keys())}")

    # VRAM profiling on CUDA
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # FP32 Forward
        model.train()
        x_fp32 = torch.randn(
            args.batch_size, args.num_frames, 3, 112, 112, device=device, requires_grad=True
        )
        out_fp32 = model(x_fp32)
        fp32_fwd = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # FP32 Forward + Backward
        loss_fp32 = out_fp32.sum()
        loss_fp32.backward()
        fp32_bwd = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # FP16 AMP Forward & Backward
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        x_fp16 = torch.randn(
            args.batch_size, args.num_frames, 3, 112, 112, device=device, requires_grad=True
        )
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out_fp16 = model(x_fp16)
            loss_fp16 = out_fp16.sum()
        fp16_fwd = torch.cuda.max_memory_allocated() / (1024 ** 2)

        loss_fp16.backward()
        fp16_bwd = torch.cuda.max_memory_allocated() / (1024 ** 2)

        logger.info("=== Realistic Training VRAM Profile (B=1, T=16) ===")
        logger.info(f"  FP32 Forward Peak:            {fp32_fwd:.2f} MB")
        logger.info(f"  FP32 Forward + Backward Peak: {fp32_bwd:.2f} MB")
        logger.info(f"  FP16 AMP Forward Peak:        {fp16_fwd:.2f} MB")
        logger.info(f"  FP16 AMP Fwd + Bwd Peak:      {fp16_bwd:.2f} MB")


def cmd_train(args: argparse.Namespace) -> None:
    """Train DeepfakeDetector model using authoritative manifest and preprocessed tensors."""
    logger = setup_logger("cli_train")
    config_path = args.config if args.config else "configs/default.yaml"
    config = load_config(config_path)

    # CLI hyperparameter overrides
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.optimizer.learning_rate = args.lr
    if args.processed_dir:
        config.storage.processed_dir = args.processed_dir
    if args.cpu:
        config.device = "cpu"

    config.validate()
    seed_everything(config.seed)

    logger.info("=== DeepfakeDetector Model Training ===")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Device: {config.get_effective_device()}")
    logger.info(f"Manifest: {args.manifest}")
    logger.info(f"Processed Directory: {config.storage.processed_dir}")

    # Build datasets for train and val splits
    train_dataset = ProcessedVideoDataset(
        manifest=args.manifest,
        split="train",
        processed_dir=config.storage.processed_dir,
        expected_num_frames=config.data.sampling.num_frames,
    )
    val_dataset = ProcessedVideoDataset(
        manifest=args.manifest,
        split="val",
        processed_dir=config.storage.processed_dir,
        expected_num_frames=config.data.sampling.num_frames,
    )

    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        seed=config.seed,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        seed=config.seed,
    )

    # Instantiate model
    model = DeepfakeDetector(config.model)

    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=config.training.checkpoint_dir,
        monitor_metric=config.training.monitor_metric,
        mode="min" if config.training.monitor_metric == "val_loss" else "max",
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_manager=checkpoint_mgr,
    )

    start_epoch = 1
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        cp = checkpoint_mgr.load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            restore_rng=True,
        )
        start_epoch = cp.get("epoch", 0) + 1
        trainer.early_stopping_counter = cp.get("early_stopping_counter", 0)
        logger.info(f"Resumed at epoch {start_epoch}, previous best metric: {checkpoint_mgr.best_metric:.4f}")

    results = trainer.fit(start_epoch=start_epoch)
    logger.info(f"Training completed. Best epoch: {results['best_epoch']} | Best metric: {results['best_metric']:.4f}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate trained checkpoint on a specified split (default: test)."""
    import torch.nn as nn

    logger = setup_logger("cli_evaluate")
    config_path = args.config if args.config else "configs/default.yaml"
    config = load_config(config_path)

    if args.processed_dir:
        config.storage.processed_dir = args.processed_dir
    if args.cpu:
        config.device = "cpu"

    effective_dev = torch.device(config.get_effective_device())

    logger.info("=== DeepfakeDetector Checkpoint Evaluation ===")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Manifest: {args.manifest}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Device: {effective_dev}")

    dataset = ProcessedVideoDataset(
        manifest=args.manifest,
        split=args.split,
        processed_dir=config.storage.processed_dir,
        expected_num_frames=config.data.sampling.num_frames,
    )
    loader = create_dataloader(
        dataset,
        batch_size=args.batch_size or config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        seed=config.seed,
    )

    model = DeepfakeDetector(config.model)
    ckpt_mgr = CheckpointManager()
    ckpt_mgr.load_checkpoint(args.checkpoint, model=model, restore_rng=False)
    model.to(effective_dev)
    model.eval()

    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.inference_mode():
        for batch in loader:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(effective_dev)
            targets = targets.to(effective_dev)

            logits = model(inputs)
            loss = criterion(logits, targets)

            total_loss += loss.item() * inputs.size(0)
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

    eval_loss = total_loss / len(dataset)
    cat_targets = torch.cat(all_targets, dim=0)
    cat_logits = torch.cat(all_logits, dim=0)

    metrics = compute_metrics(
        cat_targets,
        cat_logits,
        loss=eval_loss,
        threshold=config.evaluation.threshold,
    )
    logger.info("=== Evaluation Results ===")
    logger.info(metrics.summary_string())


def cmd_manifest_celebdf(args: argparse.Namespace) -> None:
    """Build Celeb-DF v2 manifest from dataset directory."""
    logger = setup_logger("manifest_celebdf")
    logger.info(f"Scanning Celeb-DF v2 dataset at: {args.root}")
    df = build_celebdf_manifest(
        dataset_root=args.root,
        test_list_path=args.test_list,
        split_ratios=(args.train_ratio, args.val_ratio),
        seed=args.seed,
    )
    save_manifest(df, args.output)
    logger.info(f"Saved Celeb-DF manifest with {len(df)} samples to: {args.output}")


def cmd_manifest_dfdc(args: argparse.Namespace) -> None:
    """Build DFDC manifest from dataset directory and metadata.json files."""
    logger = setup_logger("manifest_dfdc")
    logger.info(f"Scanning DFDC dataset at: {args.root}")
    df = build_dfdc_manifest_from_metadata(
        dataset_root=args.root,
        metadata_filename=args.metadata_filename,
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
    )
    save_manifest(df, args.output)
    logger.info(f"Saved DFDC manifest with {len(df)} samples to: {args.output}")


def cmd_manifest_ffpp(args: argparse.Namespace) -> None:
    """Build FaceForensics++ C23 manifest from dataset directory."""
    logger = setup_logger("manifest_ffpp")
    logger.info(f"Scanning FaceForensics++ dataset at: {args.root}")
    df = build_ffpp_manifest(
        dataset_root=args.root,
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
        include_all_methods=args.include_all,
    )
    save_manifest(df, args.output)
    logger.info(f"Saved FaceForensics++ manifest with {len(df)} samples to: {args.output}")


def cmd_run_experiment(args: argparse.Namespace) -> None:
    """Run an automated experiment from YAML configuration."""
    logger = setup_logger("cli_experiment")
    logger.info(f"Running experiment with config: {args.config}")
    runner = ExperimentRunner(
        config=args.config,
        manifest_path=args.manifest,
        output_root=args.output_root or "runs",
        experiment_name=args.name,
        device="cpu" if args.cpu else None,
    )
    summary = runner.run()
    logger.info(f"Experiment completed. Summary saved to: {summary['run_dir']}")


def cmd_cross_eval(args: argparse.Namespace) -> None:
    """Run cross-dataset evaluation of trained checkpoint across manifests."""
    logger = setup_logger("cli_cross_eval")
    logger.info(f"Evaluating checkpoint: {args.checkpoint}")

    benchmarks = {}
    for item in args.manifests:
        if "=" not in item:
            raise ValueError(f"Manifest argument must be in 'name=path' format, got: {item}")
        name, path = item.split("=", 1)
        benchmarks[name.strip()] = path.strip()

    evaluator = CrossDatasetEvaluator(
        checkpoint_path=args.checkpoint,
        config=args.config,
        device="cpu" if args.cpu else None,
        batch_size=args.batch_size or 4,
        threshold=args.threshold or 0.5,
    )
    if getattr(args, "processed_dir", None):
        evaluator.config.storage.processed_dir = args.processed_dir
    evaluator.evaluate_suite(
        benchmarks=benchmarks,
        split=args.split or "test",
        output_json=args.output,
    )
    logger.info(f"Cross-dataset evaluation complete. Results: {args.output}")


def cmd_generate_report(args: argparse.Namespace) -> None:
    """Generate comparative benchmark markdown report."""
    import json
    logger = setup_logger("cli_report")
    reporter = BenchmarkReporter(title=args.title or "Deepfake Detection Benchmark Report")

    if args.summary:
        for sum_p in args.summary:
            with open(sum_p, "r", encoding="utf-8") as f:
                data = json.load(f)
            reporter.add_indomain_experiment(data)

    if args.cross_eval:
        with open(args.cross_eval, "r", encoding="utf-8") as f:
            data = json.load(f)
        reporter.set_cross_dataset_results(data)

    report_path = reporter.save_reports(args.output_dir or "reports", filename_stem=args.stem or "benchmark_report")
    logger.info(f"Saved benchmark report to: {report_path}")


def cmd_predict(args: argparse.Namespace) -> None:
    """Run standalone inference on single video or batch directory."""
    import json
    from src.inference.predictor import VideoPredictor
    logger = setup_logger("cli_predict")

    ckpt_p = args.checkpoint or "checkpoints/ffpp_c23_baseline_soft_lstm/best.pth"
    device = "cpu" if args.cpu else (args.device or None)
    predictor = VideoPredictor(
        checkpoint_path=ckpt_p,
        config=args.config,
        device=device,
        threshold=args.threshold if args.threshold is not None else 0.5,
    )

    if args.video:
        res = predictor.predict_video(args.video, threshold=args.threshold)
        logger.info("=== Inference Prediction Result ===")
        logger.info(f"Video:       {res.video_path}")
        logger.info(f"Verdict:     {res.verdict}")
        logger.info(f"Probability: {res.probability:.4f} (Threshold={res.threshold})")
        logger.info(f"Raw Logit:   {res.raw_logit:.4f}")
        logger.info(f"Confidence:  {res.confidence:.4f}")
        logger.info(f"Frames:      {res.frames_analyzed} (Faces detected: {res.faces_detected})")
        logger.info(f"Latencies:   {res.latency_ms}")
        if res.error:
            logger.error(f"Error: {res.error}")

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(res.to_dict(), f, indent=2)
            logger.info(f"Saved prediction JSON to: {out_p}")

    elif args.video_dir:
        vid_dir = Path(args.video_dir)
        exts = (".mp4", ".avi", ".mov", ".mkv")
        video_files = [f for f in vid_dir.rglob("*") if f.suffix.lower() in exts]
        logger.info(f"Found {len(video_files)} video files in {vid_dir}")
        batch_results = predictor.predict_batch(video_files, threshold=args.threshold)
        summary_list = [r.to_dict() for r in batch_results]
        logger.info(f"Completed batch prediction on {len(batch_results)} videos.")
        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(summary_list, f, indent=2)
            logger.info(f"Saved batch prediction JSON to: {out_p}")
    else:
        logger.error("Either --video or --video-dir must be specified for prediction.")
        sys.exit(1)


def cmd_export(args: argparse.Namespace) -> None:
    """Export checkpoint to TorchScript or ONNX."""
    import json
    from src.inference.export import export_checkpoint
    logger = setup_logger("cli_export")

    ckpt_p = args.checkpoint or "checkpoints/ffpp_c23_baseline_soft_lstm/best.pth"
    out_dir = args.output_dir or "checkpoints/export"
    device = "cpu" if args.cpu else (args.device or "cpu")
    fmt = args.format or "all"

    logger.info(f"Exporting checkpoint '{ckpt_p}' to '{out_dir}' (format={fmt}, device={device})...")
    res = export_checkpoint(
        checkpoint_path=ckpt_p,
        output_dir=out_dir,
        export_format=fmt,
        device=device,
    )
    logger.info("=== Export Summary ===")
    for k, v in res.get("exports", {}).items():
        parity_str = "PASSED" if v.get("parity_verified") else "FAILED"
        logger.info(f"- {k.upper()}: Path={v.get('exported_path')} | Parity={parity_str} (max_diff={v.get('max_abs_diff')})")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


def cmd_benchmark_latency(args: argparse.Namespace) -> None:
    """Benchmark inference latency on hardware."""
    import json
    from src.inference.predictor import VideoPredictor
    from src.inference.profiler import LatencyProfiler
    logger = setup_logger("cli_benchmark")

    ckpt_p = args.checkpoint or "checkpoints/ffpp_c23_baseline_soft_lstm/best.pth"
    device = "cpu" if args.cpu else (args.device or None)
    predictor = VideoPredictor(ckpt_p, config=args.config, device=device)

    profiler = LatencyProfiler(device=device)
    num_runs = args.num_runs or 20
    warmup_runs = args.warmup_runs or 5
    logger.info(f"Benchmarking inference latency on {profiler.device} ({num_runs} runs)...")
    res = profiler.run_full_profile(
        predictor=predictor,
        video_path=args.video,
        num_runs=num_runs,
        warmup_runs=warmup_runs,
    )

    logger.info("=== Latency Profile Summary ===")
    logger.info(f"Device:            {res.device}")
    logger.info(f"Model Forward:     Mean={res.model_only['mean_ms']}ms | Median={res.model_only['median_ms']}ms | P95={res.model_only['p95_ms']}ms | FPS={res.model_only['fps']}")
    if res.end_to_end:
        logger.info(f"End-to-End Video:  Prep={res.end_to_end['mean_preprocessing_ms']}ms | Model={res.end_to_end['mean_model_forward_ms']}ms | Total={res.end_to_end['mean_total_ms']}ms")
    if res.vram_usage_mb:
        logger.info(f"VRAM Usage:        Allocated={res.vram_usage_mb['allocated_mb']}MB | Reserved={res.vram_usage_mb['reserved_mb']}MB")

    out_p = Path(args.output or "reports/latency_profile_rtx3050.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(res.to_dict(), f, indent=2)
    logger.info(f"Saved latency profile JSON to: {out_p}")


def cmd_serve(args: argparse.Namespace) -> None:
    """Launch REST API service and web dashboard."""
    import os
    import uvicorn
    logger = setup_logger("cli_serve")
    ckpt_p = args.checkpoint or "checkpoints/ffpp_c23_baseline_soft_lstm/best.pth"
    os.environ["DEEPFAKE_CHECKPOINT_PATH"] = str(Path(ckpt_p).resolve())
    if args.device:
        os.environ["DEEPFAKE_DEVICE"] = str(args.device)

    host = args.host or "127.0.0.1"
    port = int(args.port or 8000)
    logger.info(f"Starting Deepfake Detection API service on http://{host}:{port} (checkpoint: {ckpt_p})...")
    uvicorn.run("src.api.app:app", host=host, port=port, reload=bool(args.reload))


def cmd_forensic_audit(args: argparse.Namespace) -> None:
    """Execute end-to-end spatiotemporal forensic audit on video."""
    import json
    from src.forensics.spatial_temporal_pipeline import SpatialTemporalForensicEngine
    from src.forensics.context import ForensicAnalysisConfig
    logger = setup_logger("cli_forensic_audit")

    if not args.video:
        logger.error("Please provide --video path.")
        sys.exit(1)

    video_p = Path(args.video)
    if not video_p.is_file():
        logger.error(f"Video file not found: {video_p}")
        sys.exit(1)

    cfg = ForensicAnalysisConfig(
        max_windows=args.windows or 4,
        window_duration_sec=args.window_duration or 2.0,
        frames_per_window=args.frames_per_window or 16,
    )
    device = "cpu" if args.cpu else (args.device or None)
    engine = SpatialTemporalForensicEngine(
        checkpoint_path=args.checkpoint,
        config=cfg,
        device=device,
    )

    logger.info(f"Running forensic audit on {video_p.name}...")
    report = engine.analyze_video(video_p)

    logger.info("=== Forensic Audit Summary ===")
    logger.info(f"Video:              {report.video_path}")
    logger.info(f"Verdict:            {report.verdict} (Confidence: {report.verdict_confidence})")
    logger.info(f"Fake Probability:   {report.video_fake_probability:.4f} (95% CI: [{report.confidence_interval[0]:.4f}, {report.confidence_interval[1]:.4f}])")
    logger.info(f"Windows Analyzed:   {len(report.windows_analyzed)}")
    logger.info(f"Kinematic Energy:   {report.kinematic_residual_energy:.4f}")
    if report.warnings:
        for w in report.warnings:
            logger.warning(f"Forensic Warning: {w}")

    out_p = Path(args.output or f"reports/forensic_{video_p.stem}.json")
    engine.export_report_json(report, out_p)
    logger.info(f"Saved forensic report JSON to: {out_p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deepfake Video Detection CLI")
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # verify-env
    subparsers.add_parser("verify-env", help="Check hardware, CUDA, PyTorch, and MTCNN")

    # preprocess
    p_prep = subparsers.add_parser("preprocess", help="Run video preprocessing pipeline")
    p_prep.add_argument("--video", type=str, help="Path to single video file")
    p_prep.add_argument("--manifest", type=str, help="Path to CSV/JSONL manifest")
    p_prep.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration YAML")
    p_prep.add_argument("--output-dir", type=str, help="Output directory for processed tensors")
    p_prep.add_argument("--video-id", type=str, help="Custom video ID for single video")
    p_prep.add_argument("--group-id", type=str, help="Custom group ID for single video")
    p_prep.add_argument("--split", type=str, default="train", help="Target split name")

    # manifest
    p_man = subparsers.add_parser("manifest", help="Dataset manifest commands")
    p_man_sub = p_man.add_subparsers(dest="manifest_cmd", help="Manifest subcommands")
    p_split = p_man_sub.add_parser("split", help="Split manifest by indivisible group stratification")
    p_split.add_argument("--input", type=str, required=True, help="Input manifest file")
    p_split.add_argument("--output", type=str, required=True, help="Output split manifest file")
    p_split.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (default: 0.70)")
    p_split.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio (default: 0.15)")
    p_split.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (default: 0.15)")
    p_split.add_argument("--seed", type=int, default=42, help="Random seed")

    # manifest-celebdf
    p_mcdf = subparsers.add_parser("manifest-celebdf", help="Generate Celeb-DF v2 manifest")
    p_mcdf.add_argument("--root", type=str, required=True, help="Path to Celeb-DF root folder")
    p_mcdf.add_argument("--test-list", type=str, help="Path to List_of_testing_videos.txt")
    p_mcdf.add_argument("--output", type=str, required=True, help="Output manifest path")
    p_mcdf.add_argument("--train-ratio", type=float, default=0.85, help="Train ratio for non-test videos")
    p_mcdf.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio for non-test videos")
    p_mcdf.add_argument("--seed", type=int, default=42, help="Random seed")

    # manifest-dfdc
    p_mdfdc = subparsers.add_parser("manifest-dfdc", help="Generate DFDC manifest")
    p_mdfdc.add_argument("--root", type=str, required=True, help="Path to DFDC root/partition folder")
    p_mdfdc.add_argument("--metadata-filename", type=str, default="metadata.json", help="Metadata filename")
    p_mdfdc.add_argument("--output", type=str, required=True, help="Output manifest path")
    p_mdfdc.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio")
    p_mdfdc.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio")
    p_mdfdc.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio")
    p_mdfdc.add_argument("--seed", type=int, default=42, help="Random seed")

    # manifest-ffpp
    p_mffpp = subparsers.add_parser("manifest-ffpp", help="Generate FaceForensics++ C23 manifest")
    p_mffpp.add_argument("--root", type=str, required=True, help="Path to FaceForensics++ root folder")
    p_mffpp.add_argument("--output", type=str, required=True, help="Output manifest path")
    p_mffpp.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (default: 0.70)")
    p_mffpp.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio (default: 0.15)")
    p_mffpp.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (default: 0.15)")
    p_mffpp.add_argument("--include-all", action="store_true", help="Include extended methods (FaceShifter, DFD)")
    p_mffpp.add_argument("--seed", type=int, default=42, help="Random seed")

    # inspect-backbone
    p_insp = subparsers.add_parser("inspect-backbone", help="Inspect ResNeXt-50 parameters and memory footprint")
    p_insp.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    p_insp.add_argument("--freeze-strategy", type=str, choices=["none", "stem", "layer1", "layer2", "layer3", "all"], help="Override freeze strategy")
    p_insp.add_argument("--no-pretrained", action="store_true", help="Instantiate with random weights (weights=None)")
    p_insp.add_argument("--batch-size", type=int, default=1, help="Batch size for test forward pass")
    p_insp.add_argument("--num-frames", type=int, default=16, help="Sequence length for test forward pass")
    p_insp.add_argument("--cpu", action="store_true", help="Force CPU inference")

    # inspect-attention
    p_attn = subparsers.add_parser("inspect-attention", help="Inspect Attention parameters and memory footprint")
    p_attn.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    p_attn.add_argument("--attention-type", type=str, choices=["soft", "cbam", "channel", "spatial", "none"], help="Override attention type")
    p_attn.add_argument("--batch-size", type=int, default=1, help="Batch size for test forward pass")
    p_attn.add_argument("--num-frames", type=int, default=16, help="Sequence length for test forward pass")
    p_attn.add_argument("--cpu", action="store_true", help="Force CPU inference")

    # inspect-model
    p_model = subparsers.add_parser("inspect-model", help="Inspect complete DeepfakeDetector architecture and memory footprint")
    p_model.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    p_model.add_argument("--no-pretrained", action="store_true", help="Use random backbone weights (weights=None)")
    p_model.add_argument("--freeze-strategy", type=str, choices=["none", "stem", "layer1", "layer2", "layer3", "all"], help="Override backbone freeze strategy")
    p_model.add_argument("--attention-type", type=str, choices=["soft", "cbam", "channel", "spatial", "none"], help="Override attention type")
    p_model.add_argument("--temporal-type", type=str, choices=["lstm", "bilstm"], help="Override temporal type")
    p_model.add_argument("--batch-size", type=int, default=1, help="Batch size for test forward pass")
    p_model.add_argument("--num-frames", type=int, default=16, help="Sequence length for test forward pass")
    p_model.add_argument("--cpu", action="store_true", help="Force CPU inference")

    # train
    p_train = subparsers.add_parser("train", help="Train DeepfakeDetector on preprocessed data")
    p_train.add_argument("--manifest", type=str, required=True, help="Path to split manifest CSV/JSONL")
    p_train.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration YAML")
    p_train.add_argument("--processed-dir", type=str, help="Directory containing preprocessed tensors")
    p_train.add_argument("--resume", type=str, help="Path to checkpoint .pth to resume training from")
    p_train.add_argument("--epochs", type=int, help="Override number of training epochs")
    p_train.add_argument("--batch-size", type=int, help="Override batch size")
    p_train.add_argument("--lr", type=float, help="Override learning rate")
    p_train.add_argument("--cpu", action="store_true", help="Force CPU training")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate trained checkpoint on a split")
    p_eval.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint .pth")
    p_eval.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV/JSONL")
    p_eval.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration YAML")
    p_eval.add_argument("--processed-dir", type=str, help="Directory containing preprocessed tensors")
    p_eval.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Split to evaluate")
    p_eval.add_argument("--batch-size", type=int, help="Override batch size")
    p_eval.add_argument("--cpu", action="store_true", help="Force CPU evaluation")

    # run-experiment
    p_exp = subparsers.add_parser("run-experiment", help="Run automated experiment from YAML config")
    p_exp.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    p_exp.add_argument("--manifest", type=str, required=True, help="Path to dataset manifest CSV/JSONL")
    p_exp.add_argument("--name", type=str, help="Custom experiment name")
    p_exp.add_argument("--output-root", type=str, default="runs", help="Output directory root for runs")
    p_exp.add_argument("--cpu", action="store_true", help="Force CPU execution")

    # cross-eval
    p_ceval = subparsers.add_parser("cross-eval", help="Cross-dataset evaluation of trained checkpoint")
    p_ceval.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint .pth")
    p_ceval.add_argument("--manifests", nargs="+", required=True, help="Target manifests in name=path format")
    p_ceval.add_argument("--config", type=str, help="Optional configuration YAML override")
    p_ceval.add_argument("--processed-dir", type=str, help="Directory containing preprocessed tensors")
    p_ceval.add_argument("--output", type=str, required=True, help="Output JSON path")
    p_ceval.add_argument("--split", type=str, default="test", help="Split to evaluate on target manifests")
    p_ceval.add_argument("--batch-size", type=int, help="Batch size")
    p_ceval.add_argument("--threshold", type=float, help="Decision threshold")
    p_ceval.add_argument("--cpu", action="store_true", help="Force CPU execution")

    # generate-report
    p_rep = subparsers.add_parser("generate-report", help="Generate benchmark report markdown and JSON")
    p_rep.add_argument("--summary", nargs="+", help="Paths to experiment_summary.json files")
    p_rep.add_argument("--cross-eval", type=str, help="Path to cross_dataset_results.json")
    p_rep.add_argument("--output-dir", type=str, default="reports", help="Output directory for reports")
    p_rep.add_argument("--stem", type=str, default="benchmark_report", help="Report filename stem")
    p_rep.add_argument("--title", type=str, help="Custom report title")

    # predict
    p_pred = subparsers.add_parser("predict", help="Run standalone inference on single video or directory")
    p_pred.add_argument("--video", type=str, help="Path to single video file")
    p_pred.add_argument("--video-dir", type=str, help="Path to directory of video files")
    p_pred.add_argument("--checkpoint", type=str, default="checkpoints/ffpp_c23_baseline_soft_lstm/best.pth", help="Path to checkpoint .pth")
    p_pred.add_argument("--config", type=str, help="Path to config YAML")
    p_pred.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    p_pred.add_argument("--output", type=str, help="Path to save JSON prediction output")
    p_pred.add_argument("--cpu", action="store_true", help="Force CPU inference")
    p_pred.add_argument("--device", type=str, help="Target device")

    # export
    p_exp = subparsers.add_parser("export", help="Export trained checkpoint to TorchScript / ONNX")
    p_exp.add_argument("--checkpoint", type=str, default="checkpoints/ffpp_c23_baseline_soft_lstm/best.pth", help="Path to checkpoint .pth")
    p_exp.add_argument("--output-dir", type=str, default="checkpoints/export", help="Output directory for exported artifacts")
    p_exp.add_argument("--format", type=str, choices=["torchscript", "onnx", "all"], default="all", help="Export format")
    p_exp.add_argument("--output-json", type=str, help="Path to save export audit JSON")
    p_exp.add_argument("--cpu", action="store_true", help="Force CPU export")
    p_exp.add_argument("--device", type=str, default="cpu", help="Device for export")

    # benchmark-latency
    p_bm = subparsers.add_parser("benchmark-latency", help="Benchmark inference latency on hardware")
    p_bm.add_argument("--checkpoint", type=str, default="checkpoints/ffpp_c23_baseline_soft_lstm/best.pth", help="Path to checkpoint .pth")
    p_bm.add_argument("--config", type=str, help="Path to config YAML")
    p_bm.add_argument("--video", type=str, help="Path to physical video for end-to-end benchmark")
    p_bm.add_argument("--num-runs", type=int, default=20, help="Number of benchmark iterations")
    p_bm.add_argument("--warmup-runs", type=int, default=5, help="Number of warmup iterations")
    p_bm.add_argument("--output", type=str, default="reports/latency_profile_rtx3050.json", help="Output path for JSON report")
    p_bm.add_argument("--cpu", action="store_true", help="Force CPU benchmarking")
    p_bm.add_argument("--device", type=str, help="Target device")

    # serve
    p_srv = subparsers.add_parser("serve", help="Launch FastAPI detection service and web dashboard")
    p_srv.add_argument("--checkpoint", type=str, default="checkpoints/ffpp_c23_baseline_soft_lstm/best.pth", help="Path to checkpoint .pth")
    p_srv.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    p_srv.add_argument("--port", type=int, default=8000, help="Port to bind")
    p_srv.add_argument("--device", type=str, help="Target device")
    p_srv.add_argument("--reload", action="store_true", help="Enable reload mode")

    # forensic-audit
    p_fa = subparsers.add_parser("forensic-audit", help="Run spatiotemporal forensic audit on video")
    p_fa.add_argument("--video", type=str, required=True, help="Path to video file")
    p_fa.add_argument("--checkpoint", type=str, help="Path to detector checkpoint .pth")
    p_fa.add_argument("--windows", type=int, default=4, help="Max temporal windows to evaluate")
    p_fa.add_argument("--window-duration", type=float, default=2.0, help="Duration of each temporal window (sec)")
    p_fa.add_argument("--frames-per-window", type=int, default=16, help="Target frames sampled per window")
    p_fa.add_argument("--output", type=str, help="Output path for JSON forensic report")
    p_fa.add_argument("--cpu", action="store_true", help="Force CPU execution")
    p_fa.add_argument("--device", type=str, help="Target device")

    args = parser.parse_args()

    if args.subcommand == "verify-env":
        verify_env()
    elif args.subcommand == "preprocess":
        cmd_preprocess(args)
    elif args.subcommand == "manifest":
        if args.manifest_cmd == "split":
            cmd_manifest_split(args)
        else:
            p_man.print_help()
    elif args.subcommand == "manifest-celebdf":
        cmd_manifest_celebdf(args)
    elif args.subcommand == "manifest-dfdc":
        cmd_manifest_dfdc(args)
    elif args.subcommand == "manifest-ffpp":
        cmd_manifest_ffpp(args)
    elif args.subcommand == "inspect-backbone":
        cmd_inspect_backbone(args)
    elif args.subcommand == "inspect-attention":
        cmd_inspect_attention(args)
    elif args.subcommand == "inspect-model":
        cmd_inspect_model(args)
    elif args.subcommand == "train":
        cmd_train(args)
    elif args.subcommand == "evaluate":
        cmd_evaluate(args)
    elif args.subcommand == "run-experiment":
        cmd_run_experiment(args)
    elif args.subcommand == "cross-eval":
        cmd_cross_eval(args)
    elif args.subcommand == "generate-report":
        cmd_generate_report(args)
    elif args.subcommand == "predict":
        cmd_predict(args)
    elif args.subcommand == "export":
        cmd_export(args)
    elif args.subcommand == "benchmark-latency":
        cmd_benchmark_latency(args)
    elif args.subcommand == "serve":
        cmd_serve(args)
    elif args.subcommand == "forensic-audit":
        cmd_forensic_audit(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()



