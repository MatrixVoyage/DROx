"""Cross-dataset evaluation protocol for testing in-domain trained models on unseen benchmarks.

Enforces:
1. Strictly read-only evaluation (torch.inference_mode(), zero parameter/optimizer updates).
2. Evaluation of external benchmarks (e.g., Celeb-DF v2, DFDC) using the frozen in-domain best.pth.
3. Computation of generalization delta metrics (AUC drop, EER delta).
4. Structured JSON output of cross-dataset benchmark results.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from src.data.dataset import ProcessedVideoDataset, create_dataloader
from src.data.manifest import load_manifest
from src.evaluation.metrics import EvaluationMetrics, compute_metrics
from src.models.detector import DeepfakeDetector
from src.utils.config import AppConfig, load_config
from src.utils.logger import setup_logger


@dataclass
class DatasetBenchmarkResult:
    """Result for a single evaluation dataset."""
    dataset_name: str
    manifest_path: str
    num_samples: int
    metrics: EvaluationMetrics
    auc_drop_vs_indomain: Optional[float] = None
    eer_delta_vs_indomain: Optional[float] = None


class CrossDatasetEvaluator:
    """Evaluates trained DeepfakeDetector checkpoints across independent target benchmarks."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config: Optional[Union[AppConfig, str, Path]] = None,
        device: Optional[str] = None,
        batch_size: int = 4,
        threshold: float = 0.5,
    ) -> None:
        """Initialize CrossDatasetEvaluator.

        Args:
            checkpoint_path: Path to trained checkpoint .pth (typically best.pth).
            config: Optional AppConfig or YAML path. If None, restored from checkpoint.
            device: Target device ("cuda" or "cpu").
            batch_size: Evaluation batch size.
            threshold: Decision threshold for binary classification.
        """
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.batch_size = batch_size
        self.threshold = threshold
        self.logger = setup_logger("cross_dataset_eval")

        # Load checkpoint payload
        checkpoint_data = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        # Resolve configuration
        if config is not None:
            if isinstance(config, (str, Path)):
                self.config = load_config(config)
            else:
                self.config = config
        elif "config" in checkpoint_data and checkpoint_data["config"] is not None:
            from src.utils.config import _dict_to_config
            self.config = _dict_to_config(checkpoint_data["config"])
        else:
            self.config = load_config("configs/default.yaml")

        if device:
            self.config.device = device

        self.device = torch.device(self.config.get_effective_device())

        # In-domain metrics from checkpoint if available
        self.indomain_metrics: Optional[Dict[str, Any]] = checkpoint_data.get("current_metrics") or checkpoint_data.get("val_metrics")

        # Build model and load weights
        self.model = DeepfakeDetector(self.config.model)
        from src.training.checkpoint import CheckpointManager
        ckpt_mgr = CheckpointManager()
        ckpt_mgr.load_checkpoint(self.checkpoint_path, model=self.model, restore_rng=False)
        self.model.to(self.device)
        self.model.eval()

        self.logger.info(f"Loaded model checkpoint for evaluation from: {self.checkpoint_path}")
        self.logger.info(f"Target evaluation device: {self.device}")

    def evaluate_manifest(
        self,
        manifest_path: Union[str, Path],
        dataset_name: str,
        split: str = "test",
        processed_dir: Optional[Union[str, Path]] = None,
    ) -> DatasetBenchmarkResult:
        """Run read-only evaluation on a target manifest.

        Args:
            manifest_path: Path to manifest CSV/JSONL.
            dataset_name: Display name of target benchmark.
            split: Split to evaluate (default: "test").
            processed_dir: Path to directory containing preprocessed tensors.

        Returns:
            DatasetBenchmarkResult container.
        """
        manifest_p = Path(manifest_path)
        proc_dir = processed_dir or self.config.storage.processed_dir

        df = load_manifest(manifest_p)
        # Check if requested split is present
        target_split = split
        if "split" in df.columns:
            if (df["split"] == split).any():
                target_split = split
            elif (df["split"] == "").all() or df["split"].isna().all():
                # If no split assigned, evaluate full manifest
                target_split = ""

        dataset = ProcessedVideoDataset(
            manifest=manifest_p,
            split=target_split,
            processed_dir=proc_dir,
            expected_num_frames=self.config.data.sampling.num_frames,
        )

        if len(dataset) == 0:
            raise ValueError(f"No samples found for evaluation in {manifest_path} (split='{target_split}')")

        loader = create_dataloader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            seed=self.config.seed,
        )

        all_logits: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        total_loss = 0.0
        criterion = nn.BCEWithLogitsLoss()

        with torch.inference_mode():
            for batch in loader:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(inputs)
                loss = criterion(logits, targets)

                total_loss += loss.item() * inputs.size(0)
                all_logits.append(logits.detach().cpu())
                all_targets.append(targets.detach().cpu())

        cat_targets = torch.cat(all_targets, dim=0)
        cat_logits = torch.cat(all_logits, dim=0)

        eval_metrics = compute_metrics(
            cat_targets,
            cat_logits,
            loss=total_loss / len(dataset),
            threshold=self.threshold,
        )

        # Generalization delta vs in-domain
        auc_drop = None
        eer_delta = None
        if self.indomain_metrics:
            in_auc = self.indomain_metrics.get("roc_auc")
            in_eer = self.indomain_metrics.get("eer")
            if in_auc is not None and eval_metrics.roc_auc is not None:
                auc_drop = round(float(in_auc) - float(eval_metrics.roc_auc), 4)
            if in_eer is not None and eval_metrics.eer is not None:
                eer_delta = round(float(eval_metrics.eer) - float(in_eer), 4)

        self.logger.info(f"[{dataset_name}] Evaluated {len(dataset)} samples: {eval_metrics.summary_string()}")

        return DatasetBenchmarkResult(
            dataset_name=dataset_name,
            manifest_path=str(manifest_p.resolve()),
            num_samples=len(dataset),
            metrics=eval_metrics,
            auc_drop_vs_indomain=auc_drop,
            eer_delta_vs_indomain=eer_delta,
        )

    def evaluate_suite(
        self,
        benchmarks: Dict[str, Union[str, Path]],
        split: str = "test",
        output_json: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Evaluate checkpoint across multiple target benchmark manifests.

        Args:
            benchmarks: Dictionary mapping benchmark names to manifest paths.
            split: Split to evaluate on each manifest.
            output_json: Optional path to save structured results.

        Returns:
            Dictionary containing evaluation summary across all benchmarks.
        """
        results = {}
        for name, path in benchmarks.items():
            bench_result = self.evaluate_manifest(path, dataset_name=name, split=split)
            results[name] = {
                "manifest_path": bench_result.manifest_path,
                "num_samples": bench_result.num_samples,
                "metrics": asdict(bench_result.metrics),
                "auc_drop_vs_indomain": bench_result.auc_drop_vs_indomain,
                "eer_delta_vs_indomain": bench_result.eer_delta_vs_indomain,
            }

        payload = {
            "checkpoint_path": str(self.checkpoint_path.resolve()),
            "indomain_metrics": self.indomain_metrics,
            "benchmarks": results,
        }

        if output_json:
            out_p = Path(output_json)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self.logger.info(f"Saved cross-dataset evaluation report to: {out_p}")

        return payload
