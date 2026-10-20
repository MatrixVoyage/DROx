"""Diagnostic probe of individual head contributions in M13.3.

Evaluates each constituent output of FamilyInvariantDeepfakeDetector:
1. p_primary (Fused multi-domain representation)
2. p_freq (Frequency residual branch alone)
3. p_texture (Local texture ROI branch alone)
4. p_aux_fake (1 - P(Real) from multi-family auxiliary head)
5. p_composite (Ensemble combination)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from torch.utils.data import DataLoader

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ProcessedVideoDataset
from src.models.detector import DeepfakeDetector
from src.models.family_invariant_detector import FamilyInvariantDeepfakeDetector
from src.utils.config import load_config, _dict_to_config
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "mixed_robust_full.csv"
PROCESSED_DIR = PROJECT_ROOT / "processed" / "m12_full"
M12_CKPT = PROJECT_ROOT / "checkpoints" / "m12_full" / "best.pth"
M13_3_CKPT = PROJECT_ROOT / "checkpoints" / "m13_3_f2f_seed42" / "best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(MANIFEST_PATH)
test_df = df[df["split"] == "test"].copy()

clean_df = test_df[test_df["label"] == 0].copy()
f2f_df = test_df[test_df["manipulation_type"] == "Face2Face"].copy()

def filter_existing(subset_df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for _, row in subset_df.iterrows():
        p = PROCESSED_DIR / "test" / "tensors" / f"{row['video_id']}.pt"
        if p.exists():
            keep.append(row)
    return pd.DataFrame(keep)

clean_df = filter_existing(clean_df)
f2f_df = filter_existing(f2f_df)

eval_df = pd.concat([clean_df, f2f_df], ignore_index=True)
ds = ProcessedVideoDataset(manifest=eval_df, split="test", processed_dir=PROCESSED_DIR)
loader = DataLoader(ds, batch_size=8, shuffle=False)

# Load model
ckpt_m12 = torch.load(M12_CKPT, map_location=device, weights_only=False)
m12_cfg = _dict_to_config(ckpt_m12["config"]) if isinstance(ckpt_m12["config"], dict) else ckpt_m12["config"]

model = FamilyInvariantDeepfakeDetector(
    base_detector=DeepfakeDetector(config=m12_cfg.model).to(device),
    num_families=5,
    aux_hidden_dim=512,
    freq_dim=64,
    texture_dim=64,
    fusion_dim=512,
    enable_mid_head=True,
    mid_in_channels=112,
    mid_hidden_dim=128,
).to(device)

ckpt_13_3 = torch.load(M13_3_CKPT, map_location=device, weights_only=False)
model.load_state_dict(ckpt_13_3["model_state_dict"])
model.eval()

y_true = []
probs = {
    "fused": [],
    "primary": [],
    "freq": [],
    "texture": [],
    "aux_fake": [],
    "composite": [],
}

with torch.no_grad():
    for batch in loader:
        tensors, targets, vids = batch
        tensors = tensors.to(device, dtype=torch.float32)
        out = model(tensors, return_dict=True)

        y_true.extend(targets.numpy().flatten().tolist())
        probs["fused"].extend(out.p_fused.cpu().numpy().flatten().tolist())
        probs["primary"].extend(out.p_primary.cpu().numpy().flatten().tolist())
        probs["freq"].extend(out.p_freq.cpu().numpy().flatten().tolist())
        probs["texture"].extend(out.p_texture.cpu().numpy().flatten().tolist())
        probs["aux_fake"].extend(out.p_aux_fake.cpu().numpy().flatten().tolist())
        probs["composite"].extend(out.p_composite.cpu().numpy().flatten().tolist())

y_eval = np.array(y_true)
clean_mask = y_eval == 0
f2f_mask = y_eval == 1

print("=" * 80)
print(f"{'Branch / Output':<20} | {'Clean Spec':<10} | {'F2F Det':<14} | {'F2F ROC-AUC':<12} | {'F2F PR-AUC':<12}")
print("-" * 80)

for name, p_list in probs.items():
    p = np.array(p_list)
    spec = float(np.mean(p[clean_mask] < 0.5)) * 100
    det = int(np.sum(p[f2f_mask] >= 0.5))
    det_rate = float(np.mean(p[f2f_mask] >= 0.5)) * 100
    auc = float(roc_auc_score(y_eval, p))
    pr_auc = float(average_precision_score(y_eval, p))
    print(f"{name:<20} | {spec:6.2f}%    | {det_rate:5.2f}% ({det:02d}/58) | {auc:10.4f}   | {pr_auc:10.4f}")

print("=" * 80)
