"""
Generate clean, publication-grade architectural diagrams for the presentation.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_system_pipeline_diagram(output_path: Path):
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=200)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Background
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    stages = [
        ("1. Input Video\nContainer", "(MP4 / AVI / MOV)\n1080p / 30fps", "#1E293B", 0.5),
        ("2. Temporal\nSampling", "Deterministic\nUniform (N=16)", "#0284C7", 2.3),
        ("3. MTCNN Face\nDetection", "0.20 Margin\nPrimary ROI Area", "#2563EB", 4.1),
        ("4. Tensor\nPreprocessing", "112x112x3 Resizing\nImageNet Norm", "#4F46E5", 5.9),
        ("5. Selected Model\nEngine", "v1 to v5 / Ensemble\nSpatial-Temporal", "#7C3AED", 7.7),
        ("6. Calibration &\nOOD Decision", "T=2.0502 Scaling\nMahalanobis Gating", "#059669", 9.5),
        ("7. Forensic Verdict\n& Evidence", "Verdict + Prob\nOOD + Heatmap", "#0F172A", 11.3)
    ]

    for title, desc, color, x_center in stages:
        rect = patches.FancyBboxPatch(
            (x_center - 0.75, 1.2), 1.5, 2.6,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            edgecolor=color, facecolor='white', linewidth=2
        )
        ax.add_patch(rect)

        # Header bar
        header_bar = patches.FancyBboxPatch(
            (x_center - 0.75, 2.9), 1.5, 0.9,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            edgecolor=color, facecolor=color, linewidth=1
        )
        ax.add_patch(header_bar)

        ax.text(x_center, 3.35, title, ha='center', va='center', color='white',
                fontsize=8.5, fontweight='bold', family='sans-serif')
        ax.text(x_center, 2.0, desc, ha='center', va='center', color='#334155',
                fontsize=7.5, family='sans-serif')

    # Draw arrows between stages
    for i in range(len(stages) - 1):
        x_start = stages[i][3] + 0.8
        x_end = stages[i+1][3] - 0.8
        ax.annotate('', xy=(x_end, 2.5), xytext=(x_start, 2.5),
                    arrowprops=dict(arrowstyle="->", color="#64748B", lw=2, mutation_scale=15))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated pipeline diagram: {output_path}")

def generate_multibranch_diagram(output_path: Path):
    fig, ax = plt.subplots(figsize=(11, 5), dpi=200)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    # Input Block
    input_box = patches.FancyBboxPatch((0.4, 2.2), 1.6, 1.6, boxstyle="round,pad=0.1",
                                       edgecolor="#1E293B", facecolor="#1E293B")
    ax.add_patch(input_box)
    ax.text(1.2, 3.0, "Input Frames\n[16, 3, 112, 112]", ha='center', va='center',
            color='white', fontsize=9, fontweight='bold')

    # Stem Backbone
    stem_box = patches.FancyBboxPatch((2.6, 2.2), 1.6, 1.6, boxstyle="round,pad=0.1",
                                      edgecolor="#2563EB", facecolor="white", lw=2)
    ax.add_patch(stem_box)
    ax.text(3.4, 3.2, "EfficientNet-B0\nSpatial Stem", ha='center', va='center',
            color="#1E3A8A", fontsize=9, fontweight='bold')
    ax.text(3.4, 2.6, "Stages 1-4 Feature Maps", ha='center', va='center',
            color="#64748B", fontsize=7.5)

    # 3 Parallel Branches
    branches = [
        ("Branch 1: Mid-Spatial (Stage 5)", "Captures Facial Boundary Seams", 4.3, "#0284C7"),
        ("Branch 2: High-Freq Residual", "SRM / Laplacian Filter Banks", 2.5, "#D97706"),
        ("Branch 3: Texture Anomaly", "High-Pass Spatial Gradients", 0.7, "#059669")
    ]

    for bname, bdesc, y_pos, bcolor in branches:
        bbox = patches.FancyBboxPatch((5.0, y_pos), 2.4, 1.2, boxstyle="round,pad=0.1",
                                      edgecolor=bcolor, facecolor="white", lw=2)
        ax.add_patch(bbox)
        ax.text(6.2, y_pos + 0.75, bname, ha='center', va='center',
                color=bcolor, fontsize=8.5, fontweight='bold')
        ax.text(6.2, y_pos + 0.35, bdesc, ha='center', va='center',
                color="#475569", fontsize=7)

        # Arrows from stem to branches
        ax.annotate('', xy=(4.95, y_pos + 0.6), xytext=(4.25, 3.0),
                    arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=1.5))
        # Arrows from branches to Transformer
        ax.annotate('', xy=(8.15, 3.0), xytext=(7.45, y_pos + 0.6),
                    arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=1.5))

    # Transformer Encoder
    tr_box = patches.FancyBboxPatch((8.2, 2.0), 1.4, 2.0, boxstyle="round,pad=0.1",
                                    edgecolor="#7C3AED", facecolor="white", lw=2)
    ax.add_patch(tr_box)
    ax.text(8.9, 3.3, "Temporal\nTransformer", ha='center', va='center',
            color="#6D28D9", fontsize=9, fontweight='bold')
    ax.text(8.9, 2.5, "4 Attention Heads\nSeq Length 16", ha='center', va='center',
            color="#64748B", fontsize=7.5)

    # Fusion & Logit
    out_box = patches.FancyBboxPatch((10.1, 2.2), 0.7, 1.6, boxstyle="round,pad=0.1",
                                     edgecolor="#0F172A", facecolor="#0F172A")
    ax.add_patch(out_box)
    ax.text(10.45, 3.0, "Fusion\nHead\n-> Logit z", ha='center', va='center',
            color='white', fontsize=8, fontweight='bold')

    ax.annotate('', xy=(2.55, 3.0), xytext=(2.05, 3.0),
                arrowprops=dict(arrowstyle="->", color="#64748B", lw=2))
    ax.annotate('', xy=(10.05, 3.0), xytext=(9.65, 3.0),
                arrowprops=dict(arrowstyle="->", color="#64748B", lw=2))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated multibranch diagram: {output_path}")

if __name__ == "__main__":
    generate_system_pipeline_diagram(Path("reports/figures/system_pipeline_flowchart.png"))
    generate_multibranch_diagram(Path("reports/figures/multibranch_architecture.png"))
