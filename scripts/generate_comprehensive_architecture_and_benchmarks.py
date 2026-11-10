"""
Script to generate comprehensive, publication-grade architectural diagrams
and empirical model comparison benchmarks faithful to this repository.
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set global styles
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']


def generate_model_comparison_figure(output_path: Path):
    """
    Generates a 2-panel empirical benchmark comparison across all model engines:
    1. Multi-Benchmark AUROC Comparison (FF++ C23, Celeb-DF v2, Face2Face, JPEG Q50)
    2. Latency vs. Accuracy Pareto Frontier on NVIDIA RTX 3050 Laptop GPU
    """
    fig = plt.figure(figsize=(15, 6.5), dpi=300)
    fig.patch.set_facecolor('#0B0F19')

    # Models to compare
    models = [
        "v1-baseline\n(Lal et al. 2025)",
        "v2-evolved\n(EffNet+Trans)",
        "v3-robust\n(Full FF++ C23)",
        "v5-family-invariant\n(Production Champion)",
        "m18-dual-stream\n(3D Kinematics)",
        "m22-ensemble\n(Research Champion)"
    ]

    # Empirical AUROC numbers from reports (M7, M10, M12, M13.4, M18, M22)
    ffpp_auc = [0.8840, 0.9020, 0.9087, 0.9165, 0.8950, 0.9230]
    celeb_auc = [0.7120, 0.7450, 0.7780, 0.7824, 0.8166, 0.8213]
    f2f_auc = [0.6820, 0.7150, 0.7787, 0.7947, 0.7858, 0.8118]
    jpeg_auc = [0.5120, 0.5420, 0.6120, 0.6360, 0.8822, 0.8944]

    # Hardware stats on RTX 3050 (reports/latency_profile_rtx3050.json, reports/m22_promotion_evaluation_report.md)
    latencies = [27.4, 22.1, 24.5, 24.95, 48.2, 72.58]
    macro_aucs = [0.8840, 0.9020, 0.9087, 0.9165, 0.8950, 0.9230]
    vram_mb = [485.0, 420.0, 490.0, 539.9, 620.0, 680.0]

    # --- PANEL 1: Grouped Bar Chart ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_facecolor('#111827')

    x = np.arange(len(models))
    width = 0.20

    rects1 = ax1.bar(x - 1.5 * width, ffpp_auc, width, label='FF++ C23 (In-Domain)', color='#38BDF8', edgecolor='#0284C7', linewidth=1)
    rects2 = ax1.bar(x - 0.5 * width, celeb_auc, width, label='Celeb-DF v2 (Unseen OOD)', color='#34D399', edgecolor='#059669', linewidth=1)
    rects3 = ax1.bar(x + 0.5 * width, f2f_auc, width, label='Face2Face (Reenactment)', color='#FBBF24', edgecolor='#D97706', linewidth=1)
    rects4 = ax1.bar(x + 1.5 * width, jpeg_auc, width, label='JPEG Q50 (Compression)', color='#F43F5E', edgecolor='#E11D48', linewidth=1)

    ax1.set_title('Empirical Detection AUROC Across Model Generations', color='#F9FAFB', fontsize=12, fontweight='bold', pad=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, color='#D1D5DB', fontsize=8.5)
    ax1.set_ylabel('Area Under ROC Curve (AUROC)', color='#9CA3AF', fontsize=10)
    ax1.set_ylim(0.45, 1.02)
    ax1.grid(axis='y', linestyle='--', alpha=0.2, color='#9CA3AF')
    ax1.tick_params(colors='#9CA3AF')

    # Spines
    for spine in ax1.spines.values():
        spine.set_color('#374151')

    legend1 = ax1.legend(loc='lower right', facecolor='#1F2937', edgecolor='#374151', fontsize=8.5, labelcolor='#E5E7EB')

    # Highlight Champion badges
    ax1.annotate('Production\nChampion', xy=(3, 0.9165), xytext=(3, 0.97),
                 ha='center', fontsize=7.5, fontweight='bold', color='#38BDF8',
                 arrowprops=dict(arrowstyle='->', color='#38BDF8', lw=1.2))

    ax1.annotate('Highest\nAccuracy', xy=(5, 0.9230), xytext=(5, 0.98),
                 ha='center', fontsize=7.5, fontweight='bold', color='#F43F5E',
                 arrowprops=dict(arrowstyle='->', color='#F43F5E', lw=1.2))

    # --- PANEL 2: Pareto Frontier (Latency vs Accuracy) ---
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor('#111827')

    colors = ['#94A3B8', '#A78BFA', '#60A5FA', '#38BDF8', '#FBBF24', '#F43F5E']
    markers = ['o', 's', '^', '*', 'D', 'P']

    for i in range(len(models)):
        name_clean = models[i].split('\n')[0]
        size = vram_mb[i] / 2.0
        ax2.scatter(latencies[i], macro_aucs[i], s=size, color=colors[i], marker=markers[i],
                    edgecolor='white', linewidth=1.5, zorder=5, label=f"{name_clean} ({vram_mb[i]:.0f}MB)")

        # Label offsets
        dx, dy = 1.5, 0.002
        if i == 3:  # v5
            dx, dy = -16.0, 0.003
        elif i == 5:  # m22
            dx, dy = -14.0, -0.006
        ax2.text(latencies[i] + dx, macro_aucs[i] + dy, name_clean, color=colors[i],
                 fontsize=8, fontweight='bold', zorder=6)

    # 45 ms real-time ceiling threshold
    ax2.axvline(x=45.0, color='#EF4444', linestyle=':', linewidth=1.8, label='RTX 3050 Target Ceiling (45ms)')
    ax2.text(46.0, 0.885, 'Serving Latency Budget\n(Threshold <= 45ms)', color='#EF4444', fontsize=7.5)

    ax2.set_title('Hardware Pareto Frontier: Latency vs. Accuracy (RTX 3050 Laptop GPU)', color='#F9FAFB', fontsize=12, fontweight='bold', pad=14)
    ax2.set_xlabel('P50 Inference Latency per 16-frame Sequence (ms)', color='#9CA3AF', fontsize=10)
    ax2.set_ylabel('Macro AUROC (FF++ C23)', color='#9CA3AF', fontsize=10)
    ax2.set_xlim(15, 85)
    ax2.set_ylim(0.875, 0.935)
    ax2.grid(True, linestyle='--', alpha=0.2, color='#9CA3AF')
    ax2.tick_params(colors='#9CA3AF')

    for spine in ax2.spines.values():
        spine.set_color('#374151')

    # Shaded production zone
    ax2.axvspan(15, 45, ymin=0.6, ymax=1.0, color='#065F46', alpha=0.15, label='Production Deployment Envelope')

    legend2 = ax2.legend(loc='lower right', facecolor='#1F2937', edgecolor='#374151', fontsize=7.5, labelcolor='#E5E7EB')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
    plt.close()
    print(f"Generated model comparison benchmark figure: {output_path}")


def generate_comprehensive_architecture_diagram(output_path: Path):
    """
    Generates a true-to-repo, high-resolution visual diagram showing:
    1. Demuxing & Uniform 16-frame Sampling
    2. MTCNN Face Tracking & 112x112 Cropping
    3. Parallel Track A (Deep Neural Engines: v5 Champion, m18 Kinematics, m22 Ensemble)
    4. Parallel Track B (Multi-Modal Deterministic Forensics: Laplacian, Flow, EAR, FFT)
    5. Calibration Layer (T=2.0502 Scaling, Platt Logistic, Threshold Policies)
    6. Serving & Presentation Layer (FastAPI, Web Dashboard, Groq LLM Analyst)
    """
    fig, ax = plt.subplots(figsize=(16, 8.5), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    fig.patch.set_facecolor('#080C14')
    ax.set_facecolor('#080C14')

    # Helper function for styled rounded cards
    def draw_card(x, y, w, h, title, subtitle, tag, header_color, border_color, body_bg='#101726'):
        # Body box
        card = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            edgecolor=border_color, facecolor=body_bg, linewidth=1.5
        )
        ax.add_patch(card)

        # Header accent bar
        header = patches.FancyBboxPatch(
            (x, y + h - 0.55), w, 0.55,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            edgecolor=header_color, facecolor=header_color, linewidth=1
        )
        ax.add_patch(header)

        # Title
        ax.text(x + w/2, y + h - 0.28, title, ha='center', va='center',
                color='white', fontsize=8.5, fontweight='bold')
        # Subtitle lines
        ax.text(x + w/2, y + (h - 0.55)/2 + 0.1, subtitle, ha='center', va='center',
                color='#CBD5E1', fontsize=7.2, linespacing=1.3)
        # Tag
        if tag:
            tag_box = patches.FancyBboxPatch(
                (x + w/2 - 0.6, y + 0.12), 1.2, 0.32,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                edgecolor='#334155', facecolor='#1E293B', linewidth=0.8
            )
            ax.add_patch(tag_box)
            ax.text(x + w/2, y + 0.28, tag, ha='center', va='center',
                    color='#94A3B8', fontsize=6.5, fontweight='bold')

    # 1. INGESTION COLUMN (x = 0.5 to 2.2)
    draw_card(0.4, 4.6, 2.0, 2.8, "1. Ingestion & Demux", "H.264 / MP4 Container\nPTS Verification\nVariable Frame Rates", "OpenCV / PyAV", "#1E293B", "#334155")
    draw_card(0.4, 1.2, 2.0, 2.8, "2. Face Preprocessor", "MTCNN 3-Stage Cascade\n1.30 Margin Expansion\n112x112 Square Crops", "Tensor [16,3,112,112]", "#0284C7", "#0369A1")

    # Connect Ingestion to Preprocessor
    ax.annotate('', xy=(1.4, 4.0), xytext=(1.4, 4.6),
                arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=2, mutation_scale=14))

    # Fork Arrow from Preprocessor to Track A & Track B
    ax.annotate('', xy=(3.0, 6.0), xytext=(2.4, 2.6),
                arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=2, connectionstyle="arc3,rad=-0.2", mutation_scale=14))
    ax.text(2.6, 4.7, "Cropped Faces\nTensor", color="#38BDF8", fontsize=7.2, fontweight='bold', ha='center')

    ax.annotate('', xy=(3.0, 2.6), xytext=(2.4, 2.6),
                arrowprops=dict(arrowstyle="->", color="#34D399", lw=2, mutation_scale=14))

    # TRACK A: Deep Neural Model Spectrum (Top lane: y = 4.8 to 7.8)
    track_a_bg = patches.FancyBboxPatch(
        (2.9, 4.6), 6.4, 3.4, boxstyle="round,pad=0.1,rounding_size=0.2",
        edgecolor='#1E3A8A', facecolor='#0F172A', alpha=0.8, linewidth=1.2, linestyle='--'
    )
    ax.add_patch(track_a_bg)
    ax.text(3.1, 7.75, "TRACK A: Multi-Engine Deep Neural Spectrum", color="#60A5FA", fontsize=8.5, fontweight='bold')

    # Model cards inside Track A
    draw_card(3.1, 4.9, 2.0, 2.5, "v5 Champion (M13.5)", "EfficientNet-B0\nSoft Spatial Attention\n2-Layer Transformer", "tau* = 0.50 (AUC 0.9165)", "#2563EB", "#1D4ED8")
    draw_card(5.3, 4.9, 1.9, 2.5, "M18 Dual-Stream", "Static Trunk (1280-d)\n+\n3D Diff Kinematics", "Reenactment Specialist", "#7C3AED", "#6D28D9")
    draw_card(7.4, 4.9, 1.8, 2.5, "M22 Platt Ensemble", "Log-Odds Decision Fusion\n0.50 z13 + 0.50 z18\nPlatt Calibrated", "Celeb-DF AUC 0.8213", "#DB2777", "#BE185D")

    # Connect v5 & M18 to M22
    ax.annotate('', xy=(7.4, 6.15), xytext=(5.1, 6.15),
                arrowprops=dict(arrowstyle="->", color="#93C5FD", lw=1.5, mutation_scale=12))
    ax.annotate('', xy=(7.4, 5.7), xytext=(7.2, 5.7),
                arrowprops=dict(arrowstyle="->", color="#C4B5FD", lw=1.5, mutation_scale=12))

    # TRACK B: Multi-Modal Forensic Telemetry (Bottom lane: y = 0.9 to 4.1)
    track_b_bg = patches.FancyBboxPatch(
        (2.9, 0.9), 6.4, 3.4, boxstyle="round,pad=0.1,rounding_size=0.2",
        edgecolor='#065F46', facecolor='#064E3B', alpha=0.3, linewidth=1.2, linestyle='--'
    )
    ax.add_patch(track_b_bg)
    ax.text(3.1, 4.05, "TRACK B: Multi-Modal Deterministic Forensics (M24)", color="#34D399", fontsize=8.5, fontweight='bold')

    draw_card(3.1, 1.2, 1.9, 2.5, "Laplacian Seams", "Pyramidal G1 + L0\nBoundary vs Landmark\nEnergy Concentration", "Seam Localization", "#059669", "#047857")
    draw_card(5.2, 1.2, 1.9, 2.5, "Optical Flow & EAR", "Dense Farneback Flow\nVelocity Divergence\nEAR Ocular Blinks", "Kinematic Physics", "#0D9488", "#0F766E")
    draw_card(7.3, 1.2, 1.9, 2.5, "2D-FFT Spectrum", "Fast Fourier Transform\nAzimuthal Power Decay\nCheckerboard GAN Artifacts", "Spectral Residuals", "#0284C7", "#0369A1")

    # CALIBRATION & DECISION ENGINE (x = 9.8 to 12.0)
    draw_card(9.8, 4.8, 2.4, 2.9, "3. Decision Science", "Temperature Scaling T=2.0502\nMahalanobis OOD Gating (d > 3.5)\nMC-Dropout Variance", "ECE: 0.2729 -> 0.0866", "#D97706", "#B45309")
    draw_card(9.8, 1.2, 2.4, 2.9, "4. Operating Policies", "Standard (tau = 0.5000)\nYouden-Optimal (tau = 0.1912)\nHigh-Sensitivity (tau* = 0.1260)\nHigh-Specificity (tau = 0.6820)", "Deterministic Verdict", "#EA580C", "#C2410C")

    # Connect Track A & B to Calibration
    ax.annotate('', xy=(9.8, 6.25), xytext=(9.2, 6.15),
                arrowprops=dict(arrowstyle="->", color="#F59E0B", lw=2, mutation_scale=14))
    ax.annotate('', xy=(9.8, 2.65), xytext=(9.2, 2.45),
                arrowprops=dict(arrowstyle="->", color="#34D399", lw=2, mutation_scale=14))
    ax.annotate('', xy=(11.0, 4.1), xytext=(11.0, 4.8),
                arrowprops=dict(arrowstyle="->", color="#F59E0B", lw=2, mutation_scale=14))

    # PRESENTATION & SERVING LAYER (x = 12.7 to 15.5)
    draw_card(12.7, 5.0, 2.8, 2.7, "5. REST API & UI", "FastAPI Serving (/api/v1/)\nPredict / Compare / Audit\nGlassmorphic Dashboard\nJET Anomaly Heatmaps", "RTX 3050: 511 Deep FPS", "#4F46E5", "#4338CA")
    draw_card(12.7, 1.2, 2.8, 3.2, "6. Groq AI Analyst", "Evidence-Gated Narrative\nStrict Causal Guardrails\nPost-Gen Consistency Check\nDeterministic Safe Fallback", "Qwen-32B / GPT-OSS", "#9333EA", "#7E22CE")

    # Connect Decision Engine to Serving Layer
    ax.annotate('', xy=(12.7, 6.35), xytext=(12.2, 6.25),
                arrowprops=dict(arrowstyle="->", color="#818CF8", lw=2, mutation_scale=14))
    ax.annotate('', xy=(12.7, 2.8), xytext=(12.2, 2.65),
                arrowprops=dict(arrowstyle="->", color="#C084FC", lw=2, mutation_scale=14))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
    plt.close()
    print(f"Generated comprehensive architecture diagram: {output_path}")


def generate_interactive_svg(output_path: Path):
    """
    Generates a live, animated vector SVG representation of the system pipeline
    with glowing gradients and animated dash strokes that renders directly inside GitHub README.
    """
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 480" width="100%" height="100%" style="background:#080C14; border-radius:12px; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,sans-serif;">
  <defs>
    <!-- Gradients -->
    <linearGradient id="headerGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38BDF8" />
      <stop offset="50%" stop-color="#818CF8" />
      <stop offset="100%" stop-color="#C084FC" />
    </linearGradient>
    <linearGradient id="cardGrad1" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#1E293B" stop-opacity="0.8" />
      <stop offset="100%" stop-color="#0F172A" stop-opacity="0.9" />
    </linearGradient>
    <linearGradient id="glowCyan" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0284C7" />
      <stop offset="100%" stop-color="#38BDF8" />
    </linearGradient>
    <linearGradient id="glowPurple" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#7C3AED" />
      <stop offset="100%" stop-color="#A855F7" />
    </linearGradient>
    <linearGradient id="glowEmerald" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#059669" />
      <stop offset="100%" stop-color="#34D399" />
    </linearGradient>
    <linearGradient id="glowRose" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#E11D48" />
      <stop offset="100%" stop-color="#FB7185" />
    </linearGradient>

    <!-- Filter Glow -->
    <filter id="neonGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="4" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>

    <style>
      .flow-line {
        stroke-dasharray: 8, 8;
        animation: dashMove 1.5s linear infinite;
      }
      .pulse-node {
        animation: pulseEffect 3s ease-in-out infinite;
      }
      @keyframes dashMove {
        to {
          stroke-dashoffset: -16;
        }
      }
      @keyframes pulseEffect {
        0%, 100% { opacity: 0.9; }
        50% { opacity: 1; filter: drop-shadow(0 0 8px rgba(56, 189, 248, 0.4)); }
      }
    </style>
  </defs>

  <!-- Title & Status Header -->
  <text x="40" y="45" font-size="20" font-weight="800" fill="url(#headerGrad)">DEEPFAKE DETECTION ENGINE — SYSTEM DATAFLOW</text>
  <rect x="960" y="24" width="200" height="30" rx="15" fill="#1E293B" stroke="#334155" />
  <circle cx="980" cy="39" r="6" fill="#10B981" />
  <text x="995" y="44" font-size="11" font-weight="700" fill="#E2E8F0">436/436 TESTS PASS</text>

  <!-- Flow Lines (Background) -->
  <!-- 1 to 2 -->
  <path d="M 180 200 L 240 200" stroke="#38BDF8" stroke-width="2.5" class="flow-line" />
  <!-- 2 to 3 (Upper: Neural) -->
  <path d="M 380 180 C 420 180, 420 120, 460 120" fill="none" stroke="#818CF8" stroke-width="2.5" class="flow-line" />
  <!-- 2 to 4 (Lower: Forensics) -->
  <path d="M 380 220 C 420 220, 420 280, 460 280" fill="none" stroke="#34D399" stroke-width="2.5" class="flow-line" />
  <!-- 3 to 5 (Neural to Calibration) -->
  <path d="M 680 120 C 720 120, 720 180, 750 180" fill="none" stroke="#818CF8" stroke-width="2.5" class="flow-line" />
  <!-- 4 to 5 (Forensics to Calibration) -->
  <path d="M 680 280 C 720 280, 720 220, 750 220" fill="none" stroke="#34D399" stroke-width="2.5" class="flow-line" />
  <!-- 5 to 6 -->
  <path d="M 910 200 L 960 200" stroke="#F59E0B" stroke-width="2.5" class="flow-line" />

  <!-- Node 1: Video Input -->
  <g class="pulse-node">
    <rect x="40" y="130" width="140" height="140" rx="12" fill="url(#cardGrad1)" stroke="#0284C7" stroke-width="1.8" />
    <rect x="40" y="130" width="140" height="34" rx="12" fill="url(#glowCyan)" />
    <text x="110" y="152" font-size="12" font-weight="700" fill="#FFFFFF" text-anchor="middle">1. RAW VIDEO</text>
    <text x="110" y="190" font-size="11" fill="#94A3B8" text-anchor="middle">MP4 / AVI / MOV</text>
    <text x="110" y="210" font-size="10" fill="#64748B" text-anchor="middle">PTS Validation</text>
    <text x="110" y="240" font-size="10" font-weight="600" fill="#38BDF8" text-anchor="middle">N=16 Uniform</text>
  </g>

  <!-- Node 2: Face Cascade -->
  <g class="pulse-node">
    <rect x="240" y="130" width="140" height="140" rx="12" fill="url(#cardGrad1)" stroke="#0284C7" stroke-width="1.8" />
    <rect x="240" y="130" width="140" height="34" rx="12" fill="url(#glowCyan)" />
    <text x="310" y="152" font-size="12" font-weight="700" fill="#FFFFFF" text-anchor="middle">2. MTCNN CASCADE</text>
    <text x="310" y="190" font-size="11" fill="#94A3B8" text-anchor="middle">1.30 Face Margin</text>
    <text x="310" y="210" font-size="10" fill="#64748B" text-anchor="middle">112×112 Resolution</text>
    <text x="310" y="240" font-size="10" font-weight="600" fill="#38BDF8" text-anchor="middle">[16, 3, 112, 112]</text>
  </g>

  <!-- Node 3: Deep Neural Engines -->
  <g class="pulse-node">
    <rect x="460" y="60" width="220" height="120" rx="12" fill="url(#cardGrad1)" stroke="#7C3AED" stroke-width="1.8" />
    <rect x="460" y="60" width="220" height="34" rx="12" fill="url(#glowPurple)" />
    <text x="570" y="82" font-size="12" font-weight="700" fill="#FFFFFF" text-anchor="middle">TRACK A: NEURAL ENGINES</text>
    <text x="570" y="115" font-size="11" font-weight="600" fill="#C084FC" text-anchor="middle">v5 Champion (EffNet+Trans)</text>
    <text x="570" y="135" font-size="10" fill="#94A3B8" text-anchor="middle">M18 Kinematics &amp; M22 Ensemble</text>
    <text x="570" y="158" font-size="10" font-weight="600" fill="#A855F7" text-anchor="middle">RTX 3050: 511 Deep FPS</text>
  </g>

  <!-- Node 4: Multi-Modal Forensics -->
  <g class="pulse-node">
    <rect x="460" y="220" width="220" height="120" rx="12" fill="url(#cardGrad1)" stroke="#059669" stroke-width="1.8" />
    <rect x="460" y="220" width="220" height="34" rx="12" fill="url(#glowEmerald)" />
    <text x="570" y="242" font-size="12" font-weight="700" fill="#FFFFFF" text-anchor="middle">TRACK B: FORENSIC SIGNALS</text>
    <text x="570" y="275" font-size="11" font-weight="600" fill="#34D399" text-anchor="middle">Laplacian Boundary Seam (L0)</text>
    <text x="570" y="295" font-size="10" fill="#94A3B8" text-anchor="middle">Farneback Flow &amp; EAR Blinks</text>
    <text x="570" y="318" font-size="10" font-weight="600" fill="#10B981" text-anchor="middle">2D-FFT Azimuthal Power</text>
  </g>

  <!-- Node 5: Calibration & Gating -->
  <g class="pulse-node">
    <rect x="750" y="125" width="160" height="150" rx="12" fill="url(#cardGrad1)" stroke="#D97706" stroke-width="1.8" />
    <rect x="750" y="125" width="160" height="34" rx="12" fill="#D97706" />
    <text x="830" y="147" font-size="12" font-weight="700" fill="#FFFFFF" text-anchor="middle">3. CALIBRATION</text>
    <text x="830" y="180" font-size="11" font-weight="600" fill="#FBBF24" text-anchor="middle">T = 2.0502 Scaling</text>
    <text x="830" y="200" font-size="10" fill="#94A3B8" text-anchor="middle">OOD Mahalanobis Gating</text>
    <text x="830" y="220" font-size="10" fill="#64748B" text-anchor="middle">ECE: 0.27 -> 0.086</text>
    <text x="830" y="250" font-size="10" font-weight="600" fill="#F59E0B" text-anchor="middle">tau: 0.1912 / 0.50</text>
  </g>

  <!-- Node 6: Serving & Presentation -->
  <g class="pulse-node">
    <rect x="960" y="115" width="200" height="170" rx="12" fill="url(#cardGrad1)" stroke="#E11D48" stroke-width="1.8" />
    <rect x="960" y="115" width="200" height="34" rx="12" fill="url(#glowRose)" />
    <text x="1060" y="137" font-size="12" font-weight="700" fill="#FFFFFF" text-anchor="middle">4. SERVING &amp; REPORT</text>
    <text x="1060" y="170" font-size="11" font-weight="600" fill="#FB7185" text-anchor="middle">FastAPI REST (/api/v1/)</text>
    <text x="1060" y="190" font-size="10" fill="#94A3B8" text-anchor="middle">Glassmorphic Dashboard</text>
    <text x="1060" y="215" font-size="10" font-weight="600" fill="#C084FC" text-anchor="middle">Groq AI Analyst Synthesis</text>
    <text x="1060" y="235" font-size="10" fill="#64748B" text-anchor="middle">Strict Causal Guardrails</text>
    <text x="1060" y="260" font-size="10" font-weight="700" fill="#34D399" text-anchor="middle">AUTHENTIC / DEEPFAKE</text>
  </g>

  <!-- Footer Stats -->
  <text x="40" y="440" font-size="11" fill="#64748B">Hardware Envelope: NVIDIA RTX 3050 (6GB) | Peak VRAM: 539.9MB (&lt;10%) | P50 Forward: 24.95ms | E2E Latency: 2.23s</text>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_content, encoding='utf-8')
    print(f"Generated interactive SVG dataflow: {output_path}")


if __name__ == "__main__":
    generate_model_comparison_figure(Path("reports/figures/model_comparison_benchmark.png"))
    generate_comprehensive_architecture_diagram(Path("reports/figures/system_architecture_comprehensive.png"))
    generate_interactive_svg(Path("reports/figures/interactive_dataflow.svg"))
