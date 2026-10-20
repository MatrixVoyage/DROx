"""
DeepGuard Project Proposal & SRS Generator.
Generates submission/SRS_and_Design_Report/DeepGuard_SRS.docx in future-oriented proposal language,
describing the planned DeepGuard deepfake video detection system before implementation.
"""

from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, hex_color):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_header_styled(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.keep_with_next = True
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    for run in h.runs:
        run.font.name = "Calibri"
        if level == 1:
            run.font.size = Pt(17)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        elif level == 2:
            run.font.size = Pt(13.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x2F, 0x55, 0x97)
        elif level == 3:
            run.font.size = Pt(11.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return h

def add_paragraph_styled(doc, text, bold_prefix="", italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Calibri"
        r_pre.font.size = Pt(10.5)
        r_pre.font.bold = True
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(10.5)
    r.font.italic = italic
    return p

def create_table_styled(doc, headers, data, col_widths=None):
    table = doc.add_table(rows=len(data) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, header_text in enumerate(headers):
        hdr_cells[i].text = header_text
        set_cell_background(hdr_cells[i], "1F4E79")
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in p.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(9.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Data Rows
    for r_idx, row_values in enumerate(data):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = "F2F4F7" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_values):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=70, bottom=70, left=100, right=100)
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.name = "Calibri"
                run.font.size = Pt(9.0)

    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = width

    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return table

def build_proposal_srs(output_path: Path):
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # =========================================================================
    # TITLE PAGE (Proposal Stage)
    # =========================================================================
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inst.paragraph_format.space_before = Pt(60)
    r_inst = p_inst.add_run(
        "INDIAN INSTITUTE OF INFORMATION TECHNOLOGY\n"
        "DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING\n"
        "COURSE: IMAGE AND VIDEO PROCESSING (IVP)\n\n"
    )
    r_inst.font.name = "Calibri"; r_inst.font.size = Pt(12); r_inst.font.bold = True
    r_inst.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(12)
    r_title = p_title.add_run("DeepGuard: An Explainable Deepfake Video Detection and Forensic Analysis Platform")
    r_title.font.name = "Calibri"; r_title.font.size = Pt(22); r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(36)
    r_sub = p_sub.add_run(
        "Software Requirements Specification (SRS) & System Design Document\n"
        "[Initial Project Proposal & Planning Submission]"
    )
    r_sub.font.name = "Calibri"; r_sub.font.size = Pt(13); r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_meta.paragraph_format.space_after = Pt(60)
    r_team = p_meta.add_run(
        "Project Group: IVP_Group-10_DeepGuard\n\n"
        "Team Members:\n"
        "IIT2024108  |  IIT2024109  |  IIT2024110\n"
        "IIT2024115  |  IIT2024119  |  IIT2024122\n\n"
        "Document Release: Version 1.0 (Proposal & Planning Phase)\n"
        "Primary Reference Paper: Lal, Shiwani & Gandhi (2025), Scientific Reports (Nature)\n"
        "Date of Submission: October 2026"
    )
    r_team.font.name = "Calibri"; r_team.font.size = Pt(11)

    doc.add_page_break()

    # =========================================================================
    # DOCUMENT CONTROL & PLANNING METADATA
    # =========================================================================
    add_header_styled(doc, "Document Control & Project Planning Metadata", level=1)
    rev_headers = ["Document Version", "Project Phase", "Primary Authors", "Scope Summary"]
    rev_data = [
        ("1.0 (Current)", "Project Proposal & Planning", "IVP Group 10 Team", "Initial project formulation, requirements specification, proposed system architecture, and evaluation plan.")
    ]
    create_table_styled(doc, rev_headers, rev_data, [Inches(1.2), Inches(1.8), Inches(1.5), Inches(2.5)])

    # =========================================================================
    # 1. INTRODUCTION, PURPOSE & SCOPE
    # =========================================================================
    add_header_styled(doc, "1. Introduction, Purpose and Scope", level=1)
    add_paragraph_styled(doc,
        "DeepGuard is a proposed visual deepfake video detection and digital forensic analysis platform designed to authenticate digital video media. "
        "With recent advancements in deep generative modeling, facial manipulation tools can seamlessly replace identities (Deepfakes, FaceSwap), modify facial expressions (Face2Face), and synthesize photorealistic mouth dynamics (NeuralTextures). "
        "These synthetic videos present severe societal risks, including political disinformation, executive impersonation, financial extortion, and reputation defamation. "
        "The purpose of this document is to establish the formal requirements, system architecture, data governance protocols, and testing methodology that will govern the development of the DeepGuard platform.")

    add_header_styled(doc, "1.1 Scope of the Proposed System", level=2)
    add_paragraph_styled(doc,
        "The DeepGuard platform is proposed to encompass: (1) a deterministic temporal video ingestion and facial extraction pipeline utilizing MTCNN face detection; "
        "(2) an initial baseline neural network reproducing the visual soft-attention and BiLSTM architecture of Lal et al. (2025); "
        "(3) planned multi-branch architectural extensions incorporating mid-spatial boundary representations, high-frequency residual filters, and temporal Transformer modeling; "
        "(4) post-hoc statistical calibration via temperature scaling and out-of-distribution (OOD) distance estimation; "
        "(5) a high-performance RESTful API service; and (6) an interactive web dashboard for real-time video upload and forensic evidence inspection. "
        "The system will be developed under strict commodity hardware constraints, designed to operate entirely within 6.0 GB of VRAM on an NVIDIA GeForce RTX 3050 Laptop GPU.")

    add_header_styled(doc, "1.2 Academic Reference Baseline", level=2)
    add_paragraph_styled(doc,
        "The primary methodological reference selected for this project is: "
        "K. Lal, S. Shiwani, and G. C. Gandhi, 'Deepfake video deception detection using visual attention-based method,' Scientific Reports, vol. 15, no. 1, art. 40089, 2025 (DOI: 10.1038/s41598-025-23920-0). "
        "The published paper proposes a 16-frame ResNeXt-50 spatial backbone paired with soft spatial attention and a 2048-hidden bidirectional LSTM, reporting published benchmark AUC scores of 0.962 on FaceForensics++ C23, 0.950 on Celeb-DF v2, and 0.920 on DFDC. "
        "DeepGuard proposes to reproduce this architecture as its foundational baseline, investigate its behavior under compression and unseen manipulations, and engineer robust architectural extensions to address identified generalization gaps.")

    # =========================================================================
    # 2. OVERALL PERSPECTIVE, OPERATING ENVIRONMENT & CONSTRAINTS
    # =========================================================================
    add_header_styled(doc, "2. Overall Product Perspective and Constraints", level=1)
    add_paragraph_styled(doc,
        "The proposed DeepGuard system is designed as a modular, client-server platform where video processing and neural network inference operate independently from model training.")

    add_header_styled(doc, "2.1 Target Operating Environment & Hardware Constraints", level=2)
    add_paragraph_styled(doc,
        "To ensure academic reproducibility and practical deployment, all proposed models and pipelines will be engineered to function within commodity laboratory hardware: "
        "Target Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB dedicated VRAM), 16 GB System RAM, Windows 11 / Linux x64. "
        "Software Stack: Python 3.11, PyTorch >= 2.0 with CUDA 12.x support, torchvision, OpenCV, facenet-pytorch, and FastAPI. "
        "Target Memory Envelope: Peak GPU VRAM allocation during video inference shall remain strictly below 1.0 GB (< 17% of total capacity). "
        "Target Latency: Model forward pass latency will target <= 35 ms per 16-frame sequence, with end-to-end video processing completing in under 3.5 seconds.")

    add_header_styled(doc, "2.2 Proposed User Classes", level=2)
    add_paragraph_styled(doc, "• Forensic Media Analysts: Users requiring frame-level evidence, multi-branch confidence scores, and activation heatmaps to verify authentic vs. synthetic media.")
    add_paragraph_styled(doc, "• Academic Researchers & Students: Evaluators executing standardized benchmarks, running cross-dataset validation, and inspecting architectural ablations.")
    add_paragraph_styled(doc, "• General End-Users: Users seeking a clear, calibrated verdict (REAL or DEEPFAKE) through a straightforward drag-and-drop web dashboard.")

    # =========================================================================
    # 3. PROPOSED FUNCTIONAL REQUIREMENTS
    # =========================================================================
    add_header_styled(doc, "3. Proposed Functional Requirements Specification", level=1)
    add_paragraph_styled(doc,
        "The following functional requirements specify the planned capabilities of the DeepGuard system. Each requirement is assigned a unique identifier, priority, and verification criterion.")

    fr_headers = ["Req ID", "Proposed Operational Contract", "Priority", "Target Verification Method"]
    fr_data = [
        ("FR-001", "The system shall implement deterministic temporal frame sampling, extracting N=16 frames via uniform temporal stride or fixed random seed.", "P0", "Unit tests verifying deterministic index selection under fixed seed."),
        ("FR-002", "The system shall integrate MTCNN face detection to localize facial regions with a planned 0.20 (20%) bounding box margin expansion and bilinear resizing to 112x112.", "P0", "Automated tests asserting output tensor shape (B, 16, 3, 112, 112)."),
        ("FR-003", "The system shall implement the foundational reference baseline: ResNeXt-50 (32x4d) + Soft Spatial Attention + 2-layer Bidirectional LSTM (2048 hidden units).", "P0", "Layer-by-layer parameter count and tensor dimension contract tests."),
        ("FR-004", "The system shall design a multi-branch feature decomposition extension incorporating mid-spatial boundary features, high-frequency residual filters, and temporal Transformer modeling.", "P0", "Gradient flow tests verifying all parallel branches backpropagate gradients."),
        ("FR-005", "The system shall provide a modular model registry supporting independent swapping between the baseline model and research extensions.", "P0", "Predictor factory tests confirming dynamic model instantiation via registry keys."),
        ("FR-006", "The system shall implement post-hoc temperature scaling to calibrate raw logits into statistically meaningful probabilities.", "P0", "Validation script demonstrating reduction of Expected Calibration Error (ECE <= 10%)."),
        ("FR-007", "The system shall calculate penultimate feature Mahalanobis / Euclidean distance against a validation manifold threshold to flag out-of-distribution (OOD) inputs.", "P0", "Unit tests asserting OOD flag is raised on perturbed/novel distribution inputs."),
        ("FR-008", "The system shall establish two operational decision policies: standard balanced classification (tau = 0.50) and a high-sensitivity policy (tau* guaranteeing >=90% clean specificity).", "P0", "Threshold policy validation asserting >=90% clean specificity on authentic validation set."),
        ("FR-009", "The system shall provide a FastAPI REST service exposing endpoints for single-video prediction (/api/v1/predict), model metadata (/api/v1/model-info), and service health (/health).", "P0", "API test client suite asserting HTTP 200 responses and JSON schema compliance."),
        ("FR-010", "The system shall develop an interactive dark-mode web dashboard supporting drag-and-drop video upload, real-time threshold adjustment, and latency breakdown display.", "P0", "Browser interface validation asserting DOM rendering and asynchronous fetch completion."),
        ("FR-011", "The system shall provide a command-line interface (CLI) supporting prediction, evaluation across manifests, and model serving.", "P0", "CLI invocation tests asserting exit code 0 across subcommands."),
        ("FR-012", "The system shall support exporting trained model weights to ONNX and TorchScript runtime formats with high numerical parity.", "P1", "Parity tests asserting maximum absolute difference between PyTorch and ONNX < 1e-5."),
        ("FR-013", "The system shall enforce an indivisible group manifest splitting utility ensuring all video sequences derived from the same source subject reside strictly in the same split.", "P0", "Manifest integrity test asserting zero group overlap between train and test partitions."),
        ("FR-014", "The system shall implement a leakage-safe 5-fold cross-validation partitioner to satisfy academic evaluation requirements.", "P1", "Group-K-Fold unit tests asserting zero subject overlap across all 5 generated folds."),
        ("FR-015", "The system shall incorporate explainable attribution mechanisms producing spatial attention weight heatmaps across facial frames.", "P1", "Attribution generator asserting heatmap array shape matches input spatial dimensions."),
        ("FR-016", "The system shall implement late-stage log-odds evidence fusion to combine predictions from complementary spatial and spectral detector streams.", "P1", "Fusion contract tests asserting monotonicity and convexity constraints.")
    ]
    create_table_styled(doc, fr_headers, fr_data, [Inches(0.8), Inches(3.2), Inches(0.7), Inches(2.3)])

    # =========================================================================
    # 4. PROPOSED NON-FUNCTIONAL REQUIREMENTS
    # =========================================================================
    add_header_styled(doc, "4. Proposed Non-Functional Requirements Specification", level=1)
    nfr_headers = ["Req ID", "Category", "Proposed Performance Target", "Planned Verification Strategy"]
    nfr_data = [
        ("NFR-001", "Latency", "Model forward pass P50 latency shall be <= 35.0 ms per 16-frame sequence on NVIDIA RTX 3050 GPU.", "Benchmark script executing 100 consecutive forward passes with CUDA synchronization."),
        ("NFR-002", "Throughput", "End-to-end video processing latency shall complete within <= 3.5 seconds per 16-frame video clip.", "Wall-clock timing covering OpenCV decoding, MTCNN face extraction, forward pass, and reporting."),
        ("NFR-003", "Memory", "Peak dedicated GPU memory allocation shall not exceed 1.0 GB during video inference.", "PyTorch torch.cuda.max_memory_allocated() profiling during inference."),
        ("NFR-004", "Determinism", "Given identical random seeds, repeated inference on the same input video shall produce bitwise identical logits.", "Automated test asserting delta logits == 0.000000 across multiple runs."),
        ("NFR-005", "Numerical Parity", "Exported ONNX runtime models shall match PyTorch forward pass outputs within absolute error < 1e-5.", "Numerical difference validation comparing ONNX Runtime outputs with PyTorch tensors."),
        ("NFR-006", "Data Integrity", "Zero data leakage shall occur between training, validation, and test partitions across all datasets.", "Group-level validation scripts verifying disjoint sets of source identities."),
        ("NFR-007", "Calibration", "The model shall achieve an Expected Calibration Error (ECE) <= 10.0% on held-out validation data post-calibration.", "Reliability curve computation binning predictions into 10 probability intervals."),
        ("NFR-008", "Clean Specificity", "Under the high-sensitivity threshold policy, the detector shall achieve >= 90.0% specificity on clean authentic media.", "Evaluation on authentic video holdouts asserting False Positive Rate <= 10.0%.")
    ]
    create_table_styled(doc, nfr_headers, nfr_data, [Inches(0.8), Inches(1.1), Inches(2.6), Inches(2.5)])

    # =========================================================================
    # 5. PROPOSED SYSTEM ARCHITECTURE & MODULE DESIGN
    # =========================================================================
    add_header_styled(doc, "5. Proposed System Architecture and Module Design", level=1)
    add_paragraph_styled(doc,
        "The proposed DeepGuard architecture decomposes the video forensics lifecycle into modular, decoupled subsystems:")

    add_header_styled(doc, "5.1 Video Ingestion & Deterministic Sampling Module", level=2)
    add_paragraph_styled(doc,
        "The video ingestion module will accept video containers in MP4, AVI, MOV, or MKV formats. "
        "Using OpenCV / Decord, the module will inspect container metadata and verify that the video contains at least 16 decodable frames. "
        "A deterministic uniform temporal sampler will extract N=16 frames distributed evenly across the video duration. "
        "To ensure reproducibility during testing, random sampling will support seed derivation based on the SHA-256 hash of the video file.")

    add_header_styled(doc, "5.2 MTCNN Face Localization & Preprocessing Module", level=2)
    add_paragraph_styled(doc,
        "Extracted raw frames will be processed by a Multi-Task Cascaded Convolutional Network (MTCNN) to detect facial bounding boxes and 5 facial landmarks. "
        "To ensure that subtle boundary blending artifacts along the jawline, forehead, and ears are captured, the bounding box will be expanded by 20% (margin factor = 0.20). "
        "The isolated facial region will be bilinearly resized to 112 x 112 x 3 pixels and standardized using ImageNet mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]. "
        "The output will be formatted as a 5D sequence tensor of shape (B, 16, 3, 112, 112). If no face is detected, a fallback center crop will be applied to prevent pipeline crashing.")

    add_header_styled(doc, "5.3 Baseline Feature Extraction & Attention Module", level=2)
    add_paragraph_styled(doc,
        "Following the reference paper (Lal et al., 2025), the foundational model will employ an ImageNet-pretrained ResNeXt-50 (32x4d) backbone. "
        "The spatial convolutional stages will output feature maps of shape (B*T, 2048, 4, 4). "
        "A soft spatial attention module using a 1x1 convolution will compute spatial importance weights across the 4x4 grid via spatial softmax. "
        "A weighted sum across spatial coordinates will yield attended 2048-dimensional feature vectors per frame.")

    add_header_styled(doc, "5.4 Multi-Branch Research Extension", level=2)
    add_paragraph_styled(doc,
        "To overcome the known limitation where boundary detectors fail on expression reenactment (Face2Face), DeepGuard proposes a multi-branch feature decomposition: "
        "(1) A mid-spatial branch extracting Stage 5 features from an EfficientNet-B0 backbone to preserve localized boundary seams; "
        "(2) A high-frequency residual branch applying Laplacian and Spatial Rich Model (SRM) filter banks to isolate compression-resistant frequency discrepancies; and "
        "(3) A texture anomaly branch analyzing spatial gradient variance. "
        "These multi-domain representations will be aggregated across time using a 4-head Temporal Transformer encoder.")

    add_header_styled(doc, "5.5 Statistical Calibration & Decision Layer", level=2)
    add_paragraph_styled(doc,
        "To prevent misleading overconfident predictions, raw logits z will be calibrated using post-hoc temperature scaling: p_cal = sigmoid(z / T). "
        "The scalar temperature T will be fitted strictly on held-out validation sequences by minimizing negative log-likelihood (NLL). "
        "Additionally, an out-of-distribution (OOD) gating estimator will compute the regularized Mahalanobis distance of penultimate features against an empirical validation threshold (3.18 at 99th percentile), flagging anomalous or out-of-domain media.")

    add_header_styled(doc, "5.6 FastAPI Service & Web Forensics Dashboard", level=2)
    add_paragraph_styled(doc,
        "The backend will be deployed as an asynchronous FastAPI microservice exposing multipart upload endpoints. "
        "The user interface will be developed as a lightweight, dependency-free dark-mode web dashboard featuring: "
        "a video player preview, real-time threshold slider (toggling between standard tau=0.50 and high-sensitivity tau*), side-by-side model comparison, and a processing latency waterfall breakdown.")

    # =========================================================================
    # 6. DATASET REQUIREMENTS & ANTI-LEAKAGE STRATEGY
    # =========================================================================
    add_header_styled(doc, "6. Dataset Requirements & Anti-Leakage Protocol", level=1)
    add_paragraph_styled(doc,
        "The DeepGuard evaluation plan requires three distinct benchmark datasets:")
    add_paragraph_styled(doc, "• FaceForensics++ (FF++) C23: Primary benchmark dataset consisting of 1,000 real YouTube videos and 4,000 manipulated videos across Deepfakes, Face2Face, FaceSwap, and NeuralTextures, compressed at H.264 CRF 23. This dataset will serve for in-domain training, validation, and Leave-One-Family-Out cross-manipulation matrix benchmarking.")
    add_paragraph_styled(doc, "• Celeb-DF v2: External evaluation dataset containing 518 official test videos (178 real, 340 synthetic) exhibiting advanced boundary blending. Celeb-DF v2 will be reserved strictly for zero-shot cross-dataset generalization testing without retraining.")
    add_paragraph_styled(doc, "• Deepfake Detection Challenge (DFDC): Planned external evaluation benchmark comprising ~100,000 videos. Access will depend on Kaggle competition credential authorization.")
    add_paragraph_styled(doc, "• Indivisible Group Partitioning Invariant: All dataset splits will be partitioned strictly by donor subject identity (group_id). Under no circumstances will manipulated derivatives of a subject appear in a different partition from the pristine source video, guaranteeing zero identity and blending leakage.")

    # =========================================================================
    # 7. PROPOSED DATABASE & API DESIGN
    # =========================================================================
    add_header_styled(doc, "7. Proposed Database & API Design", level=1)
    add_paragraph_styled(doc,
        "The system will define lightweight relational entities to track forensic analyses: "
        "videos (id, filename, sha256_hash, duration, fps), analyses (id, video_id, engine_key, threshold, calibrated_score, verdict, ood_flag, created_at), and frame_evidence (analysis_id, frame_no, timestamp, attention_score).")
    add_paragraph_styled(doc, "Proposed REST Endpoints:")
    add_paragraph_styled(doc, "• POST /api/v1/predict — Multipart video file upload. Returns predicted verdict, calibrated probability, OOD flag, and latency metrics.")
    add_paragraph_styled(doc, "• POST /api/v1/compare — Dual-engine execution comparing the baseline ResNeXt model against the proposed multi-branch model.")
    add_paragraph_styled(doc, "• GET /api/v1/model-info — Returns active model engine metadata, parameters, and limitation disclosures.")
    add_paragraph_styled(doc, "• GET /health — Returns GPU allocation, active device, and memory consumption.")

    # =========================================================================
    # 8. TECHNICAL RISKS, FEASIBILITY & MITIGATION STRATEGY
    # =========================================================================
    add_header_styled(doc, "8. Technical Risks, Feasibility & Mitigation Strategy", level=1)
    risk_headers = ["Identified Risk", "Impact", "Planned Technical Mitigation"]
    risk_data = [
        ("GPU VRAM Exhaustion on 6GB RTX 3050", "High", "Adopt mixed precision (FP16/AMP), micro-batching (batch size 4), gradient accumulation (2 steps), and frozen backbone stems."),
        ("Identity / Subject Data Leakage", "Critical", "Enforce indivisible group_id partitioning; verify disjoint subject sets via automated unit tests prior to training."),
        ("Generalization Drop on Unseen Forgeries", "High", "Implement Leave-One-Family-Out (LOFO) matrix benchmarking and multi-branch frequency/texture representations."),
        ("Overconfident Probabilities / False Alarms", "High", "Incorporate post-hoc temperature scaling fitted strictly on validation data, and derive operating thresholds guaranteeing >=90% clean specificity."),
        ("Face Detection Failure on Occluded Faces", "Medium", "Implement fallback central cropping and landmark quality verification in the preprocessing module."),
        ("External Dataset Access Restrictions", "Medium", "Develop modular dataset adapters (e.g. for DFDC) and maintain standalone manifests allowing offline evaluation once data is authorized.")
    ]
    create_table_styled(doc, risk_headers, risk_data, [Inches(1.8), Inches(0.8), Inches(4.4)])

    # =========================================================================
    # 9. PLANNED ACTIVITY TIMELINE & WORK BREAKDOWN
    # =========================================================================
    add_header_styled(doc, "9. Planned Activity Timeline & 16-Week Phase Plan", level=1)
    time_headers = ["Project Phase", "Scheduled Duration", "Key Planned Activities & Milestone Objectives"]
    time_data = [
        ("Phase 1: Requirements & Literature", "Weeks 1 – 2", "Formalize SRS; conduct comprehensive survey of 18 peer-reviewed papers; analyze base paper methodology."),
        ("Phase 2: Dataset Acquisition & Pipeline", "Weeks 3 – 4", "Obtain FaceForensics++ C23 access; implement OpenCV decoding, uniform sampling, MTCNN face localization, and group splitting."),
        ("Phase 3: Baseline Model Reproduction", "Weeks 5 – 7", "Implement ResNeXt-50 + Soft Attention + BiLSTM; train on FF++ C23 under mixed precision; evaluate baseline metrics."),
        ("Phase 4: Multi-Branch Invariant Design", "Weeks 8 – 10", "Develop mid-spatial Stage 5, high-frequency residual, and texture branches; integrate Temporal Transformer; execute 4-fold LOFO matrix."),
        ("Phase 5: Calibration & Forensics", "Weeks 11 – 12", "Fit post-hoc temperature scaling; implement penultimate Mahalanobis OOD gating; evaluate cross-dataset transfer on Celeb-DF v2."),
        ("Phase 6: Serving & Dashboard", "Weeks 13 – 14", "Build FastAPI microservice; develop interactive dark-mode web dashboard; implement ONNX / TorchScript model export."),
        ("Phase 7: Testing & Documentation", "Weeks 15 – 16", "Execute comprehensive regression test suite; finalize design reports, presentation deck, and academic submission package.")
    ]
    create_table_styled(doc, time_headers, time_data, [Inches(1.8), Inches(1.3), Inches(3.9)])

    # Save document
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    print(f"Generated Proposal SRS DOCX: {output_path}")

if __name__ == "__main__":
    out_docx = Path("submission/SRS_and_Design_Report/DeepGuard_SRS.docx")
    build_proposal_srs(out_docx)
