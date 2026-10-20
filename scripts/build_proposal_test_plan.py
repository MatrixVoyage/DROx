"""
DeepGuard Test Plan and Test Data Documentation Generator.
Generates submission/Test_Data/DeepGuard_Test_Plan_and_Data_Documentation.docx
in future-oriented proposal language, describing the planned testing strategy and datasets.
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

def build_proposal_test_plan(output_path: Path):
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Title Page
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
    r_title = p_title.add_run("DeepGuard: Test Plan & Test Data Documentation")
    r_title.font.name = "Calibri"; r_title.font.size = Pt(22); r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(36)
    r_sub = p_sub.add_run(
        "Proposed Verification Strategy, Benchmark Datasets & Expected Evaluation Criteria\n"
        "[Initial Project Planning Submission]"
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
        "Date of Submission: October 2026"
    )
    r_team.font.name = "Calibri"; r_team.font.size = Pt(11)

    doc.add_page_break()

    # Document Control
    add_header_styled(doc, "Document Control & Test Planning Metadata", level=1)
    rev_headers = ["Document Version", "Project Phase", "Primary Authors", "Scope Summary"]
    rev_data = [
        ("1.0 (Current)", "Project Proposal & Planning", "IVP Group 10 Team", "Initial test plan, proposed evaluation datasets, anti-leakage partitioning, and planned test cases.")
    ]
    create_table_styled(doc, rev_headers, rev_data, [Inches(1.2), Inches(1.8), Inches(1.5), Inches(2.5)])

    # 1. Objectives & Scope
    add_header_styled(doc, "1. Testing Objectives and Verification Scope", level=1)
    add_paragraph_styled(doc,
        "The objective of this Test Plan is to define the testing methodologies, dataset preparation protocols, and evaluation criteria that will verify the DeepGuard deepfake video detection system once implemented. "
        "The proposed testing framework will evaluate: (1) video-level and frame-level detection accuracy across authentic and synthetic media; "
        "(2) cross-manipulation transfer across four distinct forgery families (Deepfakes, Face2Face, FaceSwap, NeuralTextures); "
        "(3) zero-shot cross-dataset generalization on Celeb-DF v2; "
        "(4) robustness under social media perturbations (compression, noise, blurring, downsampling); and "
        "(5) operational latency and GPU VRAM footprint on commodity hardware.")

    # 2. Proposed Datasets
    add_header_styled(doc, "2. Proposed Benchmark Datasets & Storage Strategy", level=1)
    add_paragraph_styled(doc,
        "The evaluation strategy will rely on three standardized deepfake video benchmarks:")

    ds_headers = ["Dataset Name", "Compression Tier", "Video Volume", "Proposed Evaluation Role", "Official Access / Provenance"]
    ds_data = [
        ("FaceForensics++ (FF++)", "C23 (HQ, H.264)", "7,000 videos (16.69 GB)\n1,000 real + 4,000 fake", "Primary in-domain training, validation & 4-fold Leave-One-Family-Out matrix", "TUM Computer Vision Group (Academic request required)"),
        ("Celeb-DF v2", "Standard YouTube", "6,529 videos total\n518 official test videos", "Independent zero-shot cross-dataset generalization benchmark", "Celeb-DF Project (Research access required)"),
        ("DFDC (Challenge)", "Kaggle / Meta", "~100,000 videos (~470 GB)", "Independent cross-dataset benchmark (Subject to Kaggle credentials)", "Kaggle Deepfake Detection Challenge")
    ]
    create_table_styled(doc, ds_headers, ds_data, [Inches(1.5), Inches(1.2), Inches(1.6), Inches(1.7), Inches(1.0)])

    # 3. Data Preparation & Anti-Leakage Protocol
    add_header_styled(doc, "3. Data Preparation & Anti-Leakage Partitioning Protocol", level=1)
    add_paragraph_styled(doc,
        "To guarantee scientific integrity and prevent data leakage, the project will enforce strict subject-level partitioning: "
        "In deepfake video datasets, multiple manipulated videos are synthesized from the same underlying source video. "
        "Splitting videos randomly allows the same actor to appear in both training and test sets, artificially inflating detection scores. "
        "DeepGuard will assign an indivisible group_id to every pristine video and all of its manipulated counterparts. "
        "All partitions (train, validation, test) will be formed by splitting disjoint sets of group_ids. "
        "For FaceForensics++ C23, the 1,000 subjects will be split into 720 training subjects (5,040 videos), 140 validation subjects (980 videos), and 140 test subjects (980 videos).")

    # 4. Planned Test Cases
    add_header_styled(doc, "4. Planned Functional & Stress Test Cases", level=1)
    add_paragraph_styled(doc,
        "The testing framework will implement the following structured test cases during the verification phase:")

    tc_headers = ["Test ID", "Test Title & Objective", "Input Specification", "Expected Success Target"]
    tc_data = [
        ("TC-001", "Authentic Video Specificity Test: Verify detection of clean real videos without false alarms.", "Prisitne videos from FF++ C23 and Celeb-DF v2 test splits.", "Clean specificity >= 90.0% (False Positive Rate <= 10.0%) under high-sensitivity threshold."),
        ("TC-002", "Whole-Face Replacement Test: Verify detection of deepfakes with boundary seams (Deepfakes, FaceSwap).", "Manipulated videos generated by autoencoder swaps and graphics blending.", "Detection accuracy >= 95.0% and ROC-AUC >= 0.95."),
        ("TC-003", "Facial Reenactment Test: Verify detection of expression reenactments (Face2Face).", "Manipulated videos where only facial expressions are transferred via 3DMM.", "ROC-AUC >= 0.78 on unseen Face2Face holdout fold."),
        ("TC-004", "Neural Texture Synthesis Test: Verify detection of GAN-rendered mouth and teeth textures.", "NeuralTextures manipulated videos from FF++ C23.", "ROC-AUC >= 0.90 on unseen NeuralTextures holdout fold."),
        ("TC-005", "Compression Stress Test: Verify model robustness under lossy compression.", "Videos re-encoded with H.264 C40 and JPEG quality factors Q=50, 25, 10.", "AUC retention >= 0.60 under severe compression; no unhandled crashes."),
        ("TC-006", "Resolution Downsampling Test: Verify pipeline handling of lower resolution media.", "Videos downsampled from 1080p to 720p, 480p, and 240p.", "Preprocessing pipeline resizes properly to 112x112 without aspect ratio distortion."),
        ("TC-007", "Occlusion & No-Face Fallback Test: Verify pipeline behavior when faces are missing or occluded.", "Video clips with extreme side profiles, hand occlusions, or no detected face.", "System executes fallback center cropping without raising fatal runtime exceptions."),
        ("TC-008", "Bitwise Determinism Test: Verify repeated inference reproducibility under fixed seed.", "Identical video evaluated 10 consecutive times under fixed seed.", "Delta logits == 0.000000 across all repeated forward passes."),
        ("TC-009", "Model Export Parity Test: Verify numerical parity of exported ONNX runtime models.", "Exported ONNX model evaluated against PyTorch checkpoint.", "Maximum absolute numerical difference between outputs < 1e-5."),
        ("TC-010", "Hardware VRAM & Latency Benchmark: Verify execution fits 6GB RTX 3050 constraints.", "100 forward passes of 16-frame sequences on NVIDIA RTX 3050.", "P50 forward latency <= 35 ms; peak VRAM allocation <= 1.0 GB.")
    ]
    create_table_styled(doc, tc_headers, tc_data, [Inches(0.8), Inches(2.3), Inches(2.1), Inches(1.8)])

    # 5. Planned Evaluation Metrics & Targets
    add_header_styled(doc, "5. Planned Evaluation Metrics & Target Success Criteria", level=1)
    add_paragraph_styled(doc,
        "The project will evaluate models across standardized quantitative metrics, establishing clear target criteria:")

    crit_headers = ["Evaluation Metric", "Mathematical Formulation", "Proposed Target Criterion", "Research Rationale"]
    crit_data = [
        ("ROC-AUC", "Area Under the Receiver Operating Characteristic Curve", ">= 0.90 (In-Domain FF++)\n>= 0.75 (Cross-Dataset)", "Primary threshold-independent ranking metric for binary classification."),
        ("Classification Accuracy", "(TP + TN) / Total Samples", ">= 90.0% (FF++ C23)", "Standard overall correctness indicator across balanced test partitions."),
        ("Clean Specificity", "TN / (TN + FP)", ">= 90.0% (Authentic Media)", "Guarantees authentic user media is not erroneously flagged as deepfake."),
        ("Equal Error Rate (EER)", "Operating point where FPR == FNR", "<= 10.0% (In-Domain)", "Standard biometrics benchmark assessing false accept vs false reject balance."),
        ("Expected Calibration Error", "Weighted difference: sum(|conf(B) - acc(B)|)", "<= 10.0% (Post-Calibration)", "Ensures model confidence reflects empirical probability of correctness."),
        ("Forward Pass Latency", "P50 execution time for 16 frames", "<= 35.0 ms (RTX 3050 GPU)", "Guarantees practical usability for near-real-time video analysis.")
    ]
    create_table_styled(doc, crit_headers, crit_data, [Inches(1.5), Inches(2.0), Inches(1.8), Inches(1.7)])

    # 6. Illustrative Output Template
    add_header_styled(doc, "6. Illustrative Evaluation Output Template", level=1)
    add_paragraph_styled(doc,
        "Note: The following table represents an illustrative evaluation template that will be populated during the testing phase. The values shown below are illustrative placeholders, NOT actual project measurements.", italic=True)

    tmpl_headers = ["Video ID / Filename", "Ground Truth", "Predicted Label", "Calibrated Prob (p_cal)", "OOD Flag", "Execution Status"]
    tmpl_data = [
        ("sample_real_001.mp4", "REAL", "REAL (Illustrative)", "0.084 (Illustrative)", "False", "PLANNED FOR TESTING"),
        ("sample_real_002.mp4", "REAL", "REAL (Illustrative)", "0.112 (Illustrative)", "False", "PLANNED FOR TESTING"),
        ("sample_fake_df_001.mp4", "FAKE (Deepfakes)", "FAKE (Illustrative)", "0.942 (Illustrative)", "False", "PLANNED FOR TESTING"),
        ("sample_fake_f2f_001.mp4", "FAKE (Face2Face)", "FAKE (Illustrative)", "0.825 (Illustrative)", "False", "PLANNED FOR TESTING"),
        ("sample_fake_fs_001.mp4", "FAKE (FaceSwap)", "FAKE (Illustrative)", "0.961 (Illustrative)", "False", "PLANNED FOR TESTING"),
        ("sample_fake_nt_001.mp4", "FAKE (NeuralTextures)", "FAKE (Illustrative)", "0.893 (Illustrative)", "False", "PLANNED FOR TESTING")
    ]
    create_table_styled(doc, tmpl_headers, tmpl_data, [Inches(1.6), Inches(1.3), Inches(1.3), Inches(1.2), Inches(0.7), Inches(0.9)])

    # Save document
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    print(f"Generated Proposal Test Plan DOCX: {output_path}")

if __name__ == "__main__":
    out_docx = Path("submission/Test_Data/DeepGuard_Test_Plan_and_Data_Documentation.docx")
    build_proposal_test_plan(out_docx)
