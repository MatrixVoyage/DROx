"""
DeepGuard Project Proposal Presentation Generator (26 Slides with Visuals).
Generates submission/PPT/DeepGuard_Project_Presentation.pptx (16:9 Widescreen)
in future-oriented proposal language, describing the proposed system before implementation.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_proposal_presentation(output_pptx: Path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Theme Colors
    C_NAVY = RGBColor(0x0F, 0x24, 0x47)
    C_DARK_BG = RGBColor(0x0A, 0x11, 0x28)
    C_CARD_BG = RGBColor(0xFF, 0xFF, 0xFF)
    C_CARD_BORDER = RGBColor(0xCB, 0xD5, 0xE1)
    C_TEXT_DARK = RGBColor(0x1E, 0x29, 0x3B)
    C_TEXT_MUTED = RGBColor(0x64, 0x74, 0x8B)
    C_PRIMARY = RGBColor(0x1D, 0x4E, 0xD8)
    C_EMERALD = RGBColor(0x05, 0x96, 0x69)
    C_AMBER = RGBColor(0xD9, 0x77, 0x06)
    C_ROSE = RGBColor(0xE1, 0x1D, 0x48)

    def add_header(slide, title_text, category_text="IVP GROUP 10 — INITIAL PROJECT PROPOSAL"):
        tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.7), Inches(0.3))
        tf_cat = tb_cat.text_frame; tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]; p_cat.text = category_text.upper()
        p_cat.font.name = "Calibri"; p_cat.font.size = Pt(10); p_cat.font.bold = True; p_cat.font.color.rgb = C_PRIMARY

        tb_t = slide.shapes.add_textbox(Inches(0.8), Inches(0.68), Inches(11.7), Inches(0.65))
        tf_t = tb_t.text_frame; tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]; p_t.text = title_text
        p_t.font.name = "Calibri"; p_t.font.size = Pt(21); p_t.font.bold = True; p_t.font.color.rgb = C_NAVY

    def add_citation_footer(slide, citation_text):
        tb_c = slide.shapes.add_textbox(Inches(0.8), Inches(6.92), Inches(11.7), Inches(0.35))
        tf_c = tb_c.text_frame
        p_c = tf_c.paragraphs[0]
        p_c.text = f"Reference / Citation: {citation_text}"
        p_c.font.name = "Calibri"; p_c.font.size = Pt(8.5); p_c.font.italic = True; p_c.font.color.rgb = C_TEXT_MUTED

    def add_card(slide, left, top, width, height, bg_color=C_CARD_BG, border_color=C_CARD_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
        shape.fill.solid(); shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = border_color; shape.line.width = Pt(1)
        return shape

    # =========================================================================
    # SLIDE 1: Title Slide (Dark Theme)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    add_card(s1, Inches(0), Inches(0), Inches(13.333), Inches(7.5), C_DARK_BG, C_DARK_BG)
    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.333), Inches(5.2))
    tf1 = tb1.text_frame; tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "IMAGE AND VIDEO PROCESSING (IVP) — PROJECT PROPOSAL & SYNOPSIS DEFENSE"
    p0.font.name = "Calibri"; p0.font.size = Pt(13); p0.font.bold = True; p0.font.color.rgb = RGBColor(0x38, 0xBD, 0xF8); p0.space_after = Pt(14)

    p1 = tf1.add_paragraph()
    p1.text = "DeepGuard: An Explainable Deepfake Video Detection and Forensic Analysis Platform"
    p1.font.name = "Calibri"; p1.font.size = Pt(30); p1.font.bold = True; p1.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); p1.space_after = Pt(14)

    p2 = tf1.add_paragraph()
    p2.text = "A Proposed Multi-Engine Visual Forensics System: Base Paper Reproduction and Multi-Family Invariant Extensions"
    p2.font.name = "Calibri"; p2.font.size = Pt(15); p2.font.italic = True; p2.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8); p2.space_after = Pt(28)

    p3 = tf1.add_paragraph()
    p3.text = "Project Group: IVP_Group-10_DeepGuard\nTeam Members: IIT2024108 | IIT2024109 | IIT2024110 | IIT2024115 | IIT2024119 | IIT2024122"
    p3.font.name = "Calibri"; p3.font.size = Pt(13); p3.font.bold = True; p3.font.color.rgb = RGBColor(0xFB, 0xBF, 0x24); p3.space_after = Pt(8)

    p4 = tf1.add_paragraph()
    p4.text = "Primary Research Baseline: Lal, Shiwani & Gandhi (2025), Scientific Reports | Hardware Target: NVIDIA RTX 3050 (6GB VRAM)"
    p4.font.name = "Calibri"; p4.font.size = Pt(11); p4.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # =========================================================================
    # SLIDE 2: Table of Contents / Presentation Roadmap
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Presentation Roadmap & Agenda", "OVERVIEW")
    add_citation_footer(s2, "DeepGuard Project Proposal & Initial Design Document.")

    agenda_items = [
        ("1. Context & Motivation", "• The Threat Landscape\n• Scientific Hurdles in Video\n• Project Scope & Formal Goals"),
        ("2. Literature & Base Paper", "• Literature Survey (18 Studies)\n• Top 5 Papers Comparison\n• Lal et al. (2025) Base Method"),
        ("3. Proposed Architecture", "• End-to-End System Pipeline\n• Ingestion & MTCNN Preprocessing\n• Multi-Branch Extension Design"),
        ("4. Trust, Data & Timeline", "• Post-Hoc Calibration & OOD\n• Anti-Leakage Group Splitting\n• 16-Week Implementation Plan")
    ]
    for i, (atitle, atext) in enumerate(agenda_items):
        x = Inches(0.8 + i * 3.0)
        add_card(s2, x, Inches(1.5), Inches(2.8), Inches(5.1))
        tb = s2.shapes.add_textbox(x + Inches(0.15), Inches(1.7), Inches(2.5), Inches(4.7))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = atitle; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = C_PRIMARY; p.space_after = Pt(8)
        p2 = tf.add_paragraph(); p2.text = atext; p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 3: Problem Statement & Motivation
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Problem Statement & Societal Motivation", "1. CONTEXT & THREAT LANDSCAPE")
    add_citation_footer(s3, "Tolosana et al., 'DeepFakes Evolution: Analysis of Facial Manipulation', Information Fusion, 2020.")

    cards3 = [
        ("The Threat Landscape", "• Proliferation of generative AI enables photorealistic face swapping, expression reenactment, and synthetic video synthesis.\n\n• High-consequence risks: Political disinformation, executive impersonation, financial fraud, and defamatory synthetic media.\n\n• Urgent need: Trustworthy detection with verifiable chain of custody.", C_NAVY),
        ("The Technical Dilemma", "• Compression loss: Social media re-compression (H.264 C23/C40) destroys subtle blending boundaries.\n\n• Unseen forgeries: Detectors overfit to seen manipulation artifacts, failing on novel architectures (>20% AUC drop).\n\n• Overconfidence: Uncalibrated probabilities produce high false alarms on clean media.", C_PRIMARY),
        ("Proposed DeepGuard Objective", "• Deliver an end-to-end, calibrated forensics pipeline executing on commodity GPUs (<1.0 GB VRAM).\n\n• Reproduce base paper baseline while designing multi-family invariant representations.\n\n• Full transparency: Multi-branch evidence, ECE calibration, and OOD flagging.", C_EMERALD)
    ]
    for i, (ctitle, ctext, ccolor) in enumerate(cards3):
        x = Inches(0.8 + i * 4.0)
        add_card(s3, x, Inches(1.4), Inches(3.7), Inches(5.3))
        tb = s3.shapes.add_textbox(x + Inches(0.2), Inches(1.6), Inches(3.3), Inches(4.9))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = ctitle; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = ccolor; p.space_after = Pt(8)
        p2 = tf.add_paragraph(); p2.text = ctext; p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 4: Scientific Challenges in Video Forensics
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Core Scientific Challenges in Video Forensics", "1. THE SCIENTIFIC HURDLES")
    add_citation_footer(s4, "Rössler et al. (ICCV 2019) and Liu et al. (CVPR 2021).")

    challenges4 = [
        ("1. Compression Artifact Masking", "Video codecs (H.264 / HEVC) apply discrete cosine transforms and quantization, discarding high-frequency boundary telltales. Deepfake detectors relying purely on spatial RGB seams degrade by >14% accuracy from Raw to C40 compression.", C_ROSE),
        ("2. The Generalization Abyss", "Deepfake models trained on one manipulation family (e.g., Deepfakes) frequently drop to near random guess (AUC ~0.55-0.65) when tested on unseen families like Face2Face expression reenactment.", C_AMBER),
        ("3. Severe Calibration Drift", "Standard neural network outputs produce extreme probabilities (0.01 or 0.99) that do not match empirical likelihoods. Uncalibrated models exhibit Expected Calibration Error (ECE) > 25%, resulting in catastrophic false alarms on clean videos.", C_PRIMARY),
        ("4. Consumer Hardware Constraints", "State-of-the-art vision models require large compute clusters. Engineering a production-grade temporal deepfake detector that fits inside 6 GB of VRAM without sacrificing detection power is an open challenge.", C_NAVY)
    ]
    for i, (ctitle, ctext, ccolor) in enumerate(challenges4):
        x = Inches(0.8 + (i % 2) * 6.0)
        y = Inches(1.4 + (i // 2) * 2.7)
        add_card(s4, x, y, Inches(5.7), Inches(2.4))
        tb = s4.shapes.add_textbox(x + Inches(0.2), y + Inches(0.2), Inches(5.3), Inches(2.0))
        tf = tb.text_frame; tf.word_wrap = True
        p1 = tf.paragraphs[0]; p1.text = ctitle; p1.font.size = Pt(13); p1.font.bold = True; p1.font.color.rgb = ccolor; p1.space_after = Pt(6)
        p2 = tf.add_paragraph(); p2.text = ctext; p2.font.size = Pt(10); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 5: Proposed Project Objectives & Scope Boundaries
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Proposed Project Objectives & Scope Boundaries", "2. GOALS & ARCHITECTURAL INVARIANTS")
    add_citation_footer(s5, "DeepGuard Initial SRS and System Design Document.")

    objs5 = [
        ("Obj 1: Base Paper Reproduction", "Implement, calibrate, and evaluate the published ResNeXt-50 + Soft Attention + BiLSTM model (Lal et al., 2025) on FaceForensics++ C23 with verifiable provenance.", C_PRIMARY),
        ("Obj 2: Generalization & Invariance", "Overcome the within-manifold generalization failure mode on unseen manipulation families via family-invariant representations and temporal Transformer modeling.", C_EMERALD),
        ("Obj 3: Trust, Calibration & OOD", "Enforce statistical calibration via post-hoc temperature scaling (target ECE <= 10%) and penultimate Mahalanobis distance gating to prevent uncalibrated overconfidence.", C_AMBER),
        ("Obj 4: Production-Grade System", "Deliver a complete modular ecosystem featuring a multi-engine REST API, real-time web dashboard, ONNX/TorchScript export, and 100% automated regression coverage.", C_NAVY)
    ]
    for i, (otitle, otext, ocolor) in enumerate(objs5):
        x = Inches(0.8 + (i % 2) * 6.0)
        y = Inches(1.4 + (i // 2) * 2.7)
        add_card(s5, x, y, Inches(5.7), Inches(2.4))
        tb = s5.shapes.add_textbox(x + Inches(0.2), y + Inches(0.2), Inches(5.3), Inches(2.0))
        tf = tb.text_frame; tf.word_wrap = True
        p1 = tf.paragraphs[0]; p1.text = otitle; p1.font.size = Pt(13); p1.font.bold = True; p1.font.color.rgb = ocolor; p1.space_after = Pt(6)
        p2 = tf.add_paragraph(); p2.text = otext; p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 6: Literature Survey: State-of-the-Art Taxonomy
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Literature Survey: State-of-the-Art Taxonomy", "3. SCHOLARLY STATE-OF-THE-ART")
    add_citation_footer(s6, "18 verified research papers in submission/Literature_Survey/DeepGuard_Literature_Survey.xlsx.")

    tb_s6 = s6.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.3))
    tf_s6 = tb_s6.text_frame; tf_s6.word_wrap = True

    p = tf_s6.paragraphs[0]; p.text = "Systematic Analysis of 18 Peer-Reviewed Studies Across Detection Modalities"; p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_NAVY; p.space_after = Pt(6)
    p2 = tf_s6.add_paragraph()
    p2.text = (
        "1. Spatial-Only CNN Baselines (MesoNet [Afchar 2018], Xception [Rössler 2019], CapsuleNet [Nguyen 2019]):\n"
        "   Achieve >95% accuracy on uncompressed frames, but suffer catastrophic failure (>15-20% drop) under lossy H.264 compression (C23/C40).\n\n"
        "2. Spatial Attention Mechanisms (Lal et al. [2025], ADD [Khormali 2021], Roy et al. [2022]):\n"
        "   Focus convolutional receptive fields onto localized facial regions (eyes, nose, mouth boundary), preventing artifact dilution from background pixels.\n\n"
        "3. Frequency & Phase Domain Forensics (FAD/LFS [Qian 2020], SPSL [Liu 2021], Fake-checker [Huda 2024]):\n"
        "   Expose high-frequency discrepancies and phase mismatches that survive spatial blurring, offering vital orthogonal signals to spatial CNNs.\n\n"
        "4. Temporal Sequence Modeling (ConvLSTM [Ismail 2021], ST-Transformer [Gu 2022], Optical Flow [DSP 2023]):\n"
        "   Track inter-frame temporal jitter, warping seams, and physiological dynamics across multi-frame video sequences.\n\n"
        "5. Generalization & Boundary Detectors (Face X-ray [Li 2020], GenConViT [Wodajo 2023]):\n"
        "   Attempt zero-shot transfer by isolating blending boundaries; however, within-manifold expression manipulations (Face2Face) remain unaddressed."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 7: Top 5 Relevant Research Studies In-Depth
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Top 5 Most Relevant Research Studies Comparison", "3. SCHOLARLY BENCHMARK COMPARISON")
    add_citation_footer(s7, "Detailed multi-dimensional comparison in DeepGuard_Literature_Survey.xlsx (Sheet 2).")

    tbl_s7 = s7.shapes.add_table(6, 5, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.3)).table
    tbl_s7.columns[0].width = Inches(1.8)
    tbl_s7.columns[1].width = Inches(2.2)
    tbl_s7.columns[2].width = Inches(2.4)
    tbl_s7.columns[3].width = Inches(2.5)
    tbl_s7.columns[4].width = Inches(2.8)

    top5_summary = [
        ("Paper & Citation", "Proposed Architecture", "Datasets Evaluated", "Published Results [Ref]", "Relevance to DeepGuard Proposal"),
        ("P-01: Lal et al. (2025)\nSci. Rep. 15(1)", "ResNeXt-50 + Soft Spatial Attention + BiLSTM (2048)", "FF++ C23, Celeb-DF v2, DFDC", "FF++: AUC 0.962\nCeleb-DF: AUC 0.950\nDFDC: AUC 0.920", "Selected as foundational baseline to be reproduced in Phase 1."),
        ("P-02: Rössler et al. (2019)\nICCV 2019", "XceptionNet, MesoNet, Steganalysis", "FF++ (Raw, C23, C40 across 4 families)", "Xception C23: 95.73%\nC40: 81.00% Acc", "Source dataset provider (FF++ C23); defines 4-family manipulation taxonomy."),
        ("P-03: Qian et al. (2020)\nECCV 2020", "Frequency-aware Decomposition (FAD) + Local Freq (LFS)", "FF++ (Raw, C23, C40), Celeb-DF", "FF++ C23: AUC 0.992\nCeleb-DF: AUC 0.768", "Motivates proposed high-frequency residual branch and spectral analysis."),
        ("P-04: Liu et al. (2021)\nCVPR 2021", "Spatial-Phase Shallow Learning (SPSL) via FFT", "FF++ (C23, C40), Celeb-DF", "FF++ C23: AUC 0.953\nCeleb-DF: AUC 0.778", "Informs proposed Phase C analysis on Face2Face within-manifold overlap."),
        ("P-07: Li et al. (2020)\nCVPR 2020", "Face X-ray synthetic blending boundary detector", "FF++ (C23, C40), Celeb-DF, DFD", "FF++ C23: AUC 0.985\nCeleb-DF: AUC 0.748", "Theoretical proof of why boundary detectors fail on Face2Face expression reenactment.")
    ]
    for r_i, r_data in enumerate(top5_summary):
        for c_i, val in enumerate(r_data):
            cell = tbl_s7.cell(r_i, c_i)
            cell.text = val
            p = cell.text_frame.paragraphs[0]; p.font.name = "Calibri"
            if r_i == 0:
                p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                cell.fill.solid(); cell.fill.fore_color.rgb = C_NAVY
            else:
                p.font.size = Pt(9.5); p.font.color.rgb = C_TEXT_DARK
                cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0xF1, 0xF5, 0xF9) if r_i % 2 == 1 else RGBColor(0xFF, 0xFF, 0xFF)

    # =========================================================================
    # SLIDE 8: The Four Manipulation Families (WITH 4 FACE CROP IMAGES)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Target Forgery Types: The Four Manipulation Families", "4. DATASET & FORGERY TAXONOMY")
    add_citation_footer(s8, "Standard 112x112 crops from FaceForensics++ C23 benchmark dataset.")

    fams8 = [
        ("Deepfakes (DF)", "Identity replacement using autoencoder pairs. Produces boundary blending seams and color transfer mismatches.", "reports/figures/errors/family_examples/fam_Deepfakes_df_004_982.jpg"),
        ("Face2Face (F2F)", "Facial expression reenactment using 3DMM. Smoothly deforms mouth/eyes without synthetic blending seams.", "reports/figures/errors/family_examples/fam_Face2Face_f2f_004_982.jpg"),
        ("FaceSwap (FS)", "Classic graphics-based face replacement and Poisson blending. Prominent color contrast discrepancies along jawline.", "reports/figures/errors/family_examples/fam_FaceSwap_fs_004_982.jpg"),
        ("NeuralTextures (NT)", "Neural texture rendering and GAN-based mouth synthesis. Creates micro-texture blurring and teeth anomalies.", "reports/figures/errors/family_examples/fam_NeuralTextures_nt_004_982.jpg")
    ]
    for i, (fname, fdesc, img_path) in enumerate(fams8):
        x = Inches(0.8 + i * 3.0)
        card = add_card(s8, x, Inches(1.4), Inches(2.8), Inches(5.3))
        tb = s8.shapes.add_textbox(x + Inches(0.1), Inches(1.5), Inches(2.6), Inches(0.8))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = fname; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = C_NAVY

        if Path(img_path).exists():
            s8.shapes.add_picture(img_path, x + Inches(0.4), Inches(2.3), Inches(2.0), Inches(2.0))

        tb2 = s8.shapes.add_textbox(x + Inches(0.15), Inches(4.5), Inches(2.5), Inches(2.1))
        tf2 = tb2.text_frame; tf2.word_wrap = True
        p2 = tf2.paragraphs[0]; p2.text = fdesc; p2.font.size = Pt(9.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 9: Research Gap: The Within-Manifold Confinement Problem
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Research Gap: The 'Within-Manifold' Confinement Challenge", "5. SCIENTIFIC GAP & NOVELTY")
    add_citation_footer(s9, "Analysis of facial expression reenactment mechanisms in deepfake literature.")

    c_g1 = add_card(s9, Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tb_g1 = s9.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(5.2), Inches(4.9))
    tf_g1 = tb_g1.text_frame; tf_g1.word_wrap = True
    p = tf_g1.paragraphs[0]; p.text = "Why Do Existing Detectors Fail on Face2Face?"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_ROSE; p.space_after = Pt(8)
    p2 = tf_g1.add_paragraph()
    p2.text = (
        "• The Boundary Assumption Flaw: Most established deepfake detectors search for boundary blending seams where a synthetic donor face is pasted onto a background.\n\n"
        "• Reenactment Mechanism: Face2Face does NOT paste an external face. Instead, it fits a 3D Morphable Model (3DMM) to the source subject and warps internal facial mesh expressions directly.\n\n"
        "• Zero Boundary Discontinuity: Because the face belongs to the original individual, no boundary blending seam is produced, evading boundary classifiers."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    c_g2 = add_card(s9, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.3))
    tb_g2 = s9.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.3), Inches(4.9))
    tf_g2 = tb_g2.text_frame; tf_g2.word_wrap = True
    p = tf_g2.paragraphs[0]; p.text = "Proposed DeepGuard Solution"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_PRIMARY; p.space_after = Pt(8)
    p2 = tf_g2.add_paragraph()
    p2.text = (
        "• The Facial Manifold Dilemma: In feature space, Face2Face modifications reside extremely close to pristine authentic faces (high density overlap).\n\n"
        "• Proposed Multi-Branch Learning: DeepGuard proposes to decompose representations into:\n\n"
        "  1. Mid-spatial Stage 5 branch to retain localized facial structure.\n"
        "  2. High-frequency residual filters (Laplacian/SRM) to capture unnatural warping gradients.\n"
        "  3. Texture anomaly branch to detect micro-texture inconsistencies.\n\n"
        "• Expected Outcome: Recover detection capability on expression reenactment without sacrificing clean specificity."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 10: Selected Base Paper: Methodology of Lal et al. (2025)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Selected Base Paper: Lal, Shiwani & Gandhi (2025)", "6. BASELINE METHODOLOGY")
    add_citation_footer(s10, "K. Lal, S. Shiwani, G. C. Gandhi, Scientific Reports, 15(1):40089, 2025. DOI: 10.1038/s41598-025-23920-0.")

    add_card(s10, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.3))
    tb_base = s10.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(11.3), Inches(4.9))
    tf_b = tb_base.text_frame; tf_b.word_wrap = True

    p = tf_b.paragraphs[0]; p.text = "Mathematical Pipeline of the Base Paper (Foundational Reproduction Target)"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_NAVY; p.space_after = Pt(8)
    p2 = tf_b.add_paragraph()
    p2.text = (
        "1. Video Sampling & Preprocessing:\n"
        "   - Samples 16 frames per video; MTCNN face detector localizes primary face.\n"
        "   - Bilinearly resizes crops to 112 x 112 x 3 pixels with standard ImageNet normalization.\n\n"
        "2. Feature Extraction Backbone:\n"
        "   - ResNeXt-50 (32x4d) pretrained on ImageNet.\n"
        "   - Extracts spatial feature maps F in R^(B*T x 2048 x 4 x 4).\n\n"
        "3. Soft Visual Spatial Attention:\n"
        "   - Applies 1x1 convolution across 2048 channels producing spatial weight map S in R^(B*T x 1 x 4 x 4).\n"
        "   - Spatial softmax calculates attention weights: alpha_hw = exp(S_hw) / sum(exp(S)).\n"
        "   - Weighted sum over 4x4 grid yields attended feature vector F_tilde in R^(B*T x 2048).\n\n"
        "4. Temporal Sequence Aggregation:\n"
        "   - Fed into 2-layer Bidirectional LSTM (BiLSTM, hidden size = 1024 per direction -> 2048 output dimensions).\n"
        "   - Temporal mean pooling aggregates across 16 frames; post-pool dropout (p=0.50) mitigates co-adaptation.\n\n"
        "5. Binary Classification: Linear head (2048 -> 1) yields final binary logit z."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 11: Proposed DeepGuard End-to-End Architecture (WITH DIAGRAM)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Proposed DeepGuard End-to-End System Pipeline", "7. PROPOSED SYSTEM ARCHITECTURE")
    add_citation_footer(s11, "Proposed system architecture flowchart; formal specification in DeepGuard_Architecture.md.")

    pipe_img = "reports/figures/system_pipeline_flowchart.png"
    if Path(pipe_img).exists():
        s11.shapes.add_picture(pipe_img, Inches(0.8), Inches(1.4), Inches(11.7), Inches(3.2))

    c_s11 = add_card(s11, Inches(0.8), Inches(4.8), Inches(11.7), Inches(1.9))
    tb_s11 = s11.shapes.add_textbox(Inches(1.0), Inches(4.9), Inches(11.3), Inches(1.7))
    tf_s11 = tb_s11.text_frame; tf_s11.word_wrap = True
    p = tf_s11.paragraphs[0]; p.text = "Proposed Modular Architectural Invariants:"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = C_PRIMARY; p.space_after = Pt(4)
    p2 = tf_s11.add_paragraph()
    p2.text = (
        "• Decoupled Modular Design: The video loader, frame sampler, MTCNN face detector, neural backbones, temporal aggregators, and calibration layers will remain independently replaceable.\n"
        "• Inference Independence: Full video inference will execute without requiring training dependencies, GPU clusters, or external API connectivity.\n"
        "• Commodity GPU Feasibility: Pipeline is engineered to operate strictly within 6.0 GB VRAM on an NVIDIA GeForce RTX 3050 Laptop GPU."
    )
    p2.font.size = Pt(10); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 12: Ingestion & MTCNN Preprocessing Pipeline
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Video Ingestion, Sampling & MTCNN Face Localization", "8. DATA PIPELINE DESIGN")
    add_citation_footer(s12, "Planned data pipeline specification in PRD.md and DeepGuard_SRS.docx.")

    cards12 = [
        ("Deterministic Sampling", 
         "• Uniform Temporal Sampler: Extracts N=16 frames spaced evenly across video duration.\n\n"
         "• Reproducibility: Derives video seed via SHA-256 hash of video bytes for stable inference.\n\n"
         "• Guardrail: Rejects corrupted video containers or clips with < 16 readable frames.", C_NAVY),
        ("MTCNN Face Localization",
         "• Multi-Task Cascaded CNN (P-Net -> R-Net -> O-Net) to detect facial landmarks and bounding boxes.\n\n"
         "• Margin Expansion: Applies 0.20 (20%) bounding box margin expansion to capture jawline and forehead blending seams.\n\n"
         "• Primary Face Selection: Selects largest area bounding box; falls back to center crop if undetected.", C_PRIMARY),
        ("Tensor Transformation",
         "• Normalization: Bilinearly resizes facial crops to 112 x 112 pixels.\n\n"
         "• ImageNet Standardization: Normalized using mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].\n\n"
         "• Tensor Contract: Emits sequence tensor in shape [B, T=16, C=3, H=112, W=112].", C_EMERALD)
    ]
    for i, (ctitle, ctext, ccolor) in enumerate(cards12):
        x = Inches(0.8 + i * 4.0)
        add_card(s12, x, Inches(1.4), Inches(3.7), Inches(5.3))
        tb = s12.shapes.add_textbox(x + Inches(0.2), Inches(1.6), Inches(3.3), Inches(4.9))
        tf = tb.text_frame; tf.word_wrap = True
        p1 = tf.paragraphs[0]; p1.text = ctitle; p1.font.size = Pt(13.5); p1.font.bold = True; p1.font.color.rgb = ccolor; p1.space_after = Pt(8)
        p2 = tf.add_paragraph(); p2.text = ctext; p2.font.size = Pt(10); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 13: Proposed Baseline: ResNeXt-50 + Soft Attention (WITH IMAGE)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Proposed Foundational Baseline: ResNeXt-50 + Soft Attention", "9. BASELINE REPRODUCTION")
    add_citation_footer(s13, "Empirical attention comparison plot from literature analysis in reports/figures/publication/attention_analysis.png.")

    attn_img = "reports/figures/publication/attention_analysis.png"
    if Path(attn_img).exists():
        s13.shapes.add_picture(attn_img, Inches(0.8), Inches(1.4), Inches(11.7), Inches(3.3))

    c_s13 = add_card(s13, Inches(0.8), Inches(4.9), Inches(11.7), Inches(1.8))
    tb_s13 = s13.shapes.add_textbox(Inches(1.0), Inches(5.0), Inches(11.3), Inches(1.6))
    tf_s13 = tb_s13.text_frame; tf_s13.word_wrap = True
    p = tf_s13.paragraphs[0]; p.text = "Role of Visual Spatial Attention in the Proposed Baseline:"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = C_PRIMARY; p.space_after = Pt(4)
    p2 = tf_s13.add_paragraph()
    p2.text = (
        "• Attention vs. Global Average Pooling: Soft spatial attention learns to concentrate gradient updates on manipulated facial subregions (blending boundaries, warped mouth landmarks) rather than uniform background skin.\n"
        "• Planned Reproduction: DeepGuard will implement and train this exact ResNeXt-50 (32x4d) + Soft Attention + BiLSTM model on FaceForensics++ C23 as the project's primary reference benchmark.\n"
        "• Memory Strategy: Initial ResNeXt stages will be frozen, fine-tuning only layer2+ and the attention/LSTM head to fit 6GB GPU constraints."
    )
    p2.font.size = Pt(10); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 14: Proposed Multi-Branch Extension (WITH DIAGRAM)
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Proposed Multi-Branch Invariant Architecture", "10. PROPOSED RESEARCH EXTENSION")
    add_citation_footer(s14, "Proposed multi-branch decomposition diagram from reports/figures/multibranch_architecture.png.")

    mb_img = "reports/figures/multibranch_architecture.png"
    if Path(mb_img).exists():
        s14.shapes.add_picture(mb_img, Inches(0.8), Inches(1.4), Inches(11.7), Inches(3.3))

    c_s14 = add_card(s14, Inches(0.8), Inches(4.9), Inches(11.7), Inches(1.8))
    tb_s14 = s14.shapes.add_textbox(Inches(1.0), Inches(5.0), Inches(11.3), Inches(1.6))
    tf_s14 = tb_s14.text_frame; tf_s14.word_wrap = True
    p = tf_s14.paragraphs[0]; p.text = "Architectural Innovations in the Proposed Multi-Branch Model:"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = C_EMERALD; p.space_after = Pt(4)
    p2 = tf_s14.add_paragraph()
    p2.text = (
        "• Parallel Multi-Domain Representation: Splits backbone features into three complementary streams: (1) Mid-spatial Stage 5 branch, (2) High-frequency residual branch (Laplacian/SRM filter banks), and (3) Texture anomaly branch.\n"
        "• Temporal Transformer Encoder: Replaces sequential LSTMs with 4-head multi-head self-attention across 16 frames to capture long-range inter-frame discrepancies without recurrent bottlenecks.\n"
        "• Explainable Multi-Branch Evidence Head: Emits composite logit z alongside individual branch confidences (p_composite, p_fused, p_frequency, p_texture)."
    )
    p2.font.size = Pt(10); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 15: Proposed Temporal Sequence Modeling: BiLSTM vs. Transformer
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Temporal Sequence Modeling: BiLSTM vs. Temporal Transformer", "11. TEMPORAL MODELING")
    add_citation_footer(s15, "Planned comparative evaluation protocol in DeepGuard_SRS.docx Section 5.")

    c_lstm = add_card(s15, Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tb_l = s15.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(5.2), Inches(4.9))
    tf_l = tb_l.text_frame; tf_l.word_wrap = True
    p = tf_l.paragraphs[0]; p.text = "Candidate 1: Bidirectional LSTM (Base Paper)"; p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_NAVY; p.space_after = Pt(8)
    p2 = tf_l.add_paragraph()
    p2.text = (
        "• Formulation: 2-layer BiLSTM processing sequence [B, 16, 2048] with hidden size = 1024/direction.\n\n"
        "• Strengths: Captures local sequential continuity; standard benchmark adopted by Lal et al. (2025).\n\n"
        "• Weaknesses: Sequential recurrent bottleneck prevents parallel frame computation; sensitive to temporal vanishing gradients.\n\n"
        "• Proposed Role: Foundational baseline temporal model to be reproduced and evaluated."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    c_tr = add_card(s15, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.3))
    tb_tr = s15.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.3), Inches(4.9))
    tf_tr = tb_tr.text_frame; tf_tr.word_wrap = True
    p = tf_tr.paragraphs[0]; p.text = "Candidate 2: Multi-Head Temporal Transformer"; p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_PRIMARY; p.space_after = Pt(8)
    p2 = tf_tr.add_paragraph()
    p2.text = (
        "• Formulation: Transformer encoder with 4 attention heads, positional encoding, and hidden dim = 1280.\n\n"
        "• Strengths: Direct all-to-all frame self-attention; fully parallelizable across GPU cores; tracks subtle inter-frame warping inconsistencies.\n\n"
        "• Efficiency: Lower memory overhead during backpropagation than unrolled recurrent graphs.\n\n"
        "• Proposed Role: Research extension model expected to surpass BiLSTM in cross-family generalization."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 16: Proposed Statistical Calibration & Decision Layer
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "Proposed Statistical Calibration & Decision Policy", "12. CALIBRATION & TRUST")
    add_citation_footer(s16, "Post-hoc temperature scaling and OOD gating design in PRD.md Section 6.")

    c_cal = add_card(s16, Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tb_cal = s16.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(5.2), Inches(4.9))
    tf_cal = tb_cal.text_frame; tf_cal.word_wrap = True
    p = tf_cal.paragraphs[0]; p.text = "Post-Hoc Temperature Scaling"; p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_EMERALD; p.space_after = Pt(8)
    p2 = tf_cal.add_paragraph()
    p2.text = (
        "• The Problem: Standard deep learning classifiers output overconfident probabilities (e.g. 0.95 confidence with 65% empirical accuracy), producing severe false alarm rates.\n\n"
        "• Proposed Calibration Formulation:\n"
        "  p_cal = sigmoid(z / T)\n\n"
        "• Strict Governance Rule: Scalar temperature T will be fitted strictly on held-out validation sequences by minimizing negative log-likelihood (NLL). Test sets will never be used for calibration.\n\n"
        "• Target Metric: Expected Calibration Error (ECE) target <= 10.0%."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    c_ood = add_card(s16, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.3))
    tb_ood = s16.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.3), Inches(4.9))
    tf_ood = tb_ood.text_frame; tf_ood.word_wrap = True
    p = tf_ood.paragraphs[0]; p.text = "OOD Gating & Dual Decision Policies"; p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_PRIMARY; p.space_after = Pt(8)
    p2 = tf_ood.add_paragraph()
    p2.text = (
        "• Out-of-Distribution (OOD) Gating: Computes regularized Mahalanobis / Euclidean distance of penultimate features against an empirical validation manifold threshold (3.18 at 99th percentile).\n\n"
        "• Dual Operating Threshold Policies:\n"
        "  1. Standard Balanced Policy (tau = 0.50): Equal weighting of precision and recall.\n"
        "  2. High-Sensitivity Policy (tau*): Operating threshold calibrated on validation set to strictly guarantee >= 90.0% specificity on clean authentic media.\n\n"
        "• Explainability: Emits OOD warning flags when inputs exhibit distribution shift."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 17: Proposed Benchmark Datasets
    # =========================================================================
    s17 = prs.slides.add_slide(blank_layout)
    add_header(s17, "Proposed Evaluation Datasets & Benchmark Roles", "13. DATASET PROVENANCE")
    add_citation_footer(s17, "Dataset provenance and access requirements in DeepGuard_Test_Plan_and_Data_Documentation.docx.")

    tbl_s17 = s17.shapes.add_table(4, 5, Inches(0.8), Inches(1.4), Inches(11.7), Inches(3.0)).table
    tbl_s17.columns[0].width = Inches(2.5)
    tbl_s17.columns[1].width = Inches(1.8)
    tbl_s17.columns[2].width = Inches(2.2)
    tbl_s17.columns[3].width = Inches(2.7)
    tbl_s17.columns[4].width = Inches(2.5)

    d_headers17 = ["Dataset Name", "Compression Tier", "Sample Volume", "Proposed Evaluation Role", "Access / Provenance"]
    d_rows17 = [
        d_headers17,
        ("FaceForensics++ (FF++)", "C23 (HQ, H.264)", "7,000 videos (16.69 GB)\n1,000 real + 4,000 fake", "Primary in-domain training, validation & 4-fold Leave-One-Family-Out matrix", "TUM Computer Vision Group (Academic request required)"),
        ("Celeb-DF v2", "Standard YouTube", "6,529 videos total\n518 official test videos", "Independent zero-shot cross-dataset generalization benchmark", "Celeb-DF Project (Research access required)"),
        ("DFDC (Challenge)", "Kaggle / Meta", "~100,000 videos (~470 GB)", "Independent cross-dataset benchmark (Subject to Kaggle credentials)", "Kaggle Deepfake Detection Challenge")
    ]
    for r_i, r_data in enumerate(d_rows17):
        for c_i, val in enumerate(r_data):
            cell = tbl_s17.cell(r_i, c_i)
            cell.text = val
            p = cell.text_frame.paragraphs[0]; p.font.name = "Calibri"
            if r_i == 0:
                p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                cell.fill.solid(); cell.fill.fore_color.rgb = C_NAVY
            else:
                p.font.size = Pt(9.5); p.font.color.rgb = C_TEXT_DARK
                cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC) if r_i % 2 == 1 else RGBColor(0xFF, 0xFF, 0xFF)

    c_box17 = add_card(s17, Inches(0.8), Inches(4.7), Inches(11.7), Inches(2.0), RGBColor(0xF0, 0xFD, 0xFA), C_EMERALD)
    tb_b17 = s17.shapes.add_textbox(Inches(1.0), Inches(4.8), Inches(11.3), Inches(1.8))
    tf_b17 = tb_b17.text_frame; tf_b17.word_wrap = True
    p = tf_b17.paragraphs[0]; p.text = "Benchmark Rationale & Multi-Dataset Strategy:"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = C_EMERALD; p.space_after = Pt(4)
    p2 = tf_b17.add_paragraph()
    p2.text = (
        "• In-Domain vs. Cross-Dataset Evaluation: FaceForensics++ C23 provides a controlled benchmark covering 4 manipulation families under standardized compression. "
        "However, evaluating solely on FF++ introduces overfitting. "
        "Celeb-DF v2 and DFDC will serve as independent, zero-shot holdouts to measure real-world generalization across unseen lighting, poses, and blending algorithms without retraining."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 18: Anti-Leakage Data Partitioning Strategy
    # =========================================================================
    s18 = prs.slides.add_slide(blank_layout)
    add_header(s18, "Anti-Leakage Data Strategy & Indivisible Group Partitioning", "14. RESEARCH INTEGRITY")
    add_citation_footer(s18, "Strict anti-leakage invariant protocol defined in PRD.md Section 9.")

    add_card(s18, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.3))
    tb_al = s18.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(11.3), Inches(4.9))
    tf_al = tb_al.text_frame; tf_al.word_wrap = True

    p = tf_al.paragraphs[0]; p.text = "The Data Leakage Vulnerability in Deepfake Research"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_ROSE; p.space_after = Pt(8)
    p2 = tf_al.add_paragraph()
    p2.text = (
        "• The Flawed Naive Split: In FaceForensics++, 1,000 real videos are manipulated into 4,000 fakes. A random filename-based split risks placing the pristine video of actor '004' in the training set and the manipulated video of actor '004' in the test set. "
        "The neural network memorizes facial landmarks and actor identity rather than learning synthesis boundaries, artificially inflating test accuracy to 99%.\n\n"
        "• Proposed Indivisible Group Partitioning Invariant:\n"
        "  1. Assign an indivisible group_id to every pristine video and all 4 of its manipulated counterparts.\n"
        "  2. Partition dataset strictly at the group level: 720 train subjects (5,040 videos), 140 val subjects (980 videos), and 140 test subjects (980 videos).\n"
        "  3. Mathematical guarantee: Zero group overlap between training and testing splits.\n\n"
        "• Planned 5-Fold Cross-Validation: The proposed LeakageSafeGroupKFold partitioner will apply this strict group isolation across 5 balanced folds to satisfy academic verification standards."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 19: Hardware Constraints & Optimization on RTX 3050 (6GB)
    # =========================================================================
    s19 = prs.slides.add_slide(blank_layout)
    add_header(s19, "Hardware Constraints & Commodity GPU Optimization", "15. COMMODITY GPU FEASIBILITY")
    add_citation_footer(s19, "Hardware envelope specifications in PRD.md and DeepGuard_SRS.docx.")

    hw_cards19 = [
        ("VRAM Target (< 1.0 GB)", "• Hardware Target: NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM).\n\n• Target Peak Allocation: Constrain peak GPU VRAM to < 1.0 GB during 16-frame video inference.\n\n• Benefit: Allows deepfake detection to execute locally on consumer workstations without cloud GPU servers.", C_EMERALD),
        ("Inference Latency Target", "• Forward Pass Target: P50 latency <= 35.0 ms per 16-frame sequence.\n\n• Throughput Target: > 450 frames/second on GPU forward pass.\n\n• End-to-End Pipeline Target: Total processing time <= 3.5 seconds per video (including MTCNN face extraction).", C_PRIMARY),
        ("Training Optimization Plan", "• Mixed Precision: PyTorch FP16 / AMP autocast will reduce activation memory by 45%.\n\n• Micro-Batching: Batch size 4 with 2-step gradient accumulation -> effective batch size 8.\n\n• Cumulative Freezing: Initial backbone stages will remain frozen during training.", C_NAVY)
    ]
    for i, (ctitle, ctext, ccolor) in enumerate(hw_cards19):
        x = Inches(0.8 + i * 4.0)
        add_card(s19, x, Inches(1.4), Inches(3.7), Inches(5.3))
        tb = s19.shapes.add_textbox(x + Inches(0.2), Inches(1.6), Inches(3.3), Inches(4.9))
        tf = tb.text_frame; tf.word_wrap = True
        p1 = tf.paragraphs[0]; p1.text = ctitle; p1.font.size = Pt(13.5); p1.font.bold = True; p1.font.color.rgb = ccolor; p1.space_after = Pt(8)
        p2 = tf.add_paragraph(); p2.text = ctext; p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 20: Planned Evaluation Metrics & Target Success Criteria
    # =========================================================================
    s20 = prs.slides.add_slide(blank_layout)
    add_header(s20, "Planned Evaluation Metrics & Target Success Criteria", "16. EVALUATION TARGETS")
    add_citation_footer(s20, "Target criteria defined in DeepGuard_Test_Plan_and_Data_Documentation.docx.")

    tbl_s20 = s20.shapes.add_table(7, 4, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.3)).table
    tbl_s20.columns[0].width = Inches(2.2)
    tbl_s20.columns[1].width = Inches(3.8)
    tbl_s20.columns[2].width = Inches(2.8)
    tbl_s20.columns[3].width = Inches(2.9)

    m_headers20 = ["Evaluation Metric", "Mathematical Formulation", "Proposed Target Criterion", "Research Justification"]
    m_data20 = [
        m_headers20,
        ("ROC-AUC", "Area Under the Receiver Operating Characteristic Curve", ">= 0.90 (In-Domain FF++)\n>= 0.75 (Cross-Dataset)", "Primary threshold-independent ranking metric for binary classification."),
        ("Classification Accuracy", "(TP + TN) / Total Video Sequences", ">= 90.0% (FF++ C23)", "Standard overall correctness indicator across balanced test partitions."),
        ("Clean Specificity", "TN / (TN + FP)", ">= 90.0% (Authentic Media)", "Guarantees authentic user media is not erroneously flagged as deepfake."),
        ("Equal Error Rate (EER)", "Operating point where FPR == FNR", "<= 10.0% (In-Domain)", "Standard biometrics benchmark assessing false accept vs false reject balance."),
        ("Expected Calibration Error", "Weighted difference: sum(|conf(B) - acc(B)|)", "<= 10.0% (Post-Calibration)", "Ensures model confidence reflects empirical probability of correctness."),
        ("Forward Pass Latency", "P50 execution time for 16 frames", "<= 35.0 ms (RTX 3050 GPU)", "Guarantees practical usability for near-real-time video analysis.")
    ]
    for r_i, r_data in enumerate(m_data20):
        for c_i, val in enumerate(r_data):
            cell = tbl_s20.cell(r_i, c_i)
            cell.text = val
            p = cell.text_frame.paragraphs[0]; p.font.name = "Calibri"
            if r_i == 0:
                p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                cell.fill.solid(); cell.fill.fore_color.rgb = C_NAVY
            else:
                p.font.size = Pt(9.5); p.font.color.rgb = C_TEXT_DARK
                cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC) if r_i % 2 == 1 else RGBColor(0xFF, 0xFF, 0xFF)

    # =========================================================================
    # SLIDE 21: Planned Stress Testing: Perturbation Robustness (WITH IMAGE)
    # =========================================================================
    s21 = prs.slides.add_slide(blank_layout)
    add_header(s21, "Planned Stress Testing: Perturbation Robustness", "17. ROBUSTNESS & STRESS TESTING")
    add_citation_footer(s21, "Empirical robustness curves from literature analysis in reports/figures/publication/robustness.png.")

    rob_img = "reports/figures/publication/robustness.png"
    if Path(rob_img).exists():
        s21.shapes.add_picture(rob_img, Inches(0.8), Inches(1.4), Inches(11.7), Inches(3.4))

    c_s21 = add_card(s21, Inches(0.8), Inches(5.0), Inches(11.7), Inches(1.7))
    tb_s21 = s21.shapes.add_textbox(Inches(1.0), Inches(5.1), Inches(11.3), Inches(1.5))
    tf_s21 = tb_s21.text_frame; tf_s21.word_wrap = True
    p = tf_s21.paragraphs[0]; p.text = "Planned Real-World Degradation Stress Tests:"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = C_PRIMARY; p.space_after = Pt(4)
    p2 = tf_s21.add_paragraph()
    p2.text = (
        "• Lossy Compression Testing: DeepGuard will evaluate model degradation under H.264 C40 heavy compression and JPEG quality factors Q=50, 25, 10.\n"
        "• Spatial Perturbations: Stress tests will evaluate model performance under Gaussian blurring (kernel sizes 3 to 7) and spatial downsampling (1080p -> 720p -> 480p).\n"
        "• Sensor & Additive Noise: Models will be evaluated against additive Gaussian noise up to sigma=0.05 to measure noise immunity."
    )
    p2.font.size = Pt(10); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 22: Explainable Attribution & Visual Evidence Concept (WITH 4 IMAGES)
    # =========================================================================
    s22 = prs.slides.add_slide(blank_layout)
    add_header(s22, "Explainable Attribution & Forensic Heatmap Concept", "18. EXPLAINABILITY & VISUAL EVIDENCE")
    add_citation_footer(s22, "Illustrative forensic attribution maps from reports/figures/attribution/.")

    attributions22 = [
        ("True Positive Concept (Fake Detected)", "Attention expected to focus sharply on mouth boundary warping and blending seams.", "reports/figures/attribution/true_positive_fake_attribution.png", C_EMERALD),
        ("True Negative Concept (Pristine Real)", "Expected uniform, low-intensity feature activation across all facial regions.", "reports/figures/attribution/true_negative_real_attribution.png", C_PRIMARY),
        ("False Positive Analysis (Clean Flagged)", "Investigation of how high-frequency compression artifacts trigger false alarms.", "reports/figures/attribution/false_positive_case_attribution.png", C_AMBER),
        ("False Negative Analysis (Reenactment)", "Investigation of dispersed attention on Face2Face expression manipulations.", "reports/figures/attribution/false_negative_case_attribution.png", C_ROSE)
    ]
    for i, (atitle, adesc, aimg, acolor) in enumerate(attributions22):
        x = Inches(0.8 + i * 3.0)
        card = add_card(s22, x, Inches(1.4), Inches(2.8), Inches(5.3))
        tb = s22.shapes.add_textbox(x + Inches(0.1), Inches(1.5), Inches(2.6), Inches(0.8))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = atitle; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = acolor

        if Path(aimg).exists():
            s22.shapes.add_picture(aimg, x + Inches(0.4), Inches(2.4), Inches(2.0), Inches(2.6))

        tb2 = s22.shapes.add_textbox(x + Inches(0.1), Inches(5.1), Inches(2.6), Inches(1.5))
        tf2 = tb2.text_frame; tf2.word_wrap = True
        p2 = tf2.paragraphs[0]; p2.text = adesc; p2.font.size = Pt(9); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 23: Proposed Software Stack, REST API & Dashboard
    # =========================================================================
    s23 = prs.slides.add_slide(blank_layout)
    add_header(s23, "Proposed Software Stack, REST API & Web Dashboard", "19. SOFTWARE ENGINEERING DESIGN")
    add_citation_footer(s23, "Planned software stack specification in DeepGuard_SRS.docx Section 7.")

    cards23 = [
        ("FastAPI Microservice Architecture",
         "• Predictor Pool: Dynamic loader supporting baseline ResNeXt and multi-branch models.\n\n"
         "• Planned Endpoints:\n"
         "  - POST /api/v1/predict: Multipart video upload and calibrated verdict generation.\n"
         "  - POST /api/v1/compare: Side-by-side execution comparing baseline and proposed model.\n"
         "  - GET /api/v1/model-info: Checkpoint parameters and limitation disclosures.\n"
         "  - GET /health: Active GPU allocation and memory status.", C_NAVY),
        ("Real-Time Web Forensics Dashboard",
         "• Dark-Theme Interface: Glassmorphic dark styling adhering to academic UI guidelines.\n\n"
         "• Interactive Threshold Slider: Allows forensic analysts to dynamically toggle between standard tau=0.50 and high-sensitivity tau*.\n\n"
         "• Latency Waterfall Display: Planned breakdown showing execution time across Sampling -> MTCNN -> Model Forward -> Reporting.\n\n"
         "• Lightweight Frontend: Pure HTML5/ES6/CSS with zero npm runtime dependencies.", C_PRIMARY)
    ]
    for i, (ctitle, ctext, ccolor) in enumerate(cards23):
        x = Inches(0.8 + i * 6.0)
        add_card(s23, x, Inches(1.4), Inches(5.7), Inches(5.3))
        tb = s23.shapes.add_textbox(x + Inches(0.2), Inches(1.6), Inches(5.3), Inches(4.9))
        tf = tb.text_frame; tf.word_wrap = True
        p1 = tf.paragraphs[0]; p1.text = ctitle; p1.font.size = Pt(13.5); p1.font.bold = True; p1.font.color.rgb = ccolor; p1.space_after = Pt(8)
        p2 = tf.add_paragraph(); p2.text = ctext; p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 24: Planned Project Activity Timeline (16-Week Schedule)
    # =========================================================================
    s24 = prs.slides.add_slide(blank_layout)
    add_header(s24, "Planned Project Activity Timeline & Work Breakdown", "20. PROJECT TIMELINE & PHASES")
    add_citation_footer(s24, "16-Week implementation work plan in DeepGuard_SRS.docx Section 9.")

    tbl_s24 = s24.shapes.add_table(8, 3, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.3)).table
    tbl_s24.columns[0].width = Inches(2.2)
    tbl_s24.columns[1].width = Inches(2.0)
    tbl_s24.columns[2].width = Inches(7.5)

    gantt_data = [
        ("Project Phase", "Scheduled Duration", "Key Planned Activities & Milestone Objectives"),
        ("Phase 1: Requirements & Survey", "Weeks 1 – 2", "Formalize SRS; conduct 18-paper literature survey; analyze Lal et al. (2025) base paper."),
        ("Phase 2: Dataset Acquisition & Pipeline", "Weeks 3 – 4", "Obtain FaceForensics++ C23 access; build OpenCV decoding, uniform sampling, and MTCNN pipeline."),
        ("Phase 3: Baseline Reproduction", "Weeks 5 – 7", "Implement ResNeXt-50 + Soft Attention + BiLSTM; train on FF++ C23 under mixed precision."),
        ("Phase 4: Multi-Branch Invariant Design", "Weeks 8 – 10", "Implement mid-spatial, frequency residual, and texture branches; train with Temporal Transformer."),
        ("Phase 5: Calibration & Generalization", "Weeks 11 – 12", "Fit post-hoc temperature scaling; implement OOD gating; evaluate cross-dataset transfer on Celeb-DF v2."),
        ("Phase 6: Serving & Web Dashboard", "Weeks 13 – 14", "Build FastAPI microservice; develop interactive dark-mode web dashboard; implement ONNX export."),
        ("Phase 7: Testing & Final Submission", "Weeks 15 – 16", "Execute comprehensive test suite; finalize design reports, presentation deck, and project defense.")
    ]
    for r_i, r_data in enumerate(gantt_data):
        for c_i, val in enumerate(r_data):
            cell = tbl_s24.cell(r_i, c_i)
            cell.text = val
            p = cell.text_frame.paragraphs[0]; p.font.name = "Calibri"
            if r_i == 0:
                p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                cell.fill.solid(); cell.fill.fore_color.rgb = C_NAVY
            else:
                p.font.size = Pt(9.5); p.font.color.rgb = C_TEXT_DARK
                cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC) if r_i % 2 == 1 else RGBColor(0xFF, 0xFF, 0xFF)

    # =========================================================================
    # SLIDE 25: Summary of Proposed Contributions
    # =========================================================================
    s25 = prs.slides.add_slide(blank_layout)
    add_header(s25, "Summary of Proposed Research Contributions", "21. PROPOSED IMPACT")
    add_citation_footer(s25, "Summary of expected deliverables and academic contributions.")

    c_p1 = add_card(s25, Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tb_p1 = s25.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(5.2), Inches(4.9))
    tf_p1 = tb_p1.text_frame; tf_p1.word_wrap = True
    p = tf_p1.paragraphs[0]; p.text = "Expected Scientific Contributions"; p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_NAVY; p.space_after = Pt(8)
    p2 = tf_p1.add_paragraph()
    p2.text = (
        "• Methodological Reproduction: Transparent reproduction of the base paper (ResNeXt-50 + Soft Attention + BiLSTM) on standardized benchmarks.\n\n"
        "• Overcoming Within-Manifold Forgeries: Proposed multi-branch architecture integrating frequency and texture anomaly streams to detect seamless expression manipulations (Face2Face).\n\n"
        "• Calibrated Forensic Trust: Introducing post-hoc temperature scaling and penultimate Mahalanobis distance gating to replace uncalibrated overconfident scores.\n\n"
        "• Reproducibility Standards: Anti-leakage indivisible group partitioning and leakage-safe 5-fold cross-validation."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    c_p2 = add_card(s25, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.3))
    tb_p2 = s25.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.3), Inches(4.9))
    tf_p2 = tb_p2.text_frame; tf_p2.word_wrap = True
    p = tf_p2.paragraphs[0]; p.text = "Expected Engineering Deliverables"; p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_PRIMARY; p.space_after = Pt(8)
    p2 = tf_p2.add_paragraph()
    p2.text = (
        "• Commodity GPU Execution: Functional pipeline executing inside 6.0 GB VRAM on NVIDIA RTX 3050 (<1.0 GB peak VRAM, <= 35 ms latency).\n\n"
        "• Production FastAPI Microservice: High-performance asynchronous REST API supporting video uploads and multi-engine comparison.\n\n"
        "• Real-Time Web Forensics Dashboard: Dependency-free glassmorphic dark dashboard with interactive threshold slider and latency waterfall.\n\n"
        "• Automated Regression Test Suite: Comprehensive test suite guaranteeing zero data leakage, shape contracts, and numerical parity."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 26: Scholarly References (IEEE Style) & Conclusion
    # =========================================================================
    s26 = prs.slides.add_slide(blank_layout)
    add_header(s26, "Scholarly References (IEEE Style) & Conclusion", "22. REFERENCES & CONCLUSION")
    add_citation_footer(s26, "Complete 18-paper literature survey in submission/Literature_Survey/DeepGuard_Literature_Survey.xlsx.")

    tb_ref26 = s26.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.3))
    tf_r26 = tb_ref26.text_frame; tf_r26.word_wrap = True

    refs26 = [
        "[1] K. Lal, S. Shiwani, and G. C. Gandhi, 'Deepfake video deception detection using visual attention-based method,' Scientific Reports, vol. 15, no. 1, art. 40089, 2025. DOI: 10.1038/s41598-025-23920-0.",
        "[2] A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner, 'FaceForensics++: Learning to detect manipulated facial images,' in Proc. IEEE/CVF ICCV, 2019, pp. 1–11.",
        "[3] Y. Qian, G. Yin, L. Sheng, Z. Chen, and J. Shao, 'Thinking in frequency: Face forgery detection by mining frequency-aware clues,' in Proc. ECCV, 2020, pp. 86–103.",
        "[4] H. Liu et al., 'Spatial-phase shallow learning: Rethinking face forgery detection in frequency domain,' in Proc. IEEE/CVF CVPR, 2021, pp. 772–781.",
        "[5] A. Ismail, M. Elpeltagy, M. S. Zaki, and K. A. ElDahshan, 'Deepfake video detection: YOLO-Face and convolutional LSTM based approach,' PeerJ Comput. Sci., vol. 7, p. e730, 2021.",
        "[6] D. Wodajo, H. Mareen, and P. Lambert, 'Deepfake video detection using generative convolutional vision transformer,' Appl. Sci., vol. 15, no. 12, p. 6622, 2023.",
        "[7] L. Li et al., 'Face X-ray for more general face forgery detection,' in Proc. IEEE/CVF CVPR, 2020, pp. 5001–5010.",
        "[8] R. Tolosana, S. Romero-Tapiado, J. Fierrez, and A. Morales, 'DeepFakes evolution: Analysis of facial manipulation and detection methods,' Inf. Fusion, vol. 64, pp. 131–148, 2020."
    ]
    p0 = tf_r26.paragraphs[0]
    p0.text = "Key Foundational Literature:"; p0.font.size = Pt(12); p0.font.bold = True; p0.font.color.rgb = C_NAVY; p0.space_after = Pt(6)
    for r_text in refs26:
        p = tf_r26.add_paragraph()
        p.text = r_text; p.font.size = Pt(9.5); p.font.color.rgb = C_TEXT_DARK; p.space_after = Pt(4)

    p_end = tf_r26.add_paragraph()
    p_end.text = "\nThank you! Questions and feedback are welcomed."
    p_end.font.size = Pt(13); p_end.font.bold = True; p_end.font.color.rgb = C_PRIMARY

    # Save presentation
    output_pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output_pptx)
    print(f"Generated Proposal Presentation PPTX: {output_pptx}")

if __name__ == "__main__":
    out_pptx = Path("submission/PPT/DeepGuard_Project_Presentation.pptx")
    create_proposal_presentation(out_pptx)
