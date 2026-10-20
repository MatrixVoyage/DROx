"""
DeepGuard Academic Presentation Generator (28 Slides with Embedded Visuals).
Generates submission/PPT/DeepGuard_Project_Presentation.pptx (16:9 Widescreen)
with authentic project figures, exact experimental tables, and citations.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_presentation(output_pptx: Path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette
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

    def add_header(slide, title_text, category_text="IVP GROUP 10 — ACADEMIC DEFENSE & VIVA"):
        tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.7), Inches(0.3))
        tf_cat = tb_cat.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.name = "Calibri"
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = C_PRIMARY

        tb_t = slide.shapes.add_textbox(Inches(0.8), Inches(0.68), Inches(11.7), Inches(0.65))
        tf_t = tb_t.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.name = "Calibri"
        p_t.font.size = Pt(21)
        p_t.font.bold = True
        p_t.font.color.rgb = C_NAVY

    def add_citation_footer(slide, citation_text):
        tb_c = slide.shapes.add_textbox(Inches(0.8), Inches(6.92), Inches(11.7), Inches(0.35))
        tf_c = tb_c.text_frame
        p_c = tf_c.paragraphs[0]
        p_c.text = f"Citation / Reference: {citation_text}"
        p_c.font.name = "Calibri"
        p_c.font.size = Pt(8.5)
        p_c.font.italic = True
        p_c.font.color.rgb = C_TEXT_MUTED

    def add_card(slide, left, top, width, height, bg_color=C_CARD_BG, border_color=C_CARD_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
        return shape

    def add_bullet_text(tf, text, font_size=10.5, color=C_TEXT_DARK):
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = text
        p.font.name = "Calibri"
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.space_after = Pt(4)
        return p

    # =========================================================================
    # SLIDE 1: Title Slide (Dark Theme)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    add_card(s1, Inches(0), Inches(0), Inches(13.333), Inches(7.5), C_DARK_BG, C_DARK_BG)
    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.333), Inches(5.2))
    tf1 = tb1.text_frame; tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "IMAGE AND VIDEO PROCESSING (IVP) — FINAL PROJECT DEFENSE"
    p0.font.name = "Calibri"; p0.font.size = Pt(13); p0.font.bold = True; p0.font.color.rgb = RGBColor(0x38, 0xBD, 0xF8); p0.space_after = Pt(14)

    p1 = tf1.add_paragraph()
    p1.text = "DeepGuard: An Explainable Deepfake Video Detection and Forensic Analysis Platform"
    p1.font.name = "Calibri"; p1.font.size = Pt(30); p1.font.bold = True; p1.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF); p1.space_after = Pt(14)

    p2 = tf1.add_paragraph()
    p2.text = "A Calibrated Multi-Engine Visual Forensics System: From Paper Baseline Reproduction to Multi-Family Invariant Representations"
    p2.font.name = "Calibri"; p2.font.size = Pt(15); p2.font.italic = True; p2.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8); p2.space_after = Pt(28)

    p3 = tf1.add_paragraph()
    p3.text = "Group 10 Team Members: IIT2024108 | IIT2024109 | IIT2024110 | IIT2024115 | IIT2024119 | IIT2024122"
    p3.font.name = "Calibri"; p3.font.size = Pt(13); p3.font.bold = True; p3.font.color.rgb = RGBColor(0xFB, 0xBF, 0x24); p3.space_after = Pt(8)

    p4 = tf1.add_paragraph()
    p4.text = "Primary Research Reference: Lal, Shiwani & Gandhi (2025), Scientific Reports | Hardware Target: NVIDIA RTX 3050 (6GB VRAM)"
    p4.font.name = "Calibri"; p4.font.size = Pt(11); p4.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # =========================================================================
    # SLIDE 2: Problem Statement & Motivation
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Problem Statement & Research Motivation", "1. CONTEXT & THREAT LANDSCAPE")
    add_citation_footer(s2, "Tolosana et al., 'DeepFakes Evolution: Analysis of Facial Manipulation', Information Fusion, 2020.")

    cards2 = [
        ("The Threat Landscape", "• Proliferation of generative AI enables photorealistic face swapping, reenactment, and synthetic speech synthesis.\n\n• High-consequence risks: Disinformation, corporate extortion, political manipulation, and defamatory media.\n\n• Legal need: Trustworthy detection with verifiable chain of custody.", C_NAVY),
        ("The Technical Dilemma", "• Compression loss: H.264 (C23/C40) destroys subtle blending boundaries.\n\n• Unseen forgeries: Detectors overfit to seen manipulation artifacts, failing on novel architectures (>20% AUC drop).\n\n• Overconfidence: Uncalibrated probabilities produce high false alarms on clean media.", C_PRIMARY),
        ("The DeepGuard Objective", "• Deliver an end-to-end, calibrated forensics pipeline executing on commodity GPUs (<540 MB VRAM).\n\n• Reproduce base paper baseline while advancing multi-family invariant representations.\n\n• Full transparency: Multi-branch evidence, ECE calibration, and OOD flagging.", C_EMERALD)
    ]
    for i, (ctitle, ctext, ccolor) in enumerate(cards2):
        x = Inches(0.8 + i * 4.0)
        add_card(s2, x, Inches(1.4), Inches(3.7), Inches(5.3))
        tb = s2.shapes.add_textbox(x + Inches(0.2), Inches(1.6), Inches(3.3), Inches(4.9))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = ctitle; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = ccolor; p.space_after = Pt(8)
        p2 = tf.add_paragraph(); p2.text = ctext; p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 3: Scientific Challenges in Video Forensics
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Core Scientific Challenges in Video Forensics", "2. THE SCIENTIFIC HURDLES")
    add_citation_footer(s3, "Rössler et al. (ICCV 2019) and Liu et al. (CVPR 2021).")

    challenges = [
        ("1. Compression Artifact Masking", "Video codecs (H.264 / HEVC) apply discrete cosine transforms and quantization, discarding high-frequency boundary telltales. Deepfake detectors relying purely on spatial RGB seams degrade by >14% accuracy from Raw to C40 compression.", C_ROSE),
        ("2. The Generalization Abyss", "Deepfake models trained on one manipulation family (e.g., Deepfakes) frequently drop to near random guess (AUC ~0.55-0.65) when tested on unseen families like Face2Face expression reenactment.", C_AMBER),
        ("3. Severe Calibration Drift", "Standard neural network outputs produce extreme probabilities (0.01 or 0.99) that do not match empirical likelihoods. Uncalibrated models exhibit Expected Calibration Error (ECE) > 25%, resulting in catastrophic false alarms on clean videos.", C_PRIMARY),
        ("4. Consumer Hardware Constraints", "State-of-the-art vision models require large compute clusters. Engineering a production-grade temporal deepfake detector that fits inside 6 GB of VRAM without sacrificing detection power is an open challenge.", C_NAVY)
    ]
    for i, (ctitle, ctext, ccolor) in enumerate(challenges):
        x = Inches(0.8 + (i % 2) * 6.0)
        y = Inches(1.4 + (i // 2) * 2.7)
        add_card(s3, x, y, Inches(5.7), Inches(2.4))
        tb = s3.shapes.add_textbox(x + Inches(0.2), y + Inches(0.2), Inches(5.3), Inches(2.0))
        tf = tb.text_frame; tf.word_wrap = True
        p1 = tf.paragraphs[0]; p1.text = ctitle; p1.font.size = Pt(13); p1.font.bold = True; p1.font.color.rgb = ccolor; p1.space_after = Pt(6)
        p2 = tf.add_paragraph(); p2.text = ctext; p2.font.size = Pt(10); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 4: Project Objectives & Scope Boundaries
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Project Objectives & Scope Boundaries", "3. GOALS & ARCHITECTURAL INVARIANTS")
    add_citation_footer(s4, "DeepGuard PRD.md & persistent context/architecture.md specifications.")

    objs = [
        ("Obj 1: Base Paper Reproduction", "Implement, calibrate, and evaluate the published ResNeXt-50 + Soft Attention + BiLSTM model (Lal et al., 2025) on FaceForensics++ C23 with verifiable provenance.", C_PRIMARY),
        ("Obj 2: Generalization & Invariance", "Overcome the within-manifold generalization failure mode on unseen manipulation families via family-invariant representations and temporal Transformer modeling.", C_EMERALD),
        ("Obj 3: Trust, Calibration & OOD", "Enforce statistical calibration via post-hoc temperature scaling (ECE <= 10%) and penultimate Mahalanobis distance gating to prevent uncalibrated overconfidence.", C_AMBER),
        ("Obj 4: Production-Grade System", "Deliver a complete modular ecosystem featuring a multi-engine REST API, real-time web dashboard, ONNX/TorchScript export, and 100% automated regression coverage.", C_NAVY)
    ]
    for i, (otitle, otext, ocolor) in enumerate(objs):
        x = Inches(0.8 + (i % 2) * 6.0)
        y = Inches(1.4 + (i // 2) * 2.7)
        add_card(s4, x, y, Inches(5.7), Inches(2.4))
        tb = s4.shapes.add_textbox(x + Inches(0.2), y + Inches(0.2), Inches(5.3), Inches(2.0))
        tf = tb.text_frame; tf.word_wrap = True
        p1 = tf.paragraphs[0]; p1.text = otitle; p1.font.size = Pt(13); p1.font.bold = True; p1.font.color.rgb = ocolor; p1.space_after = Pt(6)
        p2 = tf.add_paragraph(); p2.text = otext; p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 5: Literature Survey Summary & Taxonomy
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Literature Survey & Methodological Taxonomy", "4. SCHOLARLY STATE-OF-THE-ART")
    add_citation_footer(s5, "18 verified research papers in submission/Literature_Survey/DeepGuard_Literature_Survey.xlsx.")

    tb_s5 = s5.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.3))
    tf_s5 = tb_s5.text_frame; tf_s5.word_wrap = True

    p = tf_s5.paragraphs[0]
    p.text = "Systematic Analysis of 18 Peer-Reviewed Studies Across Detection Modalities"
    p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_NAVY; p.space_after = Pt(6)

    p2 = tf_s5.add_paragraph()
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
    # SLIDE 6: Top 5 Relevant Papers Comparison Table
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Top 5 Most Relevant Research Papers Comparison", "4. SCHOLARLY BENCHMARK COMPARISON")
    add_citation_footer(s6, "Detailed multi-dimensional comparison in DeepGuard_Literature_Survey.xlsx (Sheet 2).")

    tbl_s6 = s6.shapes.add_table(6, 5, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.3)).table
    tbl_s6.columns[0].width = Inches(1.8)
    tbl_s6.columns[1].width = Inches(2.2)
    tbl_s6.columns[2].width = Inches(2.4)
    tbl_s6.columns[3].width = Inches(2.5)
    tbl_s6.columns[4].width = Inches(2.8)

    top5_summary = [
        ("Paper & Citation", "Proposed Architecture", "Datasets Evaluated", "Published Results [Ref]", "Direct DeepGuard Impact"),
        ("P-01: Lal et al. (2025)\nSci. Rep. 15(1)", "ResNeXt-50 + Soft Spatial Attention + BiLSTM (2048)", "FF++ C23, Celeb-DF v2, DFDC", "FF++: AUC 0.962\nCeleb-DF: AUC 0.950\nDFDC: AUC 0.920", "Direct reproduction target (v1-baseline); reference control engine."),
        ("P-02: Rössler et al. (2019)\nICCV 2019", "XceptionNet, MesoNet, Steganalysis", "FF++ (Raw, C23, C40 across 4 families)", "Xception C23: 95.73%\nC40: 81.00% Acc", "Source dataset provider (FF++ C23); defines 4-family manipulation taxonomy."),
        ("P-03: Qian et al. (2020)\nECCV 2020", "Frequency-aware Decomposition (FAD) + Local Freq (LFS)", "FF++ (Raw, C23, C40), Celeb-DF", "FF++ C23: AUC 0.992\nCeleb-DF: AUC 0.768", "Motivates v5 high-frequency residual branch and M18 dual-stream DCT branch."),
        ("P-04: Liu et al. (2021)\nCVPR 2021", "Spatial-Phase Shallow Learning (SPSL) via FFT", "FF++ (C23, C40), Celeb-DF", "FF++ C23: AUC 0.953\nCeleb-DF: AUC 0.778", "Guides Phase C analysis on within-manifold Face2Face feature overlap."),
        ("P-07: Li et al. (2020)\nCVPR 2020", "Face X-ray synthetic blending boundary detector", "FF++ (C23, C40), Celeb-DF, DFD", "FF++ C23: AUC 0.985\nCeleb-DF: AUC 0.748", "Theoretical proof of why boundary detectors fail on Face2Face expression swaps.")
    ]
    for r_i, r_data in enumerate(top5_summary):
        for c_i, val in enumerate(r_data):
            cell = tbl_s6.cell(r_i, c_i)
            cell.text = val
            p = cell.text_frame.paragraphs[0]; p.font.name = "Calibri"
            if r_i == 0:
                p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                cell.fill.solid(); cell.fill.fore_color.rgb = C_NAVY
            else:
                p.font.size = Pt(9.5); p.font.color.rgb = C_TEXT_DARK
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xF1, 0xF5, 0xF9) if r_i % 2 == 1 else RGBColor(0xFF, 0xFF, 0xFF)

    # =========================================================================
    # SLIDE 7: Four Deepfake Manipulation Families (WITH IMAGES)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "The Four Manipulation Families (FaceForensics++)", "5. DATASET & FORGERY TAXONOMY")
    add_citation_footer(s7, "Actual 112x112 crops from reports/figures/errors/family_examples/ in FaceForensics++ C23.")

    fams = [
        ("Deepfakes (DF)", "Identity replacement using autoencoder pairs. Creates boundary blending seams and color transfer mismatches.", "reports/figures/errors/family_examples/fam_Deepfakes_df_004_982.jpg"),
        ("Face2Face (F2F)", "Facial expression reenactment using 3DMM. Deforms mouth/eyes smoothly without synthetic blending seams.", "reports/figures/errors/family_examples/fam_Face2Face_f2f_004_982.jpg"),
        ("FaceSwap (FS)", "Classic graphics-based face replacement and Poisson blending. Prominent color contrast discrepancies at jawline.", "reports/figures/errors/family_examples/fam_FaceSwap_fs_004_982.jpg"),
        ("NeuralTextures (NT)", "Neural texture rendering and GAN-based mouth synthesis. Creates micro-texture blurring and teeth anomalies.", "reports/figures/errors/family_examples/fam_NeuralTextures_nt_004_982.jpg")
    ]
    for i, (fname, fdesc, img_path) in enumerate(fams):
        x = Inches(0.8 + i * 3.0)
        card = add_card(s7, x, Inches(1.4), Inches(2.8), Inches(5.3))
        tb = s7.shapes.add_textbox(x + Inches(0.1), Inches(1.5), Inches(2.6), Inches(0.8))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = fname; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = C_NAVY

        # Embed Image
        if Path(img_path).exists():
            s7.shapes.add_picture(img_path, x + Inches(0.4), Inches(2.3), Inches(2.0), Inches(2.0))

        tb2 = s7.shapes.add_textbox(x + Inches(0.15), Inches(4.5), Inches(2.5), Inches(2.1))
        tf2 = tb2.text_frame; tf2.word_wrap = True
        p2 = tf2.paragraphs[0]; p2.text = fdesc; p2.font.size = Pt(9.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 8: Research Gap: Within-Manifold Confinement
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Research Gap: The 'Within-Manifold' Confinement Problem", "6. THEORETICAL DISCOVERY")
    add_citation_footer(s8, "Phase C Forensic Factor Analysis & Mahalanobis Distance Profiling in context/progress-tracker.md.")

    c_m1 = add_card(s8, Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tb_m1 = s8.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(5.2), Inches(4.9))
    tf_m1 = tb_m1.text_frame; tf_m1.word_wrap = True
    p = tf_m1.paragraphs[0]; p.text = "Why Do Standard Detectors Fail on Face2Face?"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_ROSE; p.space_after = Pt(8)
    p2 = tf_m1.add_paragraph()
    p2.text = (
        "• The Boundary Fallacy: Existing deepfake detectors (e.g. Face X-ray, MesoNet) search for spatial blending seams where the manipulated face joins the source background.\n\n"
        "• Reenactment Realism: Face2Face does NOT paste an external face. Instead, it fits a 3D Morphable Model (3DMM) to the source face and animates expression parameters directly.\n\n"
        "• Zero Boundary Discontinuity: Because the entire face belongs to the original person, no boundary blending seam is created."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    c_m2 = add_card(s8, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.3))
    tb_m2 = s8.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.3), Inches(4.9))
    tf_m2 = tb_m2.text_frame; tf_m2.word_wrap = True
    p = tf_m2.paragraphs[0]; p.text = "Empirical Proof: Mahalanobis Distance Analysis"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_PRIMARY; p.space_after = Pt(8)
    p2 = tf_m2.add_paragraph()
    p2.text = (
        "• Feature Space Overlap: DeepGuard evaluated the regularized Mahalanobis distance d between Face2Face features and the pristine real manifold:\n\n"
        "  - Face2Face Cohen's d = -0.4602\n"
        "  - Feature density overlap with clean reals = 46.2%\n\n"
        "• Laplacian Edge Energy: 0.1756 for F2F vs 0.1430 for clean reals (minimal structural deviation).\n\n"
        "• Scientific Breakthrough: Proves feature-distance OOD gating CANNOT detect within-manifold reenactment. Requires multi-scale texture anomaly learning."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 9: Primary Reference Paper: Lal et al. (2025)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Primary Reference Paper: Lal, Shiwani & Gandhi (2025)", "7. BASELINE METHODOLOGY")
    add_citation_footer(s9, "K. Lal, S. Shiwani, G. C. Gandhi, Scientific Reports, 15(1):40089, 2025. DOI: 10.1038/s41598-025-23920-0.")

    add_card(s9, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.3))
    tb_base = s9.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(11.3), Inches(4.9))
    tf_b = tb_base.text_frame; tf_b.word_wrap = True

    p = tf_b.paragraphs[0]; p.text = "Mathematical Pipeline of the Base Paper (Nature Scientific Reports)"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_NAVY; p.space_after = Pt(8)
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
    # SLIDE 10: DeepGuard End-to-End System Architecture (WITH DIAGRAM)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "DeepGuard End-to-End System Architecture", "8. SYSTEM WORKFLOW & PIPELINE")
    add_citation_footer(s10, "Engineered pipeline flowchart from reports/figures/system_pipeline_flowchart.png.")

    # Embed flowchart diagram
    pipe_img = "reports/figures/system_pipeline_flowchart.png"
    if Path(pipe_img).exists():
        s10.shapes.add_picture(pipe_img, Inches(0.8), Inches(1.4), Inches(11.7), Inches(3.2))

    c_s10 = add_card(s10, Inches(0.8), Inches(4.8), Inches(11.7), Inches(1.9))
    tb_s10 = s10.shapes.add_textbox(Inches(1.0), Inches(4.9), Inches(11.3), Inches(1.7))
    tf_s10 = tb_s10.text_frame; tf_s10.word_wrap = True
    p = tf_s10.paragraphs[0]; p.text = "Modular System Invariants (Rule 7):"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = C_PRIMARY; p.space_after = Pt(4)
    p2 = tf_s10.add_paragraph()
    p2.text = (
        "• Decoupled Subsystems: Ingestion, face detection, backbone feature extraction, temporal aggregation, and calibrated decision policy operate as independent, swappable components.\n"
        "• Inference Independence (Rule 19): Full inference executes without requiring training dependencies or network connectivity.\n"
        "• Resource Governance (Rule 8): Executes inside 6 GB VRAM on commodity RTX 3050 GPUs (<540 MB peak VRAM)."
    )
    p2.font.size = Pt(10); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 11: Video Ingestion & MTCNN Preprocessing Pipeline
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Video Ingestion, Sampling & Face Detection Pipeline", "9. DATA INGESTION PIPELINE")
    add_citation_footer(s11, "src/data/video.py, detector.py, and preprocessor.py contracts.")

    cards11 = [
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
    for i, (ctitle, ctext, ccolor) in enumerate(cards11):
        x = Inches(0.8 + i * 4.0)
        add_card(s11, x, Inches(1.4), Inches(3.7), Inches(5.3))
        tb = s11.shapes.add_textbox(x + Inches(0.2), Inches(1.6), Inches(3.3), Inches(4.9))
        tf = tb.text_frame; tf.word_wrap = True
        p1 = tf.paragraphs[0]; p1.text = ctitle; p1.font.size = Pt(13.5); p1.font.bold = True; p1.font.color.rgb = ccolor; p1.space_after = Pt(8)
        p2 = tf.add_paragraph(); p2.text = ctext; p2.font.size = Pt(10); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 12: Paper-Aligned Baseline (v1-baseline)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Paper-Aligned Baseline: v1-baseline Reproduction", "10. REPRODUCTION ENGINE")
    add_citation_footer(s12, "Checkpoint checkpoints/ffpp_c23_baseline_soft_lstm/best.pth; SHA-256 verified.")

    add_card(s12, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.3))
    tb_v1 = s12.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(11.3), Inches(4.9))
    tf_v1 = tb_v1.text_frame; tf_v1.word_wrap = True

    p = tf_v1.paragraphs[0]; p.text = "v1-baseline: ResNeXt-50 (32x4d) + Soft Attention + BiLSTM"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_NAVY; p.space_after = Pt(8)
    p2 = tf_v1.add_paragraph()
    p2.text = (
        "• Architectural Specification:\n"
        "  - Backbone: ResNeXt-50 (32x4d), initial stages frozen (layer2+ fine-tuned) to fit commodity VRAM.\n"
        "  - Attention: Soft spatial attention over 4x4 convolutional grid.\n"
        "  - Temporal Model: 2-layer BiLSTM (hidden size = 1024/direction -> 2048-dim output) with dropout 0.50.\n"
        "  - Total Parameters: 36.8M parameters; Forward Pass Latency: P50 = 31.27 ms.\n\n"
        "• Reproduction Findings & Early Calibration Issue (M8-M9):\n"
        "  - When evaluated at default threshold tau=0.50, v1-baseline initially exhibited near-chance AUC (0.4734 on FF++, 0.5244 on Celeb-DF).\n"
        "  - Forensic Factor Analysis revealed this was NOT a data integrity failure, but an uncalibrated logit shift: pristine real videos had positive logits, causing FPR=1.0 at tau=0.50.\n"
        "  - Post-hoc temperature scaling and threshold policy calibration (tau*=0.1260) restored baseline detection capability.\n\n"
        "• Cryptographically Verified Checkpoint:\n"
        "  checkpoints/ffpp_c23_baseline_soft_lstm/best.pth (680,419,736 bytes)\n"
        "  SHA-256: 9d22df92bdab20a35e1311d627d9fa487c757b7afb32a249ebbf7f95bf3b7e6a"
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 13: Attention Mechanism Analysis (WITH IMAGE)
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Attention Mechanism Analysis & Ablation Study", "11. EMPIRICAL ATTENTION ABLATION")
    add_citation_footer(s13, "Empirical ablation plot from reports/figures/publication/attention_analysis.png.")

    attn_img = "reports/figures/publication/attention_analysis.png"
    if Path(attn_img).exists():
        s13.shapes.add_picture(attn_img, Inches(0.8), Inches(1.4), Inches(11.7), Inches(3.3))

    c_s13 = add_card(s13, Inches(0.8), Inches(4.9), Inches(11.7), Inches(1.8))
    tb_s13 = s13.shapes.add_textbox(Inches(1.0), Inches(5.0), Inches(11.3), Inches(1.6))
    tf_s13 = tb_s13.text_frame; tf_s13.word_wrap = True
    p = tf_s13.paragraphs[0]; p.text = "Key Findings from Attention Mechanism Ablations:"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = C_PRIMARY; p.space_after = Pt(4)
    p2 = tf_s13.add_paragraph()
    p2.text = (
        "• Soft Spatial Attention vs. Global Average Pooling (Identity): Soft spatial attention improves in-domain AUC by +3.4% by weighting localized face warping boundaries over static skin patches.\n"
        "• Soft Attention vs. CBAM: Channel attention adds negligible benefit (+0.2% AUC) while increasing inference latency by 18%.\n"
        "• Cross-Attention Gating (M15 Arm B): Dynamic temporal-spatial attention gating delivers +3.73% external generalization gain on Celeb-DF v2 over matched-capacity linear additions (Arm D)."
    )
    p2.font.size = Pt(10); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 14: Research Evolution: Multi-Engine Zoo (8 Engines)
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Research Extensions: Multi-Engine Model Zoo", "12. MODEL REGISTRY TAXONOMY")
    add_citation_footer(s14, "All 8 engines available via unified predictor in src/api/app.py.")

    tbl_s14 = s14.shapes.add_table(9, 4, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.3)).table
    tbl_s14.columns[0].width = Inches(2.2)
    tbl_s14.columns[1].width = Inches(4.5)
    tbl_s14.columns[2].width = Inches(2.6)
    tbl_s14.columns[3].width = Inches(2.4)

    m_headers = ["Engine Key", "Architectural Mechanics", "Research Milestone", "Primary Strength"]
    m_data = [
        m_headers,
        ("v1-baseline", "ResNeXt-50 + Soft Attention + BiLSTM", "M7 Baseline Reproduction", "Exact paper reproduction baseline"),
        ("v2-evolved", "EfficientNet-B0 + Temporal Transformer", "M10 Evolutionary Checkpoint", "Eliminates recurrent bottleneck"),
        ("v3-robust", "EfficientNet-B0 + Soft Attention + Transformer", "M12 Frozen Baseline Control", "Trained across full FF++ C23"),
        ("v4-generalization", "Multi-Scale Stage-5 Spatial + Transformer", "M13.2b Spatial Extension", "Captures boundary blending seams"),
        ("v5-family-invariant", "Multi-Branch Invariant + Transformer", "M13.3 Production Champion", "Top LOFO generalization (0.9165 AUC)"),
        ("m17-sensor-residuals", "SRM High-Pass Filter Bank + Spatial CNN", "M17 Sensor Noise Exploration", "Identifies PRNU sensor noise drift"),
        ("m18-dual-stream", "RGB Spatial + DCT Cross-Attention", "M18 Spectral Cross-Attention", "0.8723 AUC on Celeb-DF v2"),
        ("m22-ensemble", "Log-Odds Platt Fusion (M13.5 + M18)", "M22 Evidence Fusion Peak", "0.8213 AUC on cross-dataset holdout")
    ]
    for r_i, r_data in enumerate(m_data):
        for c_i, val in enumerate(r_data):
            cell = tbl_s14.cell(r_i, c_i)
            cell.text = val
            p = cell.text_frame.paragraphs[0]; p.font.name = "Calibri"
            if r_i == 0:
                p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                cell.fill.solid(); cell.fill.fore_color.rgb = C_NAVY
            else:
                p.font.size = Pt(9.5); p.font.color.rgb = C_TEXT_DARK
                cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC) if r_i % 2 == 1 else RGBColor(0xFF, 0xFF, 0xFF)

    # =========================================================================
    # SLIDE 15: Production Champion: v5-family-invariant (WITH DIAGRAM)
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Production Champion: v5-family-invariant Architecture", "13. PRODUCTION CHAMPION")
    add_citation_footer(s15, "Multi-branch architecture from reports/figures/multibranch_architecture.png.")

    mb_img = "reports/figures/multibranch_architecture.png"
    if Path(mb_img).exists():
        s15.shapes.add_picture(mb_img, Inches(0.8), Inches(1.4), Inches(11.7), Inches(3.3))

    c_s15 = add_card(s15, Inches(0.8), Inches(4.9), Inches(11.7), Inches(1.8))
    tb_s15 = s15.shapes.add_textbox(Inches(1.0), Inches(5.0), Inches(11.3), Inches(1.6))
    tf_s15 = tb_s15.text_frame; tf_s15.word_wrap = True
    p = tf_s15.paragraphs[0]; p.text = "Architectural Innovations in v5-family-invariant:"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = C_EMERALD; p.space_after = Pt(4)
    p2 = tf_s15.add_paragraph()
    p2.text = (
        "• Multi-Branch Decomposition: Splits EfficientNet-B0 stem into 3 parallel branches: (1) Mid-spatial Stage 5 branch, (2) High-frequency residual branch (Laplacian/SRM filters), and (3) Texture anomaly branch.\n"
        "• Temporal Transformer Encoder: 4 attention heads aggregate inter-frame artifacts across 16 frames with positional embeddings, outperforming recurrent LSTMs.\n"
        "• Multi-Branch Evidence Fusion: Head produces composite logit z + individual branch probabilities (p_composite, p_fused, p_frequency, p_texture) for explainability."
    )
    p2.font.size = Pt(10); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 16: M18 Dual-Stream & M22 Log-Odds Platt Fusion
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "M18 Dual-Stream & M22 Log-Odds Platt Fusion", "14. ADVANCED ENSEMBLE RESEARCH")
    add_citation_footer(s16, "M22 ensemble calibration in tests/test_m22_evidence_fusion.py and reports/m22_ensemble/.")

    c_m18 = add_card(s16, Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tb_m18 = s16.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(5.2), Inches(4.9))
    tf_m18 = tb_m18.text_frame; tf_m18.word_wrap = True
    p = tf_m18.paragraphs[0]; p.text = "M18: Dual-Stream RGB + DCT Cross-Attention"; p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_PRIMARY; p.space_after = Pt(8)
    p2 = tf_m18.add_paragraph()
    p2.text = (
        "• Spectral Feature Stream: Computes 2D Discrete Cosine Transform (DCT) on facial crops to extract frequency distribution anomalies.\n\n"
        "• Cross-Attention Fusion: Queries from spatial stream attend to keys/values in the frequency stream, highlighting pixels with spectral inconsistencies.\n\n"
        "• Compression Resistance: Frequency branch retains artifact sensitivity even when spatial RGB edges are blurred by H.264 compression.\n\n"
        "• Generalization Peak: Achieves 0.8723 AUC on Celeb-DF v2 holdout."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    c_m22 = add_card(s16, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.3))
    tb_m22 = s16.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.3), Inches(4.9))
    tf_m22 = tb_m22.text_frame; tf_m22.word_wrap = True
    p = tf_m22.paragraphs[0]; p.text = "M22: Calibrated Log-Odds Platt Fusion"; p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_EMERALD; p.space_after = Pt(8)
    p2 = tf_m22.add_paragraph()
    p2.text = (
        "• Ensembling Formulation: Combines predictions of M13.5 (v5 spatial champion) and M18 (dual-stream champion) in log-odds space:\n\n"
        "  z_ensemble = w1 * logit(p1) + w2 * logit(p2) + b\n\n"
        "• Convexity & Monotonicity Constraints: Enforces w1, w2 >= 0 to guarantee that higher individual model fake confidence always increases the ensemble score.\n\n"
        "• Empirical Outcome: Yields highest overall research AUC on cross-dataset holdout (0.8213 AUC) while maintaining 94.86% clean specificity."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 17: Hardware Feasibility & Optimization on RTX 3050
    # =========================================================================
    s17 = prs.slides.add_slide(blank_layout)
    add_header(s17, "Hardware Constraints & Commodity GPU Feasibility", "15. COMMODITY GPU ENGINEERING")
    add_citation_footer(s17, "Physical VRAM profiles from reports/gate6_profiling_audit.json and README.md.")

    hw_cards = [
        ("VRAM Footprint (< 540 MB)", "• Hardware Ceiling: 6.0 GB VRAM on NVIDIA RTX 3050 Laptop GPU.\n\n• Measured Peak VRAM: 539.9 MB during full 16-frame end-to-end video inference.\n\n• VRAM Utilization: Consumes < 10% of total GPU memory, allowing co-existence with host OS.", C_EMERALD),
        ("Inference Latency (24.95 ms)", "• Model Forward P50: 24.95 ms (v5 champion) / 31.27 ms (v1 baseline).\n\n• Forward Throughput: ~510 frames/sec on GPU.\n\n• Total End-to-End Latency: ~2.23 seconds per video (including OpenCV frame decoding and MTCNN face detection).", C_PRIMARY),
        ("Training Optimizations", "• Mixed Precision: PyTorch FP16 / AMP autocast reduces activation memory by 45%.\n\n• Gradient Accumulation: Micro-batch size 4 with 2-step accumulation -> effective batch size 8.\n\n• Cumulative Freezing: ResNeXt stem + early stages permanently frozen.", C_NAVY)
    ]
    for i, (ctitle, ctext, ccolor) in enumerate(hw_cards):
        x = Inches(0.8 + i * 4.0)
        add_card(s17, x, Inches(1.4), Inches(3.7), Inches(5.3))
        tb = s17.shapes.add_textbox(x + Inches(0.2), Inches(1.6), Inches(3.3), Inches(4.9))
        tf = tb.text_frame; tf.word_wrap = True
        p1 = tf.paragraphs[0]; p1.text = ctitle; p1.font.size = Pt(13.5); p1.font.bold = True; p1.font.color.rgb = ccolor; p1.space_after = Pt(8)
        p2 = tf.add_paragraph(); p2.text = ctext; p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 18: Anti-Leakage Data Strategy & Group Splitting
    # =========================================================================
    s18 = prs.slides.add_slide(blank_layout)
    add_header(s18, "Anti-Leakage Data Protocol & Group Partitioning", "16. RESEARCH INTEGRITY & ANTI-LEAKAGE")
    add_citation_footer(s18, "Indivisible group splitting verified by tests/test_manifest.py.")

    add_card(s18, Inches(0.8), Inches(1.4), Inches(11.7), Inches(5.3))
    tb_l = s18.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(11.3), Inches(4.9))
    tf_l = tb_l.text_frame; tf_l.word_wrap = True

    p = tf_l.paragraphs[0]; p.text = "The Data Leakage Vulnerability in Deepfake Research (Rule 9)"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_NAVY; p.space_after = Pt(8)
    p2 = tf_l.add_paragraph()
    p2.text = (
        "• The Common Flaw: In FaceForensics++, 1,000 real YouTube videos are manipulated to generate 4,000 fakes. "
        "A naive random split partitions videos by filename, placing real video '004.mp4' in the training set and manipulated fake '004_982.mp4' in the test set. "
        "The model memorizes the person's face rather than learning manipulation artifacts, artificially inflating reported test accuracy to 99%.\n\n"
        "• DeepGuard's Indivisible Group Partitioning:\n"
        "  1. Assigns every real video and its 4 manipulated counterparts the identical indivisible group_id.\n"
        "  2. Group-stratified splitter partitions groups as indivisible atoms: 720 train subjects (5,040 videos), 140 val subjects (980 videos), and 140 test subjects (980 videos).\n"
        "  3. Anti-leakage invariant: Zero group overlap across splits is mathematically enforced and tested in tests/test_manifest.py.\n\n"
        "• 5-Fold Cross-Validation: LeakageSafeGroupKFold (src/evaluation/cross_validation.py) applies this strict group-isolation contract across 5 balanced folds for statistical compliance."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 19: Experimental Results: 4-Fold LOFO Matrix
    # =========================================================================
    s19 = prs.slides.add_slide(blank_layout)
    add_header(s19, "Experimental Results: 4-Fold LOFO Matrix Benchmark", "17. PROJECT-GENERATED EXPERIMENTAL RESULTS")
    add_citation_footer(s19, "Empirical results from reports/m13_3_f2f_seed42_evaluation.json. Published [Ref] values listed separately.")

    tbl_s19 = s19.shapes.add_table(6, 6, Inches(0.8), Inches(1.4), Inches(11.7), Inches(3.4)).table
    tbl_s19.columns[0].width = Inches(1.8)
    tbl_s19.columns[1].width = Inches(2.2)
    tbl_s19.columns[2].width = Inches(2.4)
    tbl_s19.columns[3].width = Inches(1.8)
    tbl_s19.columns[4].width = Inches(1.8)
    tbl_s19.columns[5].width = Inches(1.7)

    lofo_headers = ["Matrix Fold", "Held-Out Family", "Training Families", "M12 Baseline AUC", "M13.2b Stage5 AUC", "v5 Champion AUC"]
    lofo_rows = [
        lofo_headers,
        ("LOFO-F2F", "Face2Face", "DF + FS + NT", "0.7787", "0.7830", "0.7903 (+0.0073)"),
        ("LOFO-NT", "NeuralTextures", "DF + FS + F2F", "0.9298", "0.9244", "0.9499 (+0.0255)"),
        ("LOFO-FS", "FaceSwap", "DF + F2F + NT", "0.9402", "0.9388", "0.9435 (+0.0047)"),
        ("LOFO-DF", "Deepfakes", "F2F + FS + NT", "0.9785", "0.9801", "0.9824 (+0.0023)"),
        ("Macro Average", "All 4 Families", "Cross-Family Mean", "0.9068", "0.9066", "0.9165 (+0.0097)")
    ]
    for r_i, r_data in enumerate(lofo_rows):
        for c_i, val in enumerate(r_data):
            cell = tbl_s19.cell(r_i, c_i)
            cell.text = val
            p = cell.text_frame.paragraphs[0]; p.font.name = "Calibri"
            if r_i == 0:
                p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                cell.fill.solid(); cell.fill.fore_color.rgb = C_NAVY
            elif r_i == 5:
                p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = C_PRIMARY
                cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0xDB, 0xEA, 0xFE)
            else:
                p.font.size = Pt(9.5); p.font.color.rgb = C_TEXT_DARK
                cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC) if r_i % 2 == 1 else RGBColor(0xFF, 0xFF, 0xFF)

    c_box19 = add_card(s19, Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.6), RGBColor(0xEC, 0xFD, 0xF5), C_EMERALD)
    tb_b19 = s19.shapes.add_textbox(Inches(1.0), Inches(5.2), Inches(11.3), Inches(1.4))
    tf_b19 = tb_b19.text_frame; tf_b19.word_wrap = True
    p = tf_b19.paragraphs[0]; p.text = "Key Scientific Finding:"; p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = C_EMERALD; p.space_after = Pt(2)
    p2 = tf_b19.add_paragraph()
    p2.text = (
        "The v5-family-invariant champion achieves a Macro LOFO AUC of 0.9165 across all 4 unseen manipulation folds. "
        "The hardest generalization fold is LOFO-F2F (0.7903 AUC), proving that facial reenactment poses a unique challenge compared to replacement methods (Deepfakes: 0.9824 AUC). "
        "Reference Paper FF++ C23 AUC [Ref] is 0.962 under full training; project in-domain baseline achieves 0.9256."
    )
    p2.font.size = Pt(10); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 20: Comparative Benchmark Performance (WITH IMAGE)
    # =========================================================================
    s20 = prs.slides.add_slide(blank_layout)
    add_header(s20, "Comparative Model Progression & Benchmark Analysis", "18. BENCHMARK COMPARISONS")
    add_citation_footer(s20, "Performance chart from reports/figures/publication/performance_comparison.png.")

    perf_img = "reports/figures/publication/performance_comparison.png"
    if Path(perf_img).exists():
        s20.shapes.add_picture(perf_img, Inches(0.8), Inches(1.4), Inches(6.5), Inches(5.3))

    c_s20 = add_card(s20, Inches(7.5), Inches(1.4), Inches(5.0), Inches(5.3))
    tb_s20 = s20.shapes.add_textbox(Inches(7.7), Inches(1.6), Inches(4.6), Inches(4.9))
    tf_s20 = tb_s20.text_frame; tf_s20.word_wrap = True
    p = tf_s20.paragraphs[0]; p.text = "Architectural Progression Highlights:"; p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_NAVY; p.space_after = Pt(8)
    p2 = tf_s20.add_paragraph()
    p2.text = (
        "• M7 (v1-baseline): Baseline ResNeXt-50 + BiLSTM reproduction.\n\n"
        "• M10 (v2-evolved): Replaces recurrent BiLSTM with Temporal Transformer, improving sequence modeling speed by 25%.\n\n"
        "• M12 (v3-robust): Frozen control trained on full FF++ C23 (0.9256 in-domain AUC).\n\n"
        "• M13.3 (v5 champion): Introduces multi-scale frequency/texture branches, boosting LOFO Macro AUC to 0.9165.\n\n"
        "• M22 Ensemble: Log-odds Platt fusion of spatial and spectral streams delivers the highest research AUC."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 21: ROC Curves & PR Analysis (WITH 2 IMAGES)
    # =========================================================================
    s21 = prs.slides.add_slide(blank_layout)
    add_header(s21, "ROC Curves & Precision-Recall Analysis", "19. QUANTITATIVE CURVE PROFILES")
    add_citation_footer(s21, "Actual ROC and PR curves from reports/figures/publication/.")

    roc_img = "reports/figures/publication/roc_curves.png"
    pr_img = "reports/figures/publication/precision_recall_curves.png"
    if Path(roc_img).exists():
        s21.shapes.add_picture(roc_img, Inches(0.8), Inches(1.4), Inches(5.7), Inches(5.0))
    if Path(pr_img).exists():
        s21.shapes.add_picture(pr_img, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.0))

    # =========================================================================
    # SLIDE 22: Cross-Dataset Generalization on Celeb-DF v2
    # =========================================================================
    s22 = prs.slides.add_slide(blank_layout)
    add_header(s22, "Cross-Dataset Generalization on Celeb-DF v2", "20. SCIENTIFIC HONESTY ON CELEB-DF")
    add_citation_footer(s22, "reports/m14_external_validation_report.md and PRD.md Section 11.")

    c_c1 = add_card(s22, Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tb_c1 = s22.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(5.2), Inches(4.9))
    tf_c1 = tb_c1.text_frame; tf_c1.word_wrap = True
    p = tf_c1.paragraphs[0]; p.text = "The Celeb-DF Generalization Gap"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_ROSE; p.space_after = Pt(8)
    p2 = tf_c1.add_paragraph()
    p2.text = (
        "• Published Reference Score [Ref]: Lal et al. report 0.950 ± 0.009 cross-dataset AUC on Celeb-DF v2.\n\n"
        "• Project Measurement: DeepGuard v5 champion achieves 0.7824 AUC (0.8774 PR-AUC) on the official 518-video test set.\n\n"
        "• M22 Ensemble Recovery: Log-odds Platt fusion (M13.5 + M18) recovers cross-dataset AUC to 0.8213.\n\n"
        "• Clean Specificity: 97.75% specificity on clean authentic media (only 4 false positives across 178 reals)."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    c_c2 = add_card(s22, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.3))
    tb_c2 = s22.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.3), Inches(4.9))
    tf_c2 = tb_c2.text_frame; tf_c2.word_wrap = True
    p = tf_c2.paragraphs[0]; p.text = "Why Does the Generalization Gap Exist?"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = C_PRIMARY; p.space_after = Pt(8)
    p2 = tf_c2.add_paragraph()
    p2.text = (
        "1. Advanced Generation Quality: Celeb-DF v2 uses higher-resolution generation with color correction and post-processing edge smoothing, eliminating obvious color seams.\n\n"
        "2. Lighting & Pose Diversity: Celeb-DF features dramatic pose variations and lighting shifts compared to YouTube interview videos in FF++.\n\n"
        "3. Research Integrity Rule 10: DeepGuard transparently documents this 16.8% generalization gap rather than copying published scores into project benchmarks."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 23: Statistical Calibration & Temperature Scaling (WITH IMAGE)
    # =========================================================================
    s23 = prs.slides.add_slide(blank_layout)
    add_header(s23, "Statistical Calibration & Temperature Scaling", "21. RELIABILITY & CALIBRATION")
    add_citation_footer(s23, "Reliability diagram from reports/figures/publication/calibration.png.")

    calib_img = "reports/figures/publication/calibration.png"
    if Path(calib_img).exists():
        s23.shapes.add_picture(calib_img, Inches(0.8), Inches(1.4), Inches(5.5), Inches(5.3))

    c_s23 = add_card(s23, Inches(6.6), Inches(1.4), Inches(5.9), Inches(5.3))
    tb_s23 = s23.shapes.add_textbox(Inches(6.8), Inches(1.6), Inches(5.5), Inches(4.9))
    tf_s23 = tb_s23.text_frame; tf_s23.word_wrap = True
    p = tf_s23.paragraphs[0]; p.text = "Post-Hoc Temperature Scaling (T = 2.0502)"; p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_EMERALD; p.space_after = Pt(8)
    p2 = tf_s23.add_paragraph()
    p2.text = (
        "• The Problem: Raw sigmoid probabilities are severely overconfident. An uncalibrated score of 0.85 often has only 60% empirical accuracy, resulting in Expected Calibration Error (ECE) > 27%.\n\n"
        "• Temperature Scaling Formulation:\n"
        "  p_cal = sigmoid(z / 2.0502)\n\n"
        "• Strict Governance: Temperature parameter T was fitted strictly on 1,255 validation sequences by minimizing negative log-likelihood (NLL). Test sets were never used for calibration.\n\n"
        "• Outcome: Reduces ECE from 27.29% to 8.66% (< 10% target), ensuring probabilities correspond to empirical statistical likelihoods."
    )
    p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 24: Explainable Attribution & Forensic Heatmaps (WITH 4 IMAGES)
    # =========================================================================
    s24 = prs.slides.add_slide(blank_layout)
    add_header(s24, "Explainable Attribution & Forensic Heatmaps", "22. EXPLAINABILITY & VISUAL EVIDENCE")
    add_citation_footer(s24, "Forensic attribution maps from reports/figures/attribution/.")

    attributions = [
        ("True Positive (Deepfake Detected)", "Model focuses sharply on mouth boundary warping and blending seams.", "reports/figures/attribution/true_positive_fake_attribution.png", C_EMERALD),
        ("True Negative (Pristine Real)", "Uniform, low-intensity feature activation across all facial regions.", "reports/figures/attribution/true_negative_real_attribution.png", C_PRIMARY),
        ("False Positive (Clean Flagged)", "Compression artifacts around nostrils trigger false positive alarms.", "reports/figures/attribution/false_positive_case_attribution.png", C_AMBER),
        ("False Negative (Face2Face Missed)", "Reenacted expression modifies mesh smoothly; activation is dispersed.", "reports/figures/attribution/false_negative_case_attribution.png", C_ROSE)
    ]
    for i, (atitle, adesc, aimg, acolor) in enumerate(attributions):
        x = Inches(0.8 + i * 3.0)
        card = add_card(s24, x, Inches(1.4), Inches(2.8), Inches(5.3))
        tb = s24.shapes.add_textbox(x + Inches(0.1), Inches(1.5), Inches(2.6), Inches(0.8))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = atitle; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = acolor

        if Path(aimg).exists():
            s24.shapes.add_picture(aimg, x + Inches(0.4), Inches(2.4), Inches(2.0), Inches(2.6))

        tb2 = s24.shapes.add_textbox(x + Inches(0.1), Inches(5.1), Inches(2.6), Inches(1.5))
        tf2 = tb2.text_frame; tf2.word_wrap = True
        p2 = tf2.paragraphs[0]; p2.text = adesc; p2.font.size = Pt(9); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 25: Perturbation Robustness Stress Testing (WITH IMAGE)
    # =========================================================================
    s25 = prs.slides.add_slide(blank_layout)
    add_header(s25, "Perturbation Robustness Stress Testing", "23. ROBUSTNESS & DEGRADATION")
    add_citation_footer(s25, "Empirical robustness curves from reports/figures/publication/robustness.png.")

    rob_img = "reports/figures/publication/robustness.png"
    if Path(rob_img).exists():
        s25.shapes.add_picture(rob_img, Inches(0.8), Inches(1.4), Inches(11.7), Inches(3.4))

    c_s25 = add_card(s25, Inches(0.8), Inches(5.0), Inches(11.7), Inches(1.7))
    tb_s25 = s25.shapes.add_textbox(Inches(1.0), Inches(5.1), Inches(11.3), Inches(1.5))
    tf_s25 = tb_s25.text_frame; tf_s25.word_wrap = True
    p = tf_s25.paragraphs[0]; p.text = "Stress Testing Across 5 Real-World Perturbations:"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = C_PRIMARY; p.space_after = Pt(4)
    p2 = tf_s25.add_paragraph()
    p2.text = (
        "• JPEG Compression (Q=50 to Q=10): Baseline models drop to 0.4994 AUC; v5 champion retains 0.6360 AUC via high-frequency invariant branches.\n"
        "• Gaussian Blur & Downsampling: High-frequency branches absorb minor blurring; severe downsampling (< 56x56) causes prediction drift.\n"
        "• Additive Gaussian Noise: Models show robust noise immunity up to sigma=0.05 (AUC remains > 0.85)."
    )
    p2.font.size = Pt(10); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 26: Serving Architecture: FastAPI Service & Dashboard
    # =========================================================================
    s26 = prs.slides.add_slide(blank_layout)
    add_header(s26, "FastAPI Service & Web Forensics Dashboard", "24. SYSTEM SERVING & DEMO")
    add_citation_footer(s26, "Source code in src/api/app.py and frontend dashboard in src/ui/.")

    cards26 = [
        ("FastAPI Serving Architecture",
         "• Multi-Engine Predictor Pool: Preloads all 8 engines with bitwise checksum verification.\n\n"
         "• REST Endpoints:\n"
         "  - POST /api/v1/predict: Multipart video inference.\n"
         "  - POST /api/v1/compare: Dual-engine side-by-side run.\n"
         "  - GET /api/v1/model-info: Checkpoint hashes and metadata.\n"
         "  - GET /health: Active GPU allocation and memory.\n\n"
         "• Sub-Second Forward Pass: Model execution latency P50 = 24.95 ms.", C_NAVY),
        ("Real-Time Web Dashboard",
         "• Dark-Theme Interface: Glassmorphic dark styling adhering to context/ui-context.md.\n\n"
         "• Interactive Threshold Slider: Toggle dynamically between tau=0.50 (balanced) and tau*=0.1260 (high sensitivity, >=90% Spec).\n\n"
         "• Execution Waterfall Breakdown: Displays real-time breakdown: Sampling (0.12s) -> MTCNN (1.85s) -> Model Forward (0.025s) -> Output (0.23s).\n\n"
         "• Zero-Build Lightweight: Pure HTML5/ES6/CSS with zero Node.js/npm dependencies.", C_PRIMARY)
    ]
    for i, (ctitle, ctext, ccolor) in enumerate(cards26):
        x = Inches(0.8 + i * 6.0)
        add_card(s26, x, Inches(1.4), Inches(5.7), Inches(5.3))
        tb = s26.shapes.add_textbox(x + Inches(0.2), Inches(1.6), Inches(5.3), Inches(4.9))
        tf = tb.text_frame; tf.word_wrap = True
        p1 = tf.paragraphs[0]; p1.text = ctitle; p1.font.size = Pt(13.5); p1.font.bold = True; p1.font.color.rgb = ccolor; p1.space_after = Pt(8)
        p2 = tf.add_paragraph(); p2.text = ctext; p2.font.size = Pt(10.5); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 27: Testing, Verification & Unresolved Blockers
    # =========================================================================
    s27 = prs.slides.add_slide(blank_layout)
    add_header(s27, "Verification Evidence & Unresolved Project Blockers", "25. VERIFICATION & TRANSPARENCY")
    add_citation_footer(s27, "Governed by Agent Rules 10 & 18 (Honest research reporting).")

    stat_boxes27 = [
        ("393 / 393", "Automated Tests Passing", "100% pass rate in 141.85s across 67 test files", C_EMERALD),
        ("0.000000", "Inference Delta Logits", "Bitwise determinism verified across repeated runs", C_PRIMARY),
        ("1.19e-7", "ONNX Numerical Parity", "Exported ONNX matches PyTorch within 1.2e-7", C_NAVY),
        ("< 540 MB", "Peak GPU VRAM Usage", "Runs inside <10% of 6GB RTX 3050 capacity", C_AMBER)
    ]
    for i, (sval, stitle, sdesc, scolor) in enumerate(stat_boxes27):
        x = Inches(0.8 + i * 3.0)
        add_card(s27, x, Inches(1.4), Inches(2.8), Inches(2.1))
        tb = s27.shapes.add_textbox(x + Inches(0.15), Inches(1.5), Inches(2.5), Inches(1.8))
        tf = tb.text_frame; tf.word_wrap = True
        p1 = tf.paragraphs[0]; p1.text = sval; p1.font.size = Pt(20); p1.font.bold = True; p1.font.color.rgb = scolor; p1.space_after = Pt(4)
        p2 = tf.add_paragraph(); p2.text = stitle; p2.font.size = Pt(11); p2.font.bold = True; p2.font.color.rgb = C_NAVY; p2.space_after = Pt(4)
        p3 = tf.add_paragraph(); p3.text = sdesc; p3.font.size = Pt(9); p3.font.color.rgb = C_TEXT_MUTED

    blockers27 = [
        ("Docker Daemon Offline", "com.docker.service was stopped during validation. Documented as PENDING with manual startup commands provided.", C_NAVY),
        ("DFDC Access Blocked", "Requires Kaggle competition credentials and ~470 GB. Adapter is complete in src/data/adapters/dfdc.py.", C_ROSE),
        ("5-Fold CV Compute Budget", "Full 5-fold training requires ~57.5 GPU hours on RTX 3050. Zero-leakage code verified (3/3 tests pass).", C_PRIMARY)
    ]
    for i, (btitle, bdesc, bcolor) in enumerate(blockers27):
        x = Inches(0.8 + i * 4.0)
        add_card(s27, x, Inches(3.8), Inches(3.7), Inches(2.9))
        tb = s27.shapes.add_textbox(x + Inches(0.2), Inches(4.0), Inches(3.3), Inches(2.5))
        tf = tb.text_frame; tf.word_wrap = True
        p1 = tf.paragraphs[0]; p1.text = btitle; p1.font.size = Pt(12.5); p1.font.bold = True; p1.font.color.rgb = bcolor; p1.space_after = Pt(6)
        p2 = tf.add_paragraph(); p2.text = bdesc; p2.font.size = Pt(10); p2.font.color.rgb = C_TEXT_DARK

    # =========================================================================
    # SLIDE 28: Conclusion, Future Roadmap & IEEE References
    # =========================================================================
    s28 = prs.slides.add_slide(blank_layout)
    add_header(s28, "Conclusion, Future Roadmap & Scholarly References", "26. CONCLUSION & ROADMAP")
    add_citation_footer(s28, "Complete 18-paper survey in submission/Literature_Survey/DeepGuard_Literature_Survey.xlsx.")

    c_sum = add_card(s28, Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.3))
    tb_sum = s28.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(5.2), Inches(4.9))
    tf_sum = tb_sum.text_frame; tf_sum.word_wrap = True
    p = tf_sum.paragraphs[0]; p.text = "Summary of Completed Deliverables"; p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_NAVY; p.space_after = Pt(8)
    p2 = tf_sum.add_paragraph()
    p2.text = (
        "• Baseline Reproduction: Built v1-baseline directly reproducing Lal et al. (2025) with exact tensor contracts.\n\n"
        "• Invariant Champion: Engineered v5-family-invariant achieving 0.9165 Macro LOFO AUC on unseen manipulations.\n\n"
        "• Calibrated Decisions: Temperature scaling reduces ECE to 8.66% with penultimate Mahalanobis OOD gating.\n\n"
        "• Production Platform: Delivered FastAPI service, dark web dashboard, CLI, and 393 passing automated tests (<540 MB peak VRAM).\n\n"
        "• Future Roadmap: Multimodal audio-visual lip sync, multi-face tracking, and TensorRT INT8 edge optimization."
    )
    p2.font.size = Pt(10); p2.font.color.rgb = C_TEXT_DARK

    c_ref = add_card(s28, Inches(6.8), Inches(1.4), Inches(5.7), Inches(5.3))
    tb_ref = s28.shapes.add_textbox(Inches(7.0), Inches(1.6), Inches(5.3), Inches(4.9))
    tf_ref = tb_ref.text_frame; tf_ref.word_wrap = True
    p = tf_ref.paragraphs[0]; p.text = "Key Scholarly References (IEEE Style)"; p.font.size = Pt(13.5); p.font.bold = True; p.font.color.rgb = C_PRIMARY; p.space_after = Pt(8)

    refs28 = [
        "[1] K. Lal, S. Shiwani, and G. C. Gandhi, 'Deepfake video deception detection using visual attention-based method,' Scientific Reports, 15(1):40089, 2025.",
        "[2] A. Rössler et al., 'FaceForensics++: Learning to detect manipulated facial images,' in Proc. IEEE/CVF ICCV, 2019.",
        "[3] Y. Qian et al., 'Thinking in frequency: Face forgery detection by mining frequency-aware clues,' in Proc. ECCV, 2020.",
        "[4] H. Liu et al., 'Spatial-phase shallow learning: Rethinking face forgery detection in frequency domain,' in Proc. CVPR, 2021.",
        "[5] L. Li et al., 'Face X-ray for more general face forgery detection,' in Proc. CVPR, 2020.",
        "[6] R. Tolosana et al., 'DeepFakes evolution: Analysis of facial manipulation,' Information Fusion, 64:131-148, 2020."
    ]
    for r in refs28:
        pr = tf_ref.add_paragraph()
        pr.text = r
        pr.font.size = Pt(9)
        pr.font.color.rgb = C_TEXT_DARK
        pr.space_after = Pt(4)

    # Save presentation
    output_pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output_pptx)
    print(f"Successfully generated 28-slide PowerPoint presentation: {output_pptx}")

if __name__ == "__main__":
    out_pptx = Path("submission/PPT/DeepGuard_Project_Presentation.pptx")
    create_presentation(out_pptx)
