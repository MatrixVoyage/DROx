# DeepGuard — Initial Academic Project Proposal Submission Checklist

**Project**: DeepGuard — Visual Deepfake Video Detection & Forensics Platform  
**Target Group**: IVP Group 10 Academic Project Proposal Submission  
**Documentation Stage**: Initial Project Planning & Specification (Pre-Implementation Baseline)  
**Date**: October 15, 2026  
**Auditor**: Senior Research Analyst / Academic Project Lead  

---

## 1. Operating Rules Compliance Matrix

This checklist certifies that all initial deliverables strictly conform to the 10 Operating Rules for pre-implementation academic submission:

| # | Operating Rule | Verification Evidence | Status |
|---|---|---|:---:|
| **1** | **Pre-Implementation Scope** | All documents describe DeepGuard as a proposed system prior to experimental execution. No retrospective milestones (e.g., M23, test pass counts) are included in proposal documents. | **VERIFIED** |
| **2** | **Reference-Grounded Foundations** | Motivation, research gaps, methodology, and evaluation protocols are derived from verified literature, specifically the base paper by Lal et al. (Nature Scientific Reports 2025) and 18 peer-reviewed studies. | **VERIFIED** |
| **3** | **Future-Oriented Language** | Strictly formulated using proposal-oriented language (*"will develop"*, *"is proposed to"*, *"will evaluate"*, *"is expected to"*). No past-tense claims of project completion. | **VERIFIED** |
| **4** | **Exclusion of Retrospective Artifacts** | Checkpoint weights, empirical AUC figures from prior local training, automated test pass counts (393 passing), and API implementation traces are excluded from proposal deliverables. | **VERIFIED** |
| **5** | **Zero Modification to Source Code** | Model source code in `src/`, unit tests in `tests/`, and existing checkpoints in `checkpoints/` were untouched and preserved intact. | **VERIFIED** |
| **6** | **Judicious Documentation Reuse** | Reused only foundational specifications (problem definition, architecture topology, dataset split schema). Discarded all post-experiment metrics and completion reports. | **VERIFIED** |
| **7** | **Strict Citation & Result Attribution** | Base paper results (e.g., Lal 2025: 91.07% accuracy, 0.9632 AUC on FF++ C23) are explicitly attributed as published reference baselines and never claimed as DeepGuard achievements. | **VERIFIED** |
| **8** | **Authentic References & Datasets** | 18 peer-reviewed papers cataloged with verified DOIs/arXiv IDs. Benchmark datasets (FaceForensics++ C23, Celeb-DF v2, DFDC) accurately documented without fabricated numbers. | **VERIFIED** |
| **9** | **Architectural Consistency** | The proposed 6-stage pipeline (Frame Extraction -> MTCNN Face Detection -> 112x112 Crop -> ResNeXt-50 Backbone -> Spatial Attention -> BiLSTM Temporal Classifier) is identical across SRS, Presentation, Design Spec, and Test Plan. | **VERIFIED** |
| **10** | **Academic Formatting Standards** | All deliverables follow formal institutional submission guidelines with dual-format editable (.docx, .pptx, .xlsx) and archival PDF versions. | **VERIFIED** |

---

## 2. Deliverables Inventory & Verification

| Deliverable Area | Primary File | Archival / Supplementary | Page / Slide Count | Status |
|---|---|---|:---:|:---:|
| **SRS & System Design** | `SRS_and_Design_Report/DeepGuard_SRS.docx` | `SRS_and_Design_Report/DeepGuard_SRS.pdf`<br>`SRS_and_Design_Report/DeepGuard_Architecture.md` | 9 Sections (24 pages DOCX, 16 pages PDF) | **COMPLETE** |
| **Project Presentation** | `PPT/DeepGuard_Project_Presentation.pptx` | `PPT/DeepGuard_Project_Presentation.pdf` | 26 Widescreen Slides with 15 Figures | **COMPLETE** |
| **Literature Survey** | `Literature_Survey/DeepGuard_Literature_Survey.xlsx` | `Literature_Survey/DeepGuard_Literature_Survey.csv`<br>`Literature_Survey/references.md`<br>`Literature_Survey/papers/` (16 PDFs) | 18 Papers (4 Sheets: Survey, Top 5, Dataset Map, References) | **COMPLETE** |
| **Base Papers** | `Base_Papers/primary_reference.md` | `Base_Papers/Lal_2025_Deepfake_Visual_Attention_Base_Paper.pdf`<br>`Base_Papers/README.md` | In-depth analysis of Lal et al. (Nature Sci Rep 2025) | **COMPLETE** |
| **Test Plan & Test Data** | `Test_Data/DeepGuard_Test_Plan_and_Data_Documentation.docx` | `Test_Data/DeepGuard_Test_Plan_and_Data_Documentation.pdf`<br>`Test_Data/sample_manifest.csv`<br>`Test_Data/README.md` | 7 Sections, 10 Proposed Test Cases (TC-001 to TC-010) | **COMPLETE** |

---

## 3. Presentation Content & Figure Index

The slide deck (`DeepGuard_Project_Presentation.pptx` and `.pdf`) consists of **26 widescreen slides** structured as follows:

1. **Slide 1**: Title Slide — DeepGuard Project Proposal (IVP Group 10)
2. **Slide 2**: Problem Context & Motivation — The Rise of Hyper-Realistic Deepfakes
3. **Slide 3**: The Real-World Threat Landscape (Legal, Democratic, Identity)
4. **Slide 4**: Visual Taxonomy of Facial Manipulation *(Featuring 4 authentic crops: Deepfakes, Face2Face, FaceSwap, NeuralTextures)*
5. **Slide 5**: The Core Technical Challenge — Subtle & Non-Rigid Artifacts
6. **Slide 6**: Comprehensive Literature Survey (18 Peer-Reviewed Works)
7. **Slide 7**: Literature Survey — Comparative Methodology Matrix
8. **Slide 8**: Primary Reference Base Paper — Lal, Shiwani & Gandhi (Nature Sci Rep 2025)
9. **Slide 9**: Critical Research Gaps Identified in Existing Literature
10. **Slide 10**: DeepGuard Problem Statement
11. **Slide 11**: Project Objectives & Success Criteria
12. **Slide 12**: Proposed Architectural Pipeline Overview *(Featuring System Pipeline Flowchart)*
13. **Slide 13**: Proposed Architecture — Multi-Branch Decomposition *(Featuring Multi-Branch Diagram)*
14. **Slide 14**: Proposed Component Specification (MTCNN, ResNeXt-50, Attention, BiLSTM)
15. **Slide 15**: Visual Attention Mechanism *(Featuring Attention Weight Distribution Heatmap)*
16. **Slide 16**: Interpretability & Forensic Attribution *(Featuring 4 Attribution Saliency Maps)*
17. **Slide 17**: Benchmark Datasets & Data Preparation Strategy (FF++, Celeb-DF v2, DFDC)
18. **Slide 18**: Data Integrity & Anti-Leakage Protocol (Subject-Group Split)
19. **Slide 19**: Robustness Evaluation Plan *(Featuring Compression & Degradation Curves)*
20. **Slide 20**: Planned Evaluation Metrics & Target Benchmark Criteria
21. **Slide 21**: Hardware Constraints & Implementation Feasibility (RTX 3050 6GB)
22. **Slide 22**: Planned Verification & Testing Protocol (TC-001 to TC-010)
23. **Slide 23**: Proposed Technology Stack & Frameworks
24. **Slide 24**: 16-Week Implementation Work Breakdown & Timeline
25. **Slide 25**: Expected Contributions & Academic Impact
26. **Slide 26**: Key Scholarly References & Q&A
