# DeepGuard — Initial Academic Project Proposal Package

**System Title**: DeepGuard: Robust Spatiotemporal Visual Deepfake Video Detection Platform  
**Target Group**: IVP Group 10 Academic Submission  
**Academic Status**: Initial Project Proposal & Engineering Specification (Pre-Implementation Baseline)  
**Date**: October 15, 2026  
**Primary Reference Paper**: Lal, M., Shiwani, S., & Gandhi, T. (2025). *Deepfake video detection using deep learning with visual attention mechanism*. **Scientific Reports (Nature)**, 15(1), 1–15. DOI: [10.1038/s41598-025-23920-0](https://doi.org/10.1038/s41598-025-23920-0)

---

## 1. Directory Structure

```text
submission/
├── Base_Papers/
│   ├── Lal_2025_Deepfake_Visual_Attention_Base_Paper.pdf  # Primary Nature Scientific Reports (2025) paper
│   ├── primary_reference.md                              # Mathematical analysis & motivation notes
│   └── README.md                                         # Reference attribution policy
├── Literature_Survey/
│   ├── DeepGuard_Literature_Survey.xlsx                  # 4-sheet structured survey (18 papers)
│   ├── DeepGuard_Literature_Survey.csv                   # CSV survey data export
│   ├── references.md                                     # Formatted IEEE scholarly references
│   └── papers/                                           # 16 peer-reviewed academic PDFs
├── PPT/
│   ├── DeepGuard_Project_Presentation.pptx               # 26-slide presentation with 15 embedded figures
│   └── DeepGuard_Project_Presentation.pdf                # Archival PDF export
├── SRS_and_Design_Report/
│   ├── DeepGuard_SRS.docx                                # 9-section formal SRS & System Design document
│   ├── DeepGuard_SRS.pdf                                 # Archival PDF export
│   └── DeepGuard_Architecture.md                         # Proposal architecture specification & Mermaid diagrams
├── Test_Data/
│   ├── DeepGuard_Test_Plan_and_Data_Documentation.docx   # Proposed test plan & data specification
│   ├── DeepGuard_Test_Plan_and_Data_Documentation.pdf    # Archival PDF export
│   ├── sample_manifest.csv                               # Canonical manifest schema for dataset curation
│   └── README.md                                         # Data preparation and anti-leakage guidelines
└── Validation/
    └── submission_checklist.md                            # Comprehensive 10-rule proposal compliance matrix
```

---

## 2. Deliverables Summary

1. **SRS and System Design**:
   - Comprehensive academic Software Requirements Specification formulated in future tense.
   - 16 Functional Requirements (`FR-001` through `FR-016`) and 8 Non-Functional Requirements (`NFR-001` through `NFR-008`).
   - Modular architecture specification: Video Ingestion -> MTCNN Face Detection -> 112x112 Cropping -> ResNeXt-50 Spatial Feature Extraction -> Spatial Attention Refinement -> BiLSTM Temporal Aggregation -> Classification Head.
   - Dual formats: Editable DOCX and Archival PDF.

2. **Project Presentation**:
   - 26 widescreen (16:9) slides covering project motivation, threat taxonomy, comprehensive literature survey, research gaps, problem statement, objectives, proposed multi-branch architecture, attention mechanism, forensic attribution, datasets, test plan, hardware feasibility on RTX 3050 (6GB VRAM), and 16-week timeline.
   - 15 authentic visual diagrams, benchmark curves, and facial manipulation crops embedded.
   - Dual formats: PPTX and PDF.

3. **Literature Survey Workbook**:
   - 18 peer-reviewed research papers analyzed across 4 dedicated sheets:
     - `Literature Survey`: Detailed breakdown of methodology, datasets, reported metrics, strengths, and limitations.
     - `Top 5 Comparative Analysis`: In-depth side-by-side comparison of leading architectures.
     - `Dataset Benchmark Mapping`: Cross-dataset evaluation mapping across FaceForensics++ C23 and Celeb-DF v2.
     - `References`: Complete IEEE citations with DOIs.
   - Dual formats: XLSX and CSV.

4. **Base Papers**:
   - Primary reference paper PDF (*Lal et al., Nature Scientific Reports 2025*) included.
   - Extensive scholarly analysis in `primary_reference.md` distinguishing published reference metrics from proposed project goals.

5. **Test Plan and Test Data Documentation**:
   - Pre-implementation test strategy document specifying 10 planned test cases (`TC-001` through `TC-010`).
   - Anti-leakage donor-stratified partitioning rules (zero identity overlap between splits).
   - Sample manifest CSV illustrating proposed metadata schema.
   - Dual formats: DOCX and PDF.

---

## 3. Compliance and Academic Integrity

- **Pre-Implementation Formulation**: All deliverables reflect an initial proposal stage. Language is strictly future-oriented (*will develop, is proposed to, will evaluate, is expected to*).
- **Separation of Results**: Published reference metrics from literature are clearly labeled as baseline references and never conflated with DeepGuard project milestones.
- **Hardware Realism**: All proposed training schedules, batch configurations, and inference latencies are tailored to the available NVIDIA RTX 3050 (6 GB VRAM) platform.
