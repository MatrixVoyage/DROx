# DeepGuard Test Plan & Proposed Dataset Documentation

This directory provides the formal test plan, dataset provenance details, and sample manifest specifications for the proposed DeepGuard deepfake video detection system.

## 1. Documentation Index

- **Test Plan & Data Specification (DOCX)**: [`DeepGuard_Test_Plan_and_Data_Documentation.docx`](DeepGuard_Test_Plan_and_Data_Documentation.docx)
- **Test Plan & Data Specification (PDF)**: [`DeepGuard_Test_Plan_and_Data_Documentation.pdf`](DeepGuard_Test_Plan_and_Data_Documentation.pdf)
- **Sample Distribution Manifest**: [`sample_manifest.csv`](sample_manifest.csv)

---

## 2. Proposed Datasets & Access Policy

To comply with academic licensing, intellectual property agreements, and storage constraints ($> 50$ GB aggregate), raw video binaries will be downloaded directly from official academic portals during the data acquisition phase:

| Dataset | Compression Tier | Planned Role | Official Source |
|---|---|---|---|
| **FaceForensics++ (FF++)** | C23 (H.264 HQ) | Primary in-domain training, validation & 4-fold Leave-One-Family-Out matrix | [TUM Computer Vision Group](https://github.com/ondyari/FaceForensics) |
| **Celeb-DF v2** | Standard | Zero-shot cross-dataset generalization benchmark | [Celeb-DF Project](https://github.com/yuezunli/celeb-deepfakeforensics) |
| **DFDC (Challenge)** | Kaggle / Meta | Independent cross-dataset benchmark (Subject to Kaggle credential authorization) | [Kaggle DFDC](https://www.kaggle.com/c/deepfake-detection-challenge) |

---

## 3. Sample Manifest Format

The included [`sample_manifest.csv`](sample_manifest.csv) illustrates the standardized metadata schema that the DeepGuard data loader will ingest:
- `video_id`: Unique video sequence identifier.
- `group_id`: Indivisible source identity key (guarantees zero train/test leakage across donor subjects).
- `video_path`: File path to the video container.
- `label`: Binary classification target (`0` = Pristine Real, `1` = Deepfake / Manipulated).
- `dataset_name`: Name of the source benchmark repository.
- `manipulation_type`: Specific synthesis algorithm (`pristine`, `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`).
- `split`: Planned partition assignment (`train`, `val`, `test`).
