# APSIPA RADAR Challenge 2026 Baseline - SSL AASIST Anti-spoofing

This repository provides a baseline inference pipeline for the APSIPA RADAR Challenge 2026.

## Model Summary
- Architecture: Wav2Vec 2.0 SSL frontend with an AASIST classifier
- Training data: ASVspoof2019 LA
- Augmentation: RawBoost
- Model size: 300M

## Introduction
This repository contains inference code for APSIPA RADAR Challenge 2026, based on the [SSL AASIST Audio Deepfake Detection model](https://github.com/TakHemlata/SSL_Anti-spoofing) by Hemlata Tak.

Training and fine-tuning instructions are not included here. If you want to adapt or re-train the model for your own submission, please refer to the original repository.

## Getting Started
- Download the following checkpoints and place them in the `checkpoints` directory:
  - Wav2Vec 2.0 XLSR (300M): [xlsr2_300m.pt](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt)
  - Pretrained SSL AASIST checkpoint (trained on ASVspoof2019 LA): [Best_LA_model_for_DF.pth](https://drive.google.com/drive/folders/1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB)

- Extract `RADAR2026-dev.tar.gz` in the repository root. The expected structure is:
```
BASELINE-SSL_AASIST
├── checkpoints
│   ├── Best_LA_model_for_DF.pth
│   └── xlsr2_300m.pt
├── RADAR2026-dev
│   ├── flac
│   └── LICENSE
├── RADAR2026-dev.tar.gz
├── inference.py
├── LICENSE
├── model.py
├── README.md
└── run.sh
```

- Install dependencies:
  - `pip install -r requirements.txt`

- Run inference on a GPU machine:
  - `bash run.sh`
  - To run on CPU, set `DEVICE=cpu` in `run.sh`.

## Submission Score Format

WARNING: Submit the **fake score** only. A higher value should indicate a higher probability of being fake. Do **not** submit the real score.

- `run.sh` writes model outputs to `RADAR2026-dev/scores.txt` with format `<uttid> <fakescore> <realscore>`.
- It then extracts `<fakescore>` and writes the submission file `RADAR2026-dev-scores.txt` (this is the file to submit for the dev set).
- Example:
```
RADAR2026-DEV000001 4.594078540802002
RADAR2026-DEV000002 3.1839194297790527
RADAR2026-DEV000003 4.42523717880249
RADAR2026-DEV000004 1.0504467487335205
RADAR2026-DEV000005 4.370476722717285
...
```
- In this baseline, the submitted score is the raw fake score (no post-processing).
- If your system only outputs a real score, you can submit `-realscore` instead.