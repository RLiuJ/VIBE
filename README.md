# VIBE

[![License: MIT](https://img.shields.io/github/license/RLiuJ/VIBE?cacheSeconds=300)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/RLiuJ/VIBE)](https://github.com/RLiuJ/VIBE/commits)
[![Open Issues](https://img.shields.io/github/issues/RLiuJ/VIBE)](https://github.com/RLiuJ/VIBE/issues)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)

Predict a continuous behavioral stress score (z-score) from four implanted-sensor
physiological signals in rats, and apply the trained model to a continuous test
recording with a sliding window.

## Table of Contents

- [Overview](#overview)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [How It Works](#how-it-works)
- [Evaluation](#evaluation)
- [License](#license)

## Overview

Each training recording yields one labeled example: multi-channel physiology
paired with a single behavioral z-score. A regression model is trained on
hand-engineered statistical features and evaluated with leave-one-out
cross-validation across recordings. The selected model is then re-fit on all
training data and applied to a held-out continuous test recording in sliding
windows to produce a time-resolved stress trace.

Signals used (per recording):

- Blood Pressure (mmHg)
- Heart Rate (BPM)
- Glucose (mg/dL)
- Breathing Period (s)

## Repository Layout

```
.
├── main.py                      # End-to-end pipeline entry point
├── config.py                    # Paths, model list, sliding-window settings
├── dataInterface/
│   ├── dataLoader.py            # Reads training xlsx + continuous test xlsx
│   └── featureExtractor.py      # Per-signal summary statistics (10 per channel)
├── machineLearning/
│   ├── models.py                # Model registry
│   ├── preprocessing.py         # Leave-one-out splitter
│   ├── trainer.py               # CV evaluation + final fit
│   └── predictor.py             # Inference
├── plottingHelpers/
│   └── plotter.py               # Figure plotting
├── data/
│   ├── raw/                     # Per-recording xlsx training files
│   └── test_data.xlsx           # Continuous multi-channel test recording
├── figures/                     # Output PDFs
├── artifacts/                   # Pickled model + scaler + feature columns
└── test_predictions.xlsx        # Sliding-window predictions on test data
```

## Requirements

### Hardware

A standard computer with enough RAM to support in-memory operations. No GPU is required.

### Software

- Python 3.12
- All Python dependencies are listed in [`Requirements.txt`](Requirements.txt).

Tested on Windows and macOS.

## Installation

Installation on a standard computer takes approximately 1 minute.

```bash
# 1. Clone the repository
git clone https://github.com/RLiuJ/VIBE.git

# 2. Move into the project directory
cd VIBE

# 3. (Optional) create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

# 4. Install dependencies
pip install -r Requirements.txt
```

## Usage

Run the full pipeline:

```bash
python main.py
```

A complete run on the provided test data takes under 1 minute.

## Outputs

- `artifacts/` — pickled model, scaler, and feature columns.
- `test_predictions.xlsx` — sliding-window predictions on the test recording.
- `figures/` — output PDF figures.

## How It Works

`main.py` runs the pipeline end to end:

1. Loads all `data/raw/*.xlsx` training recordings.
2. Builds the feature matrix.
3. Runs leave-one-out cross-validation for every model in `Config.modelNames`.
4. Re-fits `Config.bestModel` on all data and saves `artifacts/`.
5. Runs sliding-window inference on `data/test_data.xlsx`.
6. Writes `test_predictions.xlsx`.
7. Renders all figures.

## Evaluation

- Leave-one-out CV across recordings (`machineLearning/preprocessing.py`).
- Reported metrics: `MAE`, Pearson `r`, `R²`, sign accuracy, three-class accuracy.
- Three-class buckets: `healthy` (z ≥ 0), `lowStress` (-2 < z < 0),
  `highStress` (z ≤ -2). Thresholds are defined in `trainer.py` and `predictor.py`.

## License

This repository is licensed under the MIT License. For research use only.
Please cite our work if used in academic publications.