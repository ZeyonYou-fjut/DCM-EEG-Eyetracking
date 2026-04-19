# DCM-DNN: Discrete Choice Model-Inspired Deep Neural Network for EEG-Based Consumer Decision Prediction

## Overview

DCM-DNN integrates Discrete Choice Model (DCM) theory with deep neural networks for predicting consumer purchase decisions from EEG signals. The model achieves **84.06% accuracy** on the NeuMa dataset, matching the state-of-the-art CNN-LSTM+Ensemble (84.01%) while providing theoretically grounded interpretability through DCM utility functions.

Full paper title: *Theory-Constrained Multimodal Fusion for Neural Consumer Decision Prediction: Integrating Discrete Choice Models with Deep Learning*

## Key Features

- Theory-driven architecture based on Mixed Logit discrete choice model
- EEG feature extraction (FAA, FTA, P300, LPP, N200, PLV, etc.)
- Alternative-Specific Constants (ASC) for individual heterogeneity
- Edge deployment capability (43.3 KB INT8 quantized model)
- Comprehensive evaluation: ablation, robustness, TinyML feasibility

## Dataset

**NeuMa (Neuromarketing) dataset** — 42 participants in supermarket brochure browsing scenario.

- Original: https://openneuro.org/datasets/ds004588
- Preprocessed: https://figshare.com/articles/dataset/NeuMa_PreProcessed_A_multimodal_Neuromarketing_dataset/22117124

> **Note:** Data files are not included in this repository. Please download from the links above and place in the `data/` directory.

## Requirements

- Python 3.8+
- PyTorch >= 1.10
- scikit-learn >= 1.0
- numpy >= 1.20
- pandas >= 1.3
- scipy >= 1.7
- matplotlib >= 3.4

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
eeg-project/
├── src/                          # Core model code
│   ├── dcm_dnn_eeg.py           # DCM-DNN model architecture
│   ├── data_loader.py           # NeuMa dataset loader
│   └── eeg_features.py          # EEG feature extraction
├── experiments/                  # Experiment scripts
│   ├── run_baseline_compare.py  # Baseline comparison (Table 2)
│   ├── run_main_experiment.py   # Main DCM-DNN training
│   ├── run_ablation.py          # Ablation study (Table 3)
│   ├── run_robustness.py        # Statistical robustness tests
│   ├── run_tinyml.py            # TinyML edge deployment (Table 4)
│   ├── run_final_validation.py  # Final validation
│   └── generate_figures.py      # Paper figure generation
├── results/                      # Experiment results (JSON)
├── paper/                        # LaTeX manuscript and figures
├── requirements.txt
└── LICENSE
```

## Quick Start

```bash
# 1. Download NeuMa dataset and place in data/
# 2. Run main experiment
python experiments/run_main_experiment.py
# 3. Run baseline comparison
python experiments/run_baseline_compare.py
```

## Evaluation Protocol

- 41 participants (1 excluded due to data quality)
- Stratified 10-fold cross-validation (seed=42)
- 396 valid trials

## Results Summary

| Model | Accuracy (%) | Parameters |
|-------|-------------|-----------|
| LR | 82.58 ± 6.57 | — |
| SVM (RBF) | 79.77 ± 5.39 | — |
| Simple MLP | 81.29 ± 6.39 | 6,498 |
| Matched MLP | 82.10 ± 5.86 | 19,362 |
| CNN-LSTM+Ensemble (SOTA) | 84.01 | — |
| **DCM-DNN (Ours)** | **84.06 ± 6.28** | **13,588** |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{you2025dcmdnn,
  title   = {Theory-Constrained Multimodal Fusion for Neural Consumer Decision Prediction:
             Integrating Discrete Choice Models with Deep Learning},
  author  = {You, Ziyang and He, Huilong and Yang, Xiaoke},
  journal = {},
  year    = {2025}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
