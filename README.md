# Winning Strategies in HYROX: A Machine Learning Approach to Race Performance Optimization

## Overview

This repository contains the data, analysis code, and paper for a comprehensive machine learning study of HYROX race performance. We analyze **58,852 athletes** across **58 international events** to identify the key factors that separate elite performers from the rest of the field.

Using dimensionality reduction (PCA, t-SNE, UMAP), unsupervised clustering (K-Means), and supervised learning (XGBoost, Elastic Net), we uncover actionable insights into how athletes can optimize their race strategy — revealing that **running segments**, particularly transitions (Roxzone), contribute disproportionately more to overall performance variation than workout stations.

> **Paper:** [arXiv preprint](https://arxiv.org/) *(link coming soon)*

## Project Structure

```
hyrox-race-analysis/
├── README.md
├── .gitignore
├── data/
│   └── events/              # 98 CSV files — raw race results (58,852 athletes)
├── analysis/
│   └── analysis_paper.py    # Main analysis script (PCA, clustering, XGBoost, SHAP, etc.)
├── paper/
│   ├── main.tex             # LaTeX source for the paper
│   ├── references.bib       # Bibliography
│   ├── build.sh             # Build script for compiling the paper
│   └── figures/             # 49 publication-quality figures
├── results/
│   └── analysis_results.json  # Numerical results from the analysis
└── output/                  # (gitignored) Local output directory
```

## Reproducing the Analysis

### Prerequisites

```bash
pip install pandas numpy scipy scikit-learn xgboost shap matplotlib seaborn umap-learn
```

### Run the analysis

```bash
python analysis/analysis_paper.py
```

This will process all event data and generate figures and results in the `output/` directory.

### Build the paper

```bash
cd paper
bash build.sh
```

## Data

The dataset comprises race results from 98 CSV files spanning 58 unique HYROX events worldwide (Season 7 and Season 8). Each record includes split times for all 8 running segments, 8 workout stations, and transition zones (Roxzones) for individual athletes.

**Key statistics:**
- 58,852 athletes analyzed
- 58 international events
- 16 timed segments per athlete (8 runs + 8 workouts)
- Events across 6 continents

## Key Findings

1. **Running dominates performance variance** — PCA reveals that running segments and Roxzone transitions explain the majority of variance in overall finish times.
2. **Four distinct athlete archetypes** — K-Means clustering identifies Runner-Dominant, Workout-Specialist, Balanced, and Developing profiles.
3. **Roxzone is the hidden differentiator** — SHAP analysis shows Roxzone time is the single most important predictor of finish time, yet is often overlooked in training.
4. **Diminishing returns on workouts** — Elastic Net regression confirms that beyond a baseline fitness level, further workout improvements yield smaller marginal gains than running improvements.

## Citation

If you use this dataset or analysis in your research, please cite:

```bibtex
@article{yamanoi2026hyrox,
  title={Winning Strategies in HYROX: A Machine Learning Approach to Race Performance Optimization},
  author={Yamanoi, Shuta},
  journal={arXiv preprint},
  year={2026}
}
```

## Author

**Shuta Yamanoi**

## License

This project is provided for academic and research purposes.
