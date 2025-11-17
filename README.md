# Loan Default Modeling: Interpretability, Stability, and Fairness

This repository contains a reproducible workflow to study model interpretability, stability, and fairness on a loan default dataset. 
The version on the main branch is the original group work that was done; the most updated and improved version is on the branch called "my_version". 
It includes:
- Data loading and preprocessing
- A surrogate model (Decision Tree) to approximate the provided default probabilities
- A high-performance black-box model (XGBoost) trained on the true target
- Utilities to export notebooks to PDF and clear, human-friendly documentation

## Project Structure

```
Interpretability_project/
  ├─ Data/
  │   ├─ dataproject2025.csv           # Not committed (large); place here locally
  │   └─ Data description_project_DSAIB_2025.xlsx
  ├─ data_preprocessing.ipynb          # Preprocessing, EDA, encoding, targets
  ├─ Group_9_Interpretability_project.ipynb  # Main analysis notebook (if applicable)
  ├─ Interpretability Stability and Fairness DSAIB Project Fall 2025.pdf  # Project brief
  ├─ README.md                         # You are here
  ├─ requirements.txt                  # Python dependencies
  ├─ .gitignore                        # Excludes large/derived artifacts
  └─ scripts/
      └─ export_pdf.sh                 # Export notebooks to PDF via nbconvert (webpdf)
```

## Data

- The dataset CSV is expected at `Interpretability_project/Data/dataproject2025.csv`.
- The CSV is large and is intentionally not versioned. Please copy it locally to the `Data/` folder before running the notebooks.

## Environment Setup

1) Recommended: Python 3.10+
2) Install dependencies:

```bash
pip install -r requirements.txt
```

If you plan to export notebooks to PDF, the script uses nbconvert’s webpdf exporter (headless Chromium via pyppeteer). The first run may download Chromium automatically.

## How to Run

1) Open Jupyter and run the preprocessing notebook:

```bash
jupyter notebook
```

Then open and execute:
- `data_preprocessing.ipynb`
- `Group_9_Interpretability_project.ipynb` (if present)

The preprocessing notebook performs:
- Column cleanup and missing value imputation (median for numeric, mode for categorical)
- One-hot encoding for selected categorical features with rare-category grouping
- Definition of `X`, `y_dp` (provided probabilities), and `y_true` (target)
- A Decision Tree surrogate model to understand the provided probabilities
- An XGBoost classifier (black-box model) trained on `y_true`, with evaluation

## Export to PDF

Use the helper script to export notebooks to PDF via webpdf:

```bash
bash scripts/export_pdf.sh
```

This will produce PDF files next to the notebooks. If the export fails due to system-level browser/permissions, re-run after `pip install -r requirements.txt` and ensure network access for the initial Chromium download.

## Results and Artifacts Policy

- Do not commit the large CSV file(s): they are excluded via `.gitignore`.
- PNG plots and PKL model files are also excluded by default to keep the repo lightweight.
- PDF reports are kept under version control for easy sharing.

## Reproducibility

- Random seeds are set where relevant (e.g., Decision Tree and XGBoost) for stability.
- For fully deterministic runs, ensure consistent package versions from `requirements.txt`.

## Troubleshooting

- If `xgboost` fails to compile or import wheels on your platform, try upgrading `pip` and reinstalling:
  ```bash
  python -m pip install --upgrade pip
  pip install --force-reinstall xgboost
  ```
- If nbconvert webpdf fails, try:
  ```bash
  jupyter nbconvert --to html data_preprocessing.ipynb
  ```
  (as a fallback HTML export) or install a full LaTeX distribution and use `--to pdf`.

## License

Educational use. Adapt as needed for your coursework or research.
