# VERITAS-T

**Verifiable Error-Risk Integrated Truth Assessment Score with Temporal Severity Decay**

A risk-aware factuality evaluation framework for large language models in high-stakes domains (medical, legal, financial). VERITAS-T scores claims using domain criticality (C), severity (S), factual correctness (F), and prediction-confidence variance (M), with an optional temporal decay extension for claim age.

## Overview

VERITAS-T extends the original VERITAS framework with a temporal decay term, so severity is discounted as a claim ages. The framework is evaluated on three real benchmarks:

- **Medical** — ACI-Bench (clinical note hallucination labels)
- **Legal** — LegalBench (contract Q&A, Yes/No)
- **Finance** — FinanceBench (SEC filing Q&A with hallucination labels)
- **Note on `finance_data.csv`:** This file is not included in the repository due to size constraints. It is derived from the FinanceBench dataset. To reproduce, obtain the FinanceBench Q&A data from its original source and place it at `data/finance_data.csv` with columns: `query`, `context`, `answer`, `ground_truth_label`.

across two independently run language models (Qwen2.5-7B-Instruct, Llama-3-8B-Instruct), totaling 2,796 evaluated claims with K=50 sampling per claim.

## Repository Structure

```
veritas-t/
├── README.md
├── requirements.txt
├── data/
│   ├── medical_labels.csv
│   ├── medical_outputs.csv
│   ├── legal_train.tsv
│   ├── legal_test.tsv
│   └── finance_data.csv
├── scripts/
│   ├── veritas_t_main.py          # Main pipeline: loads data, computes F/C/S/M, runs both models
│   ├── legal_rerun.py              # Legal F-derivation fix (generate-and-check vs ground truth)
│   ├── finance_m_fix.py            # Targeted fix script for an interrupted-run gap
│   └── analysis.py                 # Statistical analysis: Mann-Whitney U, Cliff's delta, CIs, decomposition
├── results/
│   ├── all_scored_qwen2.5-7b.csv
│   ├── all_scored_llama3-8b.csv
│   ├── primary_stats_results.csv
│   ├── component_decomposition.csv
│   └── legal_error_comparison.csv
└── notebooks/
    └── veritas_t_kaggle_run.ipynb  # Original Kaggle execution notebook
```

## Setup

```bash
pip install -r requirements.txt
```

GPU with at least 16GB VRAM is recommended (Kaggle T4 or equivalent). A HuggingFace token with access to `meta-llama/Meta-Llama-3-8B-Instruct` is required for the Llama runs (gated model); Qwen2.5-7B-Instruct is openly accessible.

## Running the Pipeline

```bash
python scripts/veritas_t_main.py
```

Key configuration variables (top of file):
- `K` — number of samples per claim for variance computation (paper default: 50)
- `MAX_PER_DOMAIN` — cap on claims processed per domain
- `HF_MODEL_ID` — model to run (Qwen2.5-7B-Instruct or Llama-3-8B-Instruct)

The pipeline checkpoints progress every 30 minutes or 200 claims, whichever comes first, and resumes automatically from the last checkpoint if interrupted.

## Method Notes

- **Factual correctness (F)**: derived from each benchmark's own ground-truth labels (Medical, Finance) or, for Legal, by having the model answer each contract question and comparing against the real Yes/No ground truth.
- **Severity (S)**: keyword-based heuristic per domain, cross-checked against an independent LLM severity judgment (see `severity_agreement_*.csv`).
- **Confidence variance (M)**: computed from embedding variance across K=50 sampled generations per claim.
- **Temporal decay (t)**: real SEC filing dates extracted from FinanceBench context where available; simulated domain-realistic ranges for Medical and Legal (no date fields exist in those source files).
- Baseline metrics (FActScore, RAGAS, VeriScore, etc.) are simplified proxies derived from F and M, not full reimplementations of the original published metrics.

## Citation

If you use this work, please cite the associated paper (citation details to be added upon publication).

## License

To be determined upon publication.
