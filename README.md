# Robust Prediction Learning under Temporal Behavioral Shift

An independent machine-learning research study investigating how an
online algorithm should use historical ML predictions when user behavior
changes over time.

The project uses timestamped Last.fm listening histories as a real-world
testbed for studying:

- temporal distribution shift;
- implicit-feedback recommendation;
- historical predictor degradation;
- online learning;
- expert aggregation;
- algorithms with predictions; and
- adaptive prediction trust.

The central research question is:

> **Can behavioral distribution shift be used to determine when a
> sequential learning algorithm should reduce its trust in a historical
> machine-learning predictor?**

---

## Research status

The main experimental pipeline is complete.

| Milestone | Status |
|---|---|
| Data preprocessing and chronological splitting | Complete |
| Popularity and BPR recommendation baselines | Complete |
| Temporal behavioral-shift analysis | Complete |
| Robust sequential prediction combination | Complete |
| Scientific manuscript | Draft complete |

Git checkpoints are maintained for the completed experimental stages,
including `milestone-2`, `milestone-3`, and `milestone-4`.

---

# Main findings

## 1. Behavioral shift is associated with historical predictor degradation

Using equal-interaction future windows, behavioral shift was measured
using Jensen–Shannon divergence between a user's historical artist
distribution and later listening distributions.

For the primary 250-interaction window setting:

- 595 users were analyzed;
- 6,642 temporal windows were evaluated;
- mean Jensen–Shannon divergence was `0.642`;
- pooled Spearman correlation between shift and BPR EventMass@20 was
  `-0.359`;
- median within-user Spearman correlation was `-0.252`;
- 70.1% of users with valid within-user correlations had a negative
  relationship.

The result remained qualitatively stable across window sizes:

| Window size | Users | Windows | Overall Spearman | Median user Spearman | Users with negative correlation |
|---:|---:|---:|---:|---:|---:|
| 125 | 736 | 14,153 | -0.383 | -0.284 | 74.0% |
| 250 | 595 | 6,642 | -0.359 | -0.252 | 70.1% |
| 500 | 405 | 2,848 | -0.334 | -0.300 | 66.1% |

The result is interpreted as an **association**, not as evidence that
distribution shift causally produces prediction errors.

---

## 2. Historical prediction alone is not robust

A BPR recommender trained on historical implicit feedback becomes weak
on later sequential windows.

In the final robust-learning experiment:

| Method | Mean loss | Expected EventMass@20 | Mean regret |
|---|---:|---:|---:|
| **Online Only** | **0.6421** | **0.3579** | **0.0004** |
| Hedge | 0.6815 | 0.3185 | 0.4179 |
| Static Mixture | 0.7158 | 0.2842 | 1.5119 |
| Shift-Aware Trust | 0.7599 | 0.2401 | 2.2557 |
| Prediction Only | 0.9369 | 0.0631 | 6.0464 |

The strongest method in the current experimental setting is therefore
the lightweight adaptive online expert.

Among methods that combine the historical and online predictors, Hedge
performs best.

---

## 3. Explicit shift-aware trust is informative but insufficient

The proposed candidate algorithm adjusts historical trust according to:

1. behavioral Jensen–Shannon divergence; and
2. exponentially smoothed relative loss between the historical and
   online experts.

Its historical weight is

\[
\alpha_t
=
\sigma\left(
\operatorname{logit}(\alpha_0)
-
\gamma_s s_t
-
\gamma_d d_t
\right).
\]

The method performs substantially better than relying exclusively on
the historical BPR predictor, but it does not outperform Hedge or
Online Only.

Inspection of the learned trust trajectory indicates why: the selected
shift-aware rule retains approximately 40% historical trust through
much of the test stream, whereas Hedge rapidly reduces historical trust
toward zero.

This negative result is retained deliberately. It shows that:

> **detecting behavioral shift and responding optimally to behavioral
> shift are separate algorithmic problems.**

---

# Dataset

The project uses the **Last.fm-1K Music Recommendation Dataset**,
collected by Òscar Celma.

Dataset reference:

> Òscar Celma. Last.fm Music Recommendation Dataset, version 1.2.
> DOI: `10.5281/zenodo.6090214`.

The original dataset is made available for non-commercial research use.
It is **not redistributed in this repository**.

After downloading and extracting the dataset, the expected local layout
is:

```text
data/
├── raw/
│   ├── lastfm-dataset-1K.tar.gz
│   └── lastfm-dataset-1K/
│       ├── README.txt
│       ├── userid-profile.tsv
│       └── userid-timestamp-artid-artname-traid-traname.tsv
│
└── processed/
```

The large raw and processed data files are excluded from Git.

---

# Preprocessed data

The preprocessing pipeline produces:

```text
data/processed/
├── interactions.parquet
├── train.parquet
├── validation.parquet
└── test.parquet
```

Observed dataset statistics:

| Statistic | Value |
|---|---:|
| Raw interactions | 19,098,853 |
| Processed interactions | 18,521,447 |
| Users | 966 |
| Artists | 41,045 |
| Training interactions | 12,964,572 |
| Validation interactions | 1,852,070 |
| Test interactions | 3,704,805 |

The chronological split is performed independently for every user:

```text
70% training
10% validation
20% testing
```

The code explicitly validates that no future interaction appears in an
earlier per-user split.

---

# Repository structure

```text
robust-prediction-learning/
│
├── README.md
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── experiments/
│   ├── 01_dataset_analysis.py
│   ├── 02_baselines.py
│   ├── 03_temporal_shift.py
│   └── 04_robust_learning.py
│
├── paper/
│   ├── main.tex
│   └── references.bib
│
├── results/
│   ├── figures/
│   └── tables/
│
├── src/
│   └── robust_prediction_learning/
│       ├── __init__.py
│       ├── algorithms.py
│       ├── data.py
│       ├── metrics.py
│       ├── recommenders.py
│       ├── shift.py
│       └── utils.py
│
└── tests/
    └── test_core.py
```

---

# Environment

The project is developed and tested using:

```text
Python 3.12
```

A project-local virtual environment is recommended.

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the installation:

```powershell
python -c "import robust_prediction_learning; print('Import OK')"
```

---

# Unit tests

Run:

```powershell
pytest -v
```

The completed experimental implementation currently contains 25 core
tests covering:

- preprocessing;
- chronological splitting;
- temporal leakage;
- ranking metrics;
- cumulative loss and regret;
- popularity recommendation;
- BPR fitting and exclusion logic;
- Jensen–Shannon divergence;
- sparse shift computation;
- EventMass@K;
- mixture loss;
- Static Mixture;
- Hedge;
- Shift-Aware Trust; and
- the adaptive online expert.

Expected result:

```text
25 passed
```

---

# Reproducing the study

## Milestone 1 — Dataset preprocessing and exploratory analysis

Run:

```powershell
python experiments/01_dataset_analysis.py
```

Generated processed files:

```text
data/processed/interactions.parquet
data/processed/train.parquet
data/processed/validation.parquet
data/processed/test.parquet
```

Generated exploratory outputs include:

```text
results/figures/interactions_per_user.png
results/figures/artist_popularity.png
results/figures/interactions_over_time.png

results/tables/dataset_statistics.json
```

---

## Milestone 2 — Recommendation baselines

Run:

```powershell
python experiments/02_baselines.py
```

This experiment:

1. fits a global popularity baseline;
2. performs a small validation search for BPR;
3. selects BPR parameters using validation NDCG@20;
4. refits using training plus validation data; and
5. evaluates once on the chronological test split.

Principal outputs:

```text
results/tables/bpr_validation_search.csv
results/tables/selected_bpr_config.json
results/tables/baseline_results.csv

results/figures/baseline_comparison.png
```

Selected BPR parameters:

```text
latent factors            64
learning rate             0.05
regularization            0.005
epochs                    5
batch size                8192
samples per epoch         500000
```

---

## Milestone 3 — Temporal distribution shift

Primary experiment:

```powershell
python experiments/03_temporal_shift.py --window-size 250
```

Sensitivity experiments:

```powershell
python experiments/03_temporal_shift.py --window-size 125
python experiments/03_temporal_shift.py --window-size 500
```

The primary shift measure is:

\[
JS(P_u^{history}, P_{u,t}^{future}).
\]

Equal-interaction windows are used instead of fixed calendar windows to
reduce confounding from dataset-level observation density changes.

Principal outputs:

```text
results/tables/temporal_window_metrics.csv
results/tables/per_user_shift_correlations.csv
results/tables/shift_quality_bins.csv
results/tables/temporal_shift_summary.json

results/figures/temporal_shift_distribution.png
results/figures/shift_vs_prediction_quality.png
results/figures/predictor_quality_over_windows.png
results/figures/shift_by_user.png
```

---

## Milestone 4 — Robust prediction combination

Run:

```powershell
python experiments/04_robust_learning.py
```

The experiment compares:

```text
Prediction Only
Online Only
Static Mixture
Hedge
Shift-Aware Adaptive Trust
```

Validation is used to select robust-learning parameters.

The final historical model is trained on:

```text
training + validation
```

and the final comparison is performed on the chronological test stream.

Principal outputs:

```text
results/tables/robust_learning_validation.csv
results/tables/selected_robust_parameters.json
results/tables/robust_learning_test.csv
results/tables/robust_learning_per_user.csv
results/tables/robust_learning_windows.csv
```

Figures:

```text
results/figures/robust_learning_comparison.png
results/figures/cumulative_loss.png
results/figures/trust_over_time.png
results/figures/performance_by_shift.png
```

---

# Evaluation measures

## Recall@K

\[
\operatorname{Recall@K}
=
\frac{
|\text{recommended}_{K}
\cap
\text{relevant}|
}{
|\text{relevant}|
}.
\]

## NDCG@K

Binary-relevance normalized discounted cumulative gain is used to
measure ranking quality.

## EventMass@K

For sequential music consumption, repeated listening events are
meaningful. EventMass@K therefore measures:

\[
\operatorname{EventMass@K}
=
\frac{
\text{future listening events covered by the top-K artists}
}{
\text{all future listening events}
}.
\]

## Sequential loss

For an expert:

\[
\ell_t
=
1
-
\operatorname{EventMass@K}_t.
\]

## Regret

The robust-learning experiment also measures cumulative loss relative to
the better fixed expert.

---

# Experimental protocol

Several methodological constraints are enforced throughout the study.

### Chronological evaluation

Future interactions are never randomly mixed into historical training
data.

### Validation-only hyperparameter selection

BPR and robust-learning parameters are selected using validation data.

### No current-window look-ahead

During robust sequential evaluation:

```text
predict
→ evaluate current window
→ update expert losses
→ update online model
→ proceed to next window
```

The current window is therefore not used before its prediction is
evaluated.

### User-level shift analysis

Pooled temporal windows are not treated as independent observations for
the main interpretation. Per-user Spearman correlations are computed to
account for repeated windows from the same listener.

---

# Important limitations

This study should not be interpreted as a state-of-the-art music
recommendation benchmark.

Key limitations include:

- one historical music dataset;
- active-user selection induced by equal-interaction windows;
- artist-level rather than track-level modeling;
- normalized artist names as operational item identifiers;
- observational rather than causal distribution-shift analysis;
- a deliberately simple adaptive online expert;
- expected mixture loss rather than deterministic score fusion;
- only one initial shift-aware trust formulation; and
- no claim that the current shift-aware algorithm is theoretically
  optimal.

The negative robust-learning result is retained rather than hidden:
Hedge and Online Only outperform the initial Shift-Aware Adaptive Trust
method.

---

# Paper

The research manuscript is contained in:

```text
paper/main.tex
paper/references.bib
```

To compile with a standard LaTeX installation:

```powershell
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Alternatively, with `latexmk` installed:

```powershell
cd paper
latexmk -pdf main.tex
```

The generated PDF should remain local or be added to the repository only
if desired.

---

# Research interpretation

The study does **not** conclude that behavioral distribution shift
causes recommendation failure.

The evidence supports the narrower conclusion:

> Larger divergence from historical listening behavior is consistently
> associated with lower quality from a frozen historical predictor.

The subsequent robust-learning experiment further shows:

> Knowing that behavior has shifted is not sufficient by itself to
> determine the optimal amount of trust that should remain in historical
> predictions.

This distinction motivates future work on theoretically grounded,
locally adaptive algorithms with predictions.

---

# Future work

Potential extensions include:

- context-dependent rather than global trust;
- local item-region reliability estimation;
- change-point-aware expert resets;
- regret guarantees involving prediction error or number of shifts;
- adaptive window sizes;
- user-cluster-specific trust calibration;
- additional recommendation datasets;
- stronger sequential online experts;
- deterministic score-space fusion; and
- bootstrap or hierarchical inference at the user level.

---

# References

Core references include:

- Rendle et al. (2009), *BPR: Bayesian Personalized Ranking from
  Implicit Feedback*.
- Freund and Schapire (1997), *A Decision-Theoretic Generalization of
  On-Line Learning and an Application to Boosting*.
- Lin (1991), *Divergence Measures Based on the Shannon Entropy*.
- Gama et al. (2014), *A Survey on Concept Drift Adaptation*.
- Lykouris and Vassilvitskii (2018), *Competitive Caching with Machine
  Learned Advice*.
- Mitzenmacher and Vassilvitskii (2022), *Algorithms with Predictions*.
- Celma (2010), *Music Recommendation and Discovery*.

Complete BibTeX entries are provided in:

```text
paper/references.bib
```

---

# License and dataset note

Source-code licensing and third-party dataset licensing should be
treated separately.

The Last.fm dataset is not included in this repository and remains
subject to the conditions specified by its original distributor.

This repository contains only the code, derived small result summaries,
figures, tests, and research manuscript required to reproduce the study
once the dataset has been obtained separately.
