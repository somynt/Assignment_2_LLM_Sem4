# Toxic Comment Classification with ELECTRA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/somynt/Assignment_2_LLM_Sem4/blob/main/Assignment_2_LLM.ipynb)

Fine-tuning a transformer language model (ELECTRA) to flag six kinds of toxicity in online comments, with tuned decision thresholds and word-level explanations of each prediction.

**Skills shown:** NLP · transformer fine-tuning (Hugging Face) · PyTorch · imbalanced multi-label classification · hyperparameter optimisation (Optuna) · explainable AI (LIME) · model evaluation

---

## Results

Evaluated on a held-out 20% validation split of the Jigsaw training data.

| Metric | Default 0.5 thresholds | Tuned per-label thresholds |
|---|---|---|
| Macro F1 | 0.668 | **0.679** |
| Macro ROC AUC | 0.986 | — |

Per-label performance (default thresholds):

| Label | F1 | ROC AUC | PR AUC |
|---|---|---|---|
| toxic | 0.82 | 0.985 | 0.90 |
| obscene | 0.82 | 0.992 | 0.89 |
| insult | 0.76 | 0.987 | 0.78 |
| identity_hate | 0.58 | 0.982 | 0.56 |
| threat | 0.54 | 0.979 | 0.46 |
| severe_toxic | 0.48 | 0.989 | 0.42 |

**What this shows:** the model ranks comments very well (ROC AUC ≈ 0.98–0.99 on every label), and is reliable on the common labels. The rare labels (threat 0.3%, identity_hate 0.9%, severe_toxic 1.0% of comments) are much harder, which is why PR AUC and F1 drop there. Macro F1 is used as the headline metric because it weights every label equally instead of letting the common labels dominate.

## The problem

The [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset contains Wikipedia talk-page comments labelled by human raters. A comment can carry any combination of six labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. The classes are heavily imbalanced, so a naive model scores well by predicting "not toxic" for everything.

## Approach

1. **Model:** `google/electra-base-discriminator` fine-tuned for multi-label classification (one sigmoid output per label), max length 256 tokens.
2. **Class imbalance:** positive-class weighting in the loss (`BCEWithLogitsLoss` with `pos_weight`), plus SMOTETomek resampling of training folds.
3. **Hyperparameter search:** Optuna (TPE sampler, Hyperband pruning) with 5-fold cross-validation, maximising macro F1.
4. **Threshold tuning:** a second Optuna search picks a separate decision threshold for each label, lifting macro F1 from 0.668 to 0.679.
5. **Explainability:** LIME highlights the words that pushed each prediction up or down, so individual decisions can be checked by a person.

## How to run

The notebook is built for **Google Colab with a GPU** (Runtime → Change runtime type → GPU).

1. Click the *Open in Colab* badge above.
2. **Use your own Kaggle API token.** On Kaggle go to *Settings → API → Create New Token*; this downloads `kaggle.json`. Upload it to the Colab session when the notebook asks. Never commit this file to GitHub.
3. You must accept the competition rules on the Kaggle competition page before the data download works.
4. Run the cells in order. The notebook mounts Google Drive and saves models, plots and the Optuna study to `Config.GOOGLE_DRIVE_SAVE_BASE_PATH`. Change that path to save somewhere else.

To run locally instead, install the packages in `requirements.txt`, place `train.csv`, `test.csv`, `test_labels.csv` and `sample_submission.csv` next to the notebook, skip the Colab/Drive cells, and set `GOOGLE_DRIVE_SAVE_BASE_PATH` to a local folder.

## Repository contents

| File | Purpose |
|---|---|
| `Assignment_2_LLM.ipynb` | Full pipeline: data download, EDA, training, tuning, evaluation, LIME explanations |
| `requirements.txt` | Python packages used |

## Limitations and next steps

- The hyperparameter search ran only 5 Optuna trials because of GPU time limits; a longer search would likely improve the rare labels.
- Results are on a validation split, not the official Kaggle private leaderboard.
- Next: try class-specific data augmentation for the rare labels, and package the model behind a small API for real-time moderation.

---

*MSc Data Science, University of Hertfordshire.*
