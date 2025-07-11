 Interpretable Multi-Label Toxic Comment Classification with ELECTRA
This repository contains Assignment_2_LLM.ipynb, a Jupyter Notebook designed for training, fine-tuning, and evaluating a BERT-style Large Language Model (LLM) for multi-label toxic comment classification. The project focuses on building a robust and interpretable model capable of identifying various forms of toxicity in online text, addressing common challenges such as class imbalance and model opacity.

Project Overview
The primary objective of this assignment is to develop a high-performing and explainable multi-label classifier for toxic comments. It leverages state-of-the-art transformer models (ELECTRA), advanced hyperparameter optimisation techniques (Optuna), strategies for handling imbalanced datasets (SMOTETomeK, class weighting), and explainable AI (LIME) to provide insights into model predictions.

Features
Multi-Label Classification: Classifies comments into six distinct toxicity categories: toxic, severe_toxic, obscene, threat, insult, identity_hate.

ELECTRA-base-discriminator: Utilises a powerful pre-trained transformer model for fine-tuning.

Optuna Hyperparameter Optimisation (HPO): Employs Optuna with Hyperband pruning and TPE Sampler for efficient and effective search for optimal training parameters, maximising Macro F1-score.

Class Imbalance Handling: Implements a combination of positive class weighting in the loss function and SMOTETomeK hybrid oversampling to address the severe imbalance in toxicity datasets.

Post-Training Threshold Optimisation: A dedicated Optuna-based phase to find optimal classification thresholds for each label, further enhancing Macro F1-score.

Explainable AI (LIME): Integrates LIME to provide local, word-level explanations for individual model predictions, highlighting influential terms.

Comprehensive Evaluation Metrics: Calculates and visualises F1-score, Precision, Recall, ROC AUC, PR AUC, and Confusion Matrices for each label, as well as micro and macro averages.

Reproducibility: Ensures consistent results through extensive random seed setting.

Google Drive Integration: Designed to save all models, plots, and results persistently to a specified Google Drive path.

Dataset
The model is trained and evaluated on the Jigsaw Toxic Comment Classification Challenge dataset. This dataset consists of online comments annotated by human raters for the presence of the six toxicity categories.

Expected Dataset Files (to be placed in the same directory as the notebook):

train.csv

test.csv

test_labels.csv

sample_submission.csv

Setup and Installation
This notebook is designed to run efficiently in a Google Colab environment, leveraging GPU acceleration.

Open in Google Colab: Upload Assignment_2_LLM.ipynb to your Google Drive and open it with Google Colab.

Enable GPU Runtime: Go to Runtime -> Change runtime type -> Select GPU as the hardware accelerator.

Mount Google Drive: The notebook includes a commented-out section to mount Google Drive. Uncomment and run this cell to allow the notebook to save/load files from your Drive.

# from google.colab import drive
# drive.mount('/content/drive')
# print("\nGoogle Drive mounted!", flush=True)

Install Dependencies: Run the initial cells to install required libraries: transformers, scikit-learn, optuna, imbalanced-learn, lime, seaborn, matplotlib.

Notebook Structure and Usage
The notebook is structured into logical cells, each performing a specific part of the machine learning pipeline.

1. Configuration (Config Class)
This class defines all key parameters for the model, training, and optimisation. Users can modify these parameters to experiment with different settings.

MODEL_NAME: Specifies the pre-trained transformer model (e.g., google/electra-base-discriminator).

MAX_LENGTH: Maximum token sequence length for input.

DEVICE: Automatically set to cuda if GPU is available, else cpu.

RANDOM_SEED: For reproducibility.

N_TRIALS: Number of Optuna trials for HPO.

CV_FOLDS: Number of cross-validation folds during Optuna.

PATIENCE: Early stopping patience for training.

LABEL_COLS: List of target toxicity labels.

SEARCH_SPACE: Defines the hyperparameter ranges for Optuna.

GOOGLE_DRIVE_SAVE_BASE_PATH: Crucial path in your Google Drive for saving all outputs.

OPTUNA_DB_PATH: Path for Optuna's persistent study database.

LOAD_EXISTING_STUDY_RESULTS: Set to True to load previous Optuna results and skip re-optimisation.

LOAD_PREVIOUSLY_TRAINED_FINAL_MODEL: Set to True to load a saved final model and skip its retraining.

2. Helper Functions
set_seed(seed): Sets random seeds across NumPy, PyTorch, and Python's random module for reproducibility.

3. ToxicDataset Class
Purpose: Prepares text and labels for input to the transformer model.

__init__(self, texts, labels_df, tokenizer, max_length, augment=False): Initialises with texts, labels (as a DataFrame), tokenizer, max length, and an optional augment flag.

__len__(self): Returns the number of samples.

__getitem__(self, idx): Tokenises the text, applies optional simple augmentation, and returns input features and multi-hot encoded labels as PyTorch tensors.

_augment_text(self, text): A simple text augmentation method (e.g., random case change for a word).

4. AdvancedTrainer Class
Purpose: Customises the Hugging Face Trainer to support multi-label classification with class weighting.

__init__(self, pos_weights=None, thresholds=None, **kwargs): Takes pos_weights (a tensor for BCEWithLogitsLoss) and thresholds (for evaluation metrics).

compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): Overrides the default loss computation to use torch.nn.BCEWithLogitsLoss with pos_weight for handling class imbalance.

5. MetricsCalculator Class
Purpose: Computes a comprehensive suite of evaluation metrics and handles threshold optimisation.

__init__(self, label_cols): Initialises with the list of label columns.

compute_metrics(self, eval_pred, thresholds=None):

Takes eval_pred (predictions and true labels from Trainer.evaluate()).

Applies sigmoid to predictions to get probabilities.

Binarises predictions using provided thresholds (or 0.5 by default).

Calculates F1-score, Precision, Recall, ROC AUC, PR AUC, and Confusion Matrix components for each individual label.

Computes Micro-averaged and Macro-averaged metrics across all labels.

Crucially, sets metrics['eval_f1'] to the macro_f1 score, which is used by Hugging Face Trainer for selecting the best model checkpoint during training.

optimize_thresholds(self, y_true, y_pred_proba, n_trials=100):

Uses Optuna to find the optimal classification threshold for each label.

The objective is to maximise the overall Macro F1-score on the validation set.

Returns the list of best thresholds and the corresponding Macro F1 score.

6. ModelOptimizer Class
Purpose: Orchestrates the entire model training, hyperparameter optimisation, and final evaluation pipeline.

__init__(self, train_df, val_df=None): Initialises with the training DataFrame and an optional validation DataFrame. Calculates pos_weights for the loss function.

_calculate_pos_weights(self): Computes the pos_weight for each label (negative samples / positive samples) to address class imbalance.

_create_model(self, trial=None): Loads the pre-trained ELECTRA model and applies trial-specific dropout hyperparameters if in an Optuna trial.

_apply_sampling(self, X_train, y_train, oversampling_ratio, sampling_strategy='hybrid'): Applies SMOTETomeK hybrid sampling to the training data based on a pseudo-label (whether a comment is toxic at all).

objective(self, trial): The core Optuna objective function.

Suggests hyperparameters for the current trial.

Performs K-Fold cross-validation on the training data.

Applies SMOTETomeK sampling to the training fold.

Trains a model for each fold using AdvancedTrainer.

Returns the mean Macro F1-score across all folds for Optuna to optimise.

run_optimization(self): Executes the Optuna study, loading existing studies if Config.LOAD_EXISTING_STUDY_RESULTS is True.

train_final_model(self, best_params, save_path=None):

Trains the final model on the full training dataset (after splitting for validation) using the best hyperparameters from Optuna.

Includes logic to load a previously trained model if Config.LOAD_PREVIOUSLY_TRAINED_FINAL_MODEL is True and a model exists at save_path.

Saves the final trained model and tokenizer.

Saves the training log_history to a JSON file.

Performs post-training threshold optimisation using MetricsCalculator.optimize_thresholds.

Generates and saves various plots: training history, confusion matrices, precision-recall curves, ROC curves, and probability distributions.

Saves the final optimised thresholds to a CSV.

7. predict_proba_for_lime Function
Purpose: A helper function designed to be passed to LIME.

Takes a list of raw texts, the model, tokenizer, max_length, and device.

Tokenises the texts, runs them through the model, applies sigmoid, and returns prediction probabilities as a NumPy array.

8. main() Function
Purpose: The entry point of the script, orchestrating the entire workflow.

Sets global random seeds.

Loads datasets.

Performs initial data cleaning (filling NaN).

Prints dataset shapes and label distributions.

Splits the training data into training and validation sets for the final model.

Initialises ModelOptimizer.

Runs Optuna optimisation (or loads previous results).

Calls optimizer.train_final_model to train/load and evaluate the final model.

Prints Optuna summary data.

Initiates Explainable AI (XAI) Analysis with LIME:

Loads the final trained model and tokenizer.

Prepares lime_predictor function.

Selects sample comments from the validation set.

For each sample, it generates and prints LIME explanations for relevant toxicity labels, showing influential words and their weights.

Output Files
All generated outputs are saved to the directory specified by Config.GOOGLE_DRIVE_SAVE_BASE_PATH (default: /content/drive/MyDrive/my_electra_multilabel_classifier_results).

optuna_electra_multilabel_study.db: Optuna's SQLite database for persistent study results.

best_electra_multilabel_model/: Directory containing the saved best model checkpoint and tokenizer.

final_electra_multilabel_model_results/: Directory containing:

final_model_training_log_history.json: Detailed training and evaluation metrics per epoch.

training_metrics_plot_Final_Model.png: Plot of training and validation loss/F1 over epochs.

multilabel_confusion_matrices_Final_Model.png: Grid of confusion matrices for each label.

multilabel_precision_recall_curves_Final_Model.png: Precision-Recall curves for each label.

multilabel_roc_curves_Final_Model.png: ROC curves for each label.

multilabel_probability_distributions_Final_Model.png: Histograms of predicted probabilities for true positives/negatives.

optimized_thresholds_multilabel.csv: CSV file containing the final optimised classification thresholds for each label.

Reproducibility
The Config.RANDOM_SEED variable is used to seed NumPy, PyTorch, and Python's random module, aiming for maximum reproducibility of results across runs.

Customisation
Users can easily customise the following:

Model: Change Config.MODEL_NAME to experiment with other Hugging Face transformer models.

Hyperparameter Search: Adjust Config.SEARCH_SPACE and Config.N_TRIALS for different optimisation strategies.

Imbalance Handling: Modify oversampling_ratio in Config.SEARCH_SPACE or the sampling_strategy in _apply_sampling.

XAI Examples: Adjust num_xai_examples in the main function to explain more or fewer comments.
