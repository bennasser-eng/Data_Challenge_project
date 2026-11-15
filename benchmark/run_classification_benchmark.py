"""Script of benchmark classification."""

import sys
import os
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.training_utils import tune_models, retrain_best_models, evaluate_models
from src.models import clf_pipelines, clf_param_grids

# --- Metrics ---
clf_metrics = {
    "F1": lambda y, pred: f1_score(y, pred, average="macro"),
    "Precision": lambda y, pred: precision_score(y, pred, average="macro", zero_division=0),
    "Recall": lambda y, pred: recall_score(y, pred, average="macro", zero_division=0)
}


def run_benchmark(X_train, y_train, X_test, y_test):

    # --- Temporal Separation ---
    tscv = TimeSeriesSplit(n_splits=3)

    # --- Executing the pipeline ---
    print("\n=== Phase 1 : TUNING (Cross Validation) ===")
    optimal_configs = tune_models(
        X_train, y_train, 
        models_dict=clf_pipelines, 
        param_grids=clf_param_grids, 
        cv_strategy=tscv, 
        scoring_metric='f1_macro'
    )

    print("\n=== Phase 2 : RETRAINING (on training data) ===")
    final_models = retrain_best_models(
        X_train, y_train, 
        models_dict=clf_pipelines, 
        best_configs=optimal_configs
    )

    print("\n=== Phase 3 : EVALUATION ===")
    results = evaluate_models(
        fitted_models=final_models, 
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test, 
        metrics_dict=clf_metrics
    )

    return results


if __name__ == "__main__":
    # Load data
    data = pd.read_csv('../data/processed/cleaned_data.csv')
    
    # Features separation
    X = data.drop(columns=['target_clf', 'target_reg']) 
    y = data['target_clf']
    
    # temporel Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.17, shuffle=False)
    
    # Executing the benchmark
    results = run_benchmark(X_train, y_train, X_test, y_test)
    
    # Performances
    print("\n--- PERFORMANCES ---")
    colonnes = ['model_name', 'F1_train', 'F1_test', 'Precision_test', 'Recall_test']
    
    print(results[colonnes].sort_values(by='F1_test', ascending=False).to_string(index=False))