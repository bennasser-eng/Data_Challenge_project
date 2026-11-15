"""Script with all utils to train models"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone


# cross validation to tune parameters 
def tune_models(X_train, y_train, models_dict, param_grids, cv_strategy, scoring_metric):
    """ scoring_metric: e.g 'f1_macro' or 'neg_mean_absolute_error'"""
    best_configs = {}
    
    for name, base_pipeline in models_dict.items():
        print(f"[*] Tuning : {name}...")
        pipeline_to_tune = clone(base_pipeline)
        
        grid_search = GridSearchCV(
            estimator=pipeline_to_tune,
            param_grid=param_grids.get(name, {}), scoring=scoring_metric,
            cv=cv_strategy, n_jobs=-1, refit=False, error_score='raise'
        )
        
        grid_search.fit(X_train, y_train)
        
        best_configs[name] = {
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_
        }
    return best_configs


# Retrain model with the best configuration pof params on all training data
def retrain_best_models(X_train, y_train, models_dict, best_configs):
    fitted_models = {}
    
    for name, base_pipeline in models_dict.items():
        print(f"[*] Re-traîning : {name}...")
        pipeline_to_retrain = clone(base_pipeline)
        pipeline_to_retrain.set_params(**best_configs[name]["best_params"])
        pipeline_to_retrain.fit(X_train, y_train)
        fitted_models[name] = pipeline_to_retrain
        
    return fitted_models
    




# Evaluate model on test data
def evaluate_models(fitted_models, X_train, y_train, X_test, y_test, metrics_dict):
    """metrics_dict : Dictionnary with { "Name of metric": callable_metric_function }"""
    results = []
    
    for name, model in fitted_models.items():
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        
        # Initialization of result
        model_scores = {"model_name": name}
        
        # Compute of metrics
        for metric_name, metric_func in metrics_dict.items():
            model_scores[f"{metric_name}_train"] = metric_func(y_train, pred_train)
            model_scores[f"{metric_name}_test"] = metric_func(y_test, pred_test)
            
        results.append(model_scores)
    return pd.DataFrame(results)