"""All models used in our benchmark"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb


# pipeline for each model
clf_pipelines = {
    # Need StandardScaler
    "Regression_Logistique": Pipeline([
        ('scaler', StandardScaler()), 
        ('model', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    
    # Need StandardScaler
    "SVM": Pipeline([
        ('scaler', StandardScaler()), 
        ('model', SVC(random_state=42))
    ]),
    
    # No need for StandardScaler
    "RandomForest": Pipeline([
        ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
    ]),
    
    # No need for StandardScaler
    "LightGBM": Pipeline([
        ('model', lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1))
    ])

    # Naive bayes , KNN, Decision tree , XGBoost, Ensemble...
}


# hyperparamètres
clf_param_grids = {
    "Regression_Logistique": {'model__C': [0.01, 0.1, 1.0], 'model__penalty': ['l2']},
    "SVM": {'model__C': [0.1, 1.0], 'model__kernel': ['rbf']},
    "RandomForest": {'model__n_estimators': [50, 100], 'model__max_depth': [5, 10]},
    "LightGBM": {'model__n_estimators': [100, 300], 'model__learning_rate': [0.03, 0.1]}
}