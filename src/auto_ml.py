import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import numpy as np

# Load data
df = pd.read_csv("../data/processed/cleaned_data.csv")

df["target_reg"] = df["target_reg"].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["target_reg"]).reset_index(drop=True)

# The first 80% of data for training, the last 20% for test
train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)

# train AutoML for MSE
predictor = TabularPredictor(
    label="target_reg", problem_type="regression",
    eval_metric="mean_squared_error").fit(train_data=train_data, time_limit=120)

# Évaluation of performance on test data
scores = predictor.evaluate(test_data)
print("Scores de performance (temporel) :", scores)
