import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

df = pd.read_csv("../data/cleaned_data.csv")

valid_features = [
    'SLA_DELAY',
    'submit_hour',
    'submit_dayofweek',
    'URGENCY_VALUE',
    'IMPACT_VALUE',
    'REQUEST_ORIGIN',
    'SD_CATALOG_PATH_EN',
    'SUBMITTED_BY_ID'
]

df_boruta = df[valid_features + ['target_reg']].dropna().copy()
df_boruta = df_boruta[df_boruta['target_reg'] > 0]

# Encodage by fréquency on variables of catégories
for col in ['REQUEST_ORIGIN', 'SD_CATALOG_PATH_EN', 'SUBMITTED_BY_ID']:
    freq = df_boruta[col].value_counts(normalize=True)
    df_boruta[col] = df_boruta[col].map(freq).fillna(0)

X = df_boruta[valid_features].values
# Transformation on log1p to neutralize outliers
y = np.log1p(df_boruta['target_reg'].values)

# Exécution of Boruta with a limited max_depth 
rf = RandomForestRegressor(n_jobs=-1, max_depth=7, n_estimators=100, random_state=42)

boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=1, random_state=42, max_iter=50)

boruta_selector.fit(X, y)

# Results
results = pd.DataFrame({
    'Attribut': valid_features,
    'Boruta_Rank': boruta_selector.ranking_,
    'Keep': boruta_selector.support_
}).sort_values('Boruta_Rank')

print(results)

"""
Iteration: 	20 / 50
Confirmed: 	8
Tentative: 	0
Rejected: 	0

            Attribut  Boruta_Rank  Keep
0           SLA_DELAY            1  True
1         submit_hour            1  True
2    submit_dayofweek            1  True
3       URGENCY_VALUE            1  True
4        IMPACT_VALUE            1  True
5      REQUEST_ORIGIN            1  True
6  SD_CATALOG_PATH_EN            1  True
7     SUBMITTED_BY_ID            1  True

"""