# Presentation of the course
* Max resolution date vs End_date -> classification
* End_date vs stard_date -> regression


# Explanatory data (see data_analysis.ipynb)
* We added "target_regression" and "target_classification" features

# Split train, val and test:
* **Expanding Window Time-Series Cross-Validation**, known as ***TimeSeriesSplit*** in Scikit-Learn. (see docs/Articles)
* **Cross validation**:
  - Test (10% of latest data)
  - Split train data on B plis:
    for each validation step:
       train on i plis and do validation on the next pli
       benchmark hyperparams
    then train on full training data with the best hyperparameters
    then test on the 10% data (test data)