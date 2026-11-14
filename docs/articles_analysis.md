"""This file contains analysis of artciles"""

# Machine learning in IT service Management
* IT Service Management (ITSM) is a variety of activities directed towards maintenance of IT infrastructure.
* This paper aims to propose predictive method of the incident resolution time estimation
* SLA (Service-level agreement) is a document that describes the performance criteria a provider promises to meet while
delivering a service; This agreement also sets out the remedial actions and any penalties that will take effect if
performance falls below the promised standard
* The evaluation method is F1 score. It’s a harmonic mean of the precision and recall rate. Prediction and recall are
the common metrics used in classification. F1 score will take into account both the precision and recall.
* 

**Feature Engineering:**
* We first **filter not closed tickets**, transform duration fields from string format to integer, covert data-time string to date-time and encode filds to integers.
* To label a training instance, we use “1” to represent the ticket SLA was violated, this case is called positive instance;
“0” represents the situation when ticket SLA was not violated, and we note this case as a negative instance.
* Add features like "is_day_off" , "day_of_week" , "hour_of_sub"


# Performance Evaluation of Machine Learning Techniques on Resolution Time Prediction in Helpdesk Support System:
* Estimating incident resolution times accurately is critical to maintaining an effective resource allocation for customer service.
* The resolution time prediction is the projection of the duration required for a customer support agent to
address a customer's problem, question, or grievance.
* Apart from that, responding to customer inquiries as quickly as feasible would also greatly enhance customer loyalty.
  
* **data preprocessing**, by removing outliers and missing values and determining any irregularities in the resolution times distribution
* **Several types of ML methods**, such as gradient boosting, Random Forest (RF), and Decision Tree (DT).

**Questions to respond generally:**
- What are the factors that contributed to the support of the ticketing system? (hour, day, SLA_DELAY, n_actions ..)
- How do we propose a new prediction model to predict resolution time? (KNN, Naive-Bayes, SVM, DT, RF, XGBoost ...)
- How to conduct model evaluation for the proposed method? (RMSE, MSE and MAE ...)

**Models used in the article:**
* **Decision Tree** is a non-parametric supervised learning approach commonly used in regression and classification tasks. This technique is particularly useful in decision analysis and helps the user identify a strategy most likely to attain a goal (predicting potential event outcomes and making financial decisions).
In the model training stage, a DT will consider measures such as entropy or Gini impurity to decide the best attribute for splitting the data. These measures evaluate the level of random or disorder of the subgroups.
Among all attributes, the attribute which has a balance of maximum information acquisition and minimizes impurity reduction after the data split will be chosen as the output.

* **Random Forest** technique builds a collection of DTs to produce a forecast or result that is more accurate. This "forest" comprises several DTs trained using the bagging technique (combining different learning models to improve the final product).
It is well known for being flexible and easy to use, and it works well in solving both regression and classification problems.

* **Extreme Gradient Boosting XGBoost** is a class of ensemble ML algorithms capable of solving both classification and regression tasks. This method is known for its efficiency and accuracy. It belongs to the class of ensemble learning techniques known as boosting algorithms (this algorithm combines the predictions of multiple weaker learners to gain results).
In training the model, this algorithm methodically constructs simple and brief DTs. Due to its natural bias, every tree is referred to as a "weak learner". In order to address forecast mistakes caused by previous models, these trees are gradually added to the ensemble and modified.

**Comparaison of models:**
* DT is the most simple approch because it adopts an information-based approach, and so excludes data preprocessing steps like data normalization and data scaling. However, DT is unstable since even a small change in the data will have a big effect on the structure of the DT.

* the RF method may counteract DT's tendency to overfit its training set, often producing an excellent model performance outcome. However, this approach cannot comprehend the results, the potential for overfitting, and the
requirement to predefine the number of trees to be incorporated into the model.

* XGBoost, can speed up data processing time as it does not involve feature normalization. On the other hand, if
the created trees are sufficiently deep with noisy data, this strategy may lead to overfitting, just as the RF algorithm.

**Results in the article: Forecasting customer support resolution times
through automated machine learning:**
* AutoML models have the highest performance. The best model is a stacked ensemble model, a combination of 100 different model predictions.
* The most effective individual model in AutoML models is the Gradient Boosting Machine (GBM), which shows high performance very close to the stacked ensemble models.

**Result is this article**
* the best method is XGBoost (btained 81.102384% in accuracy, surpassing other methods).
* The second best method is the Ensemble method, and followed by RF as well as DT.
* The ensemble method achieved 80.012906% accuracy while RF got 79.801571% accuracy.
* The model with the lowest accuracy is DT.
  
* XGBoost achieves the best result, which is 2.5081 in RMSE, 6.2907 in MSE and 1.6593 in MAE.
* The second best method is Ensemble Method, which obtained 2.5794 in RMSE, 6.6534 in MSE, and 1.6916 in MAE.

From the aspect of model fitting time, 
* the best method is DT. This technique obtained a fitting time that was significantly shorter than that of other methods, which was 0.2854 seconds in precision. 
* The second best method is XGBoost, which got 12.8610 seconds.
* The third best method is Ensemble Method, which only requires 14.6554 seconds for model fitting,
* while the fourth best method is RF which requires 15.9628 seconds.

**As a result, it is unquestionably shown that XGBoost is the best option compared to other approaches, which maintain its effectiveness and accuracy in customer support ticket systems.**