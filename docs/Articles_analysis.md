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

## Feature Engineering:
* We first **filter not closed tickets**, transform duration fields from string format to integer, covert data-time string to date-time and encode filds to integers.
* To label a training instance, we use “1” to represent the ticket SLA was violated, this case is called positive instance;
“0” represents the situation when ticket SLA was not violated, and we note this case as a negative instance.
* Add features like "is_day_off" , "day_of_week" , "hour_of_sub"
