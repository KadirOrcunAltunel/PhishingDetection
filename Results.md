# Results
Neural Network, Random Forest and XGBoost algorithms are used for phishing detection. They all show exceptional performance in the phishing dataset.

**Accuracy**: The Neural Network has the highest accuracy, followed closely by Random Forest, and then XGBoost.

**Precision**: Random Forest has the highest precision, followed by XGBoost. The precision for the Neural Network is not provided, but the high overall accuracy suggests it is likely competitive.

**Recall**: XGBoost has the highest recall, followed by the Neural Network, and then Random Forest.

**F1-Score**: XGBoost has the highest F1-score, indicating the best balance between precision and recall. Random Forest has a slightly lower F1-score. The F1-score for the Neural Network is not provided but would likely be high based on the other metrics.

In the scenario, where phishing detection is crucial, often the most important metric is recall because the cost of missing a phishing attempt can be very high. As seen from the results XGBoost has the highest recall, making it the most suitable candidate for the phishing detection.
However, the Neural Network's balanced high accuracy and recall also make it a strong candidate, potentially offering a more nuanced balance of performance metrics, which could be more desirable in certain applications.
