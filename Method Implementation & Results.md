# Method Implementation

To start the analysis of the phishing dataset, we begin by implementing data exploration. This foundational process is especially vital when dealing with datasets such as phishing data. Data exploration enables analysts to develop a comprehensive understanding of the dataset, laying the groundwork for extracting valuable insights. Through thorough investigation and an open- minded approach, data exploration empowers analysts to navigate the dataset effectively, thereby enhancing the overall analytical process. Its emphasis on in-depth inquiry is essential for informing decision-making and driving advancements in cybersecurity research, particularly in the field of phishing prevention.


<img width="989" alt="Screenshot 2024-06-06 at 3 16 49 PM" src="https://github.com/KadirOrcunAltunel/PhishingDetection/assets/63982765/1f919a53-87a3-40c2-82b8-3b4fa94824f9">

For the phishing detection project, we integrate a comprehensive cybersecurity dataset titled "dataset_cybersecurity_michelle.csv" into Python working environment. Leveraging the Google Drive API, we mount Google Drive to gain access to this extensive collection of data. Utilizing the pandas library, renowned for its robust data manipulation capabilities, we import the dataset into a DataFrame, which enables us to work with the data in a structured tabular format.

We adjust the display settings of pandas to ensure full visibility of the dataset's numerous attributes. This preliminary glimpse into the data is achieved through the invocation of the `head()`
function, which displays the first five entries. These entries reveal a variety of features, such as counts of dots, hyphens, underlines, and slashes within URLs—potentially indicative markers in the identification of phishing attempts.
The scale of the dataset is substantial, with its dimensions being revealed as 129,698 rows and 112 columns. This indicates a rich, multidimensional dataset that is poised to be an invaluable asset for the subsequent data analysis and machine learning applications, focusing on the nuances of cybersecurity threats.

<img width="988" alt="Screenshot 2024-06-06 at 3 18 13 PM" src="https://github.com/KadirOrcunAltunel/PhishingDetection/assets/63982765/e14cebfb-8289-4868-b36e-04b82935c1b8">

Next, we perform a statistical examination of the dataset to derive insights into the characteristics of URLs that may be indicative of phishing activity. To achieve this, we calculate the count of unique values across the dataset using   `nunique(axis=0)` to understand the diversity in the URL features. Subsequently, we use the `describe()` function to obtain a comprehensive summary of the data, which includes count, mean, standard deviation, minimum and maximum values, as well as the 25th, 50th (median), and 75th percentiles for each feature related to URL characteristics.

For instance, the 'qty_dot_url' feature, which represents the count of dots in a URL, shows an average (mean) of approximately 2.22 with a standard deviation of 1.31, indicating variability in how many dots URLs typically contain. The maximum number reported for this feature is 24, which can be an outlier or indicative of a complex or potentially malicious URL.

This statistical summary offers critical insights into the dataset and assists in identifying patterns or anomalies that could be further investigated for phishing detection. It sets the groundwork for the application of machine learning algorithms to classify URLs as phishing attempts or benign by quantitatively assessing URL patterns.

<img width="983" alt="Screenshot 2024-06-06 at 3 18 58 PM" src="https://github.com/KadirOrcunAltunel/PhishingDetection/assets/63982765/c5ed8f58-6e7e-47e1-8b46-3bb6609e0ecc">

Next, we detail a meticulous data cleaning process conducted on the cybersecurity dataset. This process commences with the identification of missing values across different features in the dataset. To accurately pinpoint these missing values, which are denoted by the placeholder '-1' within the dataset, we calculate the sum of occurrences of this placeholder across all features using the command `(phishing_dataset==-1).sum()`.

Subsequently, we deploy the pandas library’s context manager to override the default display settings, allowing me to view the full breadth of the dataset without any truncation. By invoking `sort_values(ascending=False)` on the series of missing values and printing the results, we obtain a descending order list of features by the count of missing values, providing a clearpicture of which features required attention due to the prevalence of missing data.

The output reveals that several parameters, including 'qty_plus_params', 'qty_asterisk_params', 'qty_hashtag_params', and others, each had 117,020 instances marked as missing. This information is critical as it influences the subsequent steps in data preprocessing, where such missing values must be addressed through imputation or elimination to refine the dataset for the predictive modeling phase of the project on phishing URL detection.

<img width="998" alt="Screenshot 2024-06-06 at 3 19 29 PM" src="https://github.com/KadirOrcunAltunel/PhishingDetection/assets/63982765/722be849-78a0-4a97-9bb6-24ad642bef19">


Following the identification of missing values within the dataset, we proceed with data cleaning by discarding features that exhibited a significant number of missing entries. To maintain the integrity of the analysis, we determine that any feature with more than half of its values missing —specifically, more than 59,703 missing entries out of the total of 129,698—should be considered too sparse for reliable analysis and hence removed from the dataset.

The selection of columns to be dropped is executed using a filtering condition on the previously calculated series of missing values, identifying those columns which surpassed the threshold for missing data. The identified columns are then eliminated from the dataset, resulting in a reduced set of features that we deem more suitable for building robust predictive models.

After this pruning process, the dataset is pared down to 92 features from the original 112, as evidenced by the output of `phishing_dataset_cleaned.shape`.

<img width="990" alt="Screenshot 2024-06-06 at 3 20 02 PM" src="https://github.com/KadirOrcunAltunel/PhishingDetection/assets/63982765/4563c19a-7f90-4ad3-9636-5aa4144cbb6e">

In the data preprocessing stage of the project, we address missing values in the cleaned dataset by employing a statistical imputation technique known as `Multiple Imputation by Chained Equations (MICE)` which can be accessed in detail in the Appendix: Methods section of the report. We instantiate an `ImputationKernel` from the respective Python library, configuring it with the cleaned dataset, setting `save_all_iterations` to `True` for thorough analysis, and defining a `random_state` for reproducibility.

Then, we initiate the MICE process, requesting ten iterations to refine the imputation with a verbose output to monitor the progress. Upon completion of this process, we call the `complete_data()` method on the kernel object to retrieve the imputed dataset.

To ensure the integrity of the imputation, we perform a verification step, checking for null values using the `isnull.sum()`  method. The output confirms that the dataset no longer contains any missing values across all features, which is validated by sorting and displaying the sum of null values for each feature, resulting in zero for all.

This rigorous imputation process ensures that our dataset is robust, with all missing values judiciously estimated, thereby enhancing the reliability of the subsequent machine learning models that we apply to the data in the project.

<img width="995" alt="Screenshot 2024-06-06 at 3 20 22 PM" src="https://github.com/KadirOrcunAltunel/PhishingDetection/assets/63982765/2ad1c0f6-1566-44f5-b208-df1b5c504c18">

After deploying MICE for our phishing dataset, we continue the project by implementing an anomaly detection algorithm using the Isolation Forest method. We configure an Isolation Forest model with 500 estimators, which refers to the number of base estimators in the ensemble, and set the contamination parameter to 'auto' to allow the algorithm to automatically determine the threshold for outlier detection. The random_state parameter is fixed at 42 to ensure reproducibility of results.

We fit the model to our imputed phishing dataset, which has already undergone preprocessing to deal with missing values. Once the model is trained, we use it to predict the outliers within our dataset. The prediction yields an array of labels where '-1' signifies an outlier and '1' denotes an inlier.

By calculating the sum of occurrences of '-1' and '1' in our prediction array, we are able to quantify the number of outliers and inliers detected by the model. In our case, the model identifies 3,655 outliers and 126,043 inliers, providing us with a clear perspective on the distribution of anomalies within our data. This step is crucial as it helps us isolate potential anomalies that could be indicative of phishing activities, thereby enhancing the robustness of our cybersecurity analysis. 

<img width="1008" alt="Screenshot 2024-06-06 at 3 20 49 PM" src="https://github.com/KadirOrcunAltunel/PhishingDetection/assets/63982765/25188a08-dadb-4168-a08d-12da0cdf19c7">

In the current phase of the project, we apply the `eXtreme Gradient Boosting (XGBoost)` algorithm to further refine our phishing detection model. The XGBoost model is favored for its performance and speed. Labels are assigned to our dataset where outliers are marked with '1' and inliers with '0'. We then split our dataset into training and testing sets, allocating 20% of the data for testing while maintaining a random state for consistent train-test splits across experiments.

To address class imbalance, we calculate the ratio of inliers to outliers and set the `scale_pos_weight` parameter in our XGBoost classifier to this ratio. This technique helps in normalizing the influence of each class on the learning process. We configure the classifier with 100 trees and ensure reproducibility by fixing the `random_state`. The `use_label_encoder=False` parameter is set to avoid using the deprecated label encoder in XGBoost.

Post-training, we deploy the model to make predictions on our test set. The performance is quantitatively assessed using a confusion matrix and a classification report. The confusion matrix indicates a substantial number of true positives (700) and true negatives (25,164), suggesting a strong ability to correctly identify both phishing and non-phishing instances. The classification report provides detailed metrics such as precision, recall, and F1-score. A precision of 0.93 for the phishing class and an F1-score of 0.95 demonstrate the model's high accuracy and balanced performance. Moreover, the weighted averages of precision, recall, and F1-score all stand at 1.00, reflecting the model's overall accuracy and its robustness in the context of phishing detection.

<img width="992" alt="Screenshot 2024-06-06 at 3 21 20 PM" src="https://github.com/KadirOrcunAltunel/PhishingDetection/assets/63982765/8107c7b4-38d6-4fc0-b4db-650b32a90dad">

Next, we continute evaluating our model by applying a `Random Forest Classifier`, a machine learning model known for its high accuracy and ability to operate over large datasets with a multitude of features.
 
Our Random Forest Classifier is configured with 100 estimators—individual decision trees that contribute to the overall decision-making process. We apply a 'balanced' class weighting to account for any discrepancies in the representation of classes within the dataset. This helps to mitigate the bias toward the majority class that would otherwise occur in an imbalanced dataset.

The confusion matrix indicates a substantial number of true positives (647) and true negatives (25,198), again, suggesting a strong ability to correctly identify both phishing and non-phishing instances. A precision of 0.98 for the phishing class and an F1-score of 0.93 shows that our Random Forest model demonstrates exceptional performance, indicating its potential efficacy in identifying and mitigating phishing threats within cybersecurity frameworks.

<img width="1010" alt="Screenshot 2024-06-06 at 3 22 08 PM" src="https://github.com/KadirOrcunAltunel/PhishingDetection/assets/63982765/2000e608-1d0f-4125-895e-4cfec83224fa">

For the final algorithm for our phishing detection, we use neural networks.

Prior to training, we standardize our features using the   to ensure that our neural network receives data that is on a similar scale, which is essential for the effective training of neural networks.

We construct a neural network using the Keras library with the following architecture:
- An input layer with 64 neurons and the 'ReLU' activation function.
- A hidden layer with 32 neurons, also using the 'ReLU' activation function.
- An output layer with a single neuron with the 'sigmoid' activation function, suitable for binary classification.

Our model is compiled with the binary cross-entropy loss function and the Adam optimizer with a learning rate of 0.001. We track the accuracy and recall during training as performance metrics.

To prevent overfitting, we employ an early stopping mechanism with a patience of 10 epochs, which terminates the training process if the validation loss does not improve for 10 consecutive epochs. The best weights observed during training are restored to the model at the end of training.

The model is trained on the scaled training data for up to 75 epochs with a batch size of 32 and a validation split of 20%. The EarlyStopping callback is used to monitor the model's performance on the validation set.

<img width="965" alt="Screenshot 2024-06-06 at 3 23 08 PM" src="https://github.com/KadirOrcunAltunel/PhishingDetection/assets/63982765/cd521083-3dc2-482c-9c32-484dc2a9c5ca">

Results from our neural network indicate a high level of accuracy and a relatively high recall. The test loss being low suggests that the model's predictions are quite close to the actual labels. High recall indicates that the model is capable of identifying most of the positive instances correctly, which, in the context of phishing detection, means catching a high number of actual phishing attempts.

# Results
Neural Network, Random Forest and XGBoost algorithms are used for phishing detection. They all show exceptional performance in the phishing dataset.

**Accuracy**: The Neural Network has the highest accuracy, followed closely by Random Forest, and then XGBoost.

**Precision**: Random Forest has the highest precision, followed by XGBoost. The precision for the Neural Network is not provided, but the high overall accuracy suggests it is likely competitive.

**Recall**: XGBoost has the highest recall, followed by the Neural Network, and then Random Forest.

**F1-Score**: XGBoost has the highest F1-score, indicating the best balance between precision and recall. Random Forest has a slightly lower F1-score. The F1-score for the Neural Network is not provided but would likely be high based on the other metrics.

In the scenario, where phishing detection is crucial, often the most important metric is recall because the cost of missing a phishing attempt can be very high. As seen from the results XGBoost has the highest recall, making it the most suitable candidate for the phishing detection.
However, the Neural Network's balanced high accuracy and recall also make it a strong candidate, potentially offering a more nuanced balance of performance metrics, which could be more desirable in certain applications.
