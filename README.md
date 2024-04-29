### 1.)Gender Classification by Voice 

https://colab.research.google.com/drive/1CaOyCRVAMFPbt6H3gRUhS92HGE54_Cyj

The Voice Gender Classification project aims to analyze and classify gender based on voice samples.
The dataset used in this project contains 21 acoustic features such as mean frequency, standard
deviation, median, and other relevant characteristics extracted from 1,00,000 voice recordings of
male and female voices , from this
http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/8kHz_16bit/
Repository .
Each voice sample is stored as a .WAV file, which is then pre-processed for acoustic analysis using
the specan function from the  WarbleR  R package. Specan measures 22 acoustic parameters on
acoustic signals for which the start and end times are provided.
The output from the pre-processed WAV files were saved into a CSV file, containing 3168 rows and
21 columns (20 columns for each feature and one label column for the classification of male or
female).The dataset on preprocessing and further analysis is shown to have equal count of male and
female voices .

Acoustic Properties Measured
The following acoustic properties of each voice are measured:
 duration: length of signal
 meanfreq: mean frequency (in kHz)
 sd: standard deviation of frequency
 median: median frequency (in kHz)
 Q25: first quantile (in kHz)
 Q75: third quantile (in kHz)
 IQR: interquantile range (in kHz)
 skew: skewness (see note in specprop description)
 kurt: kurtosis (see note in specprop description)
 sp.ent: spectral entropy
 sfm: spectral flatness
 mode: mode frequency
 centroid: frequency centroid (see specprop)
 peakf: peak frequency (frequency with highest energy)
 meanfun: average of fundamental frequency measured across acoustic signal
 minfun: minimum fundamental frequency measured across acoustic signal
 maxfun: maximum fundamental frequency measured across acoustic signal
 meandom: average of dominant frequency measured across acoustic signal

 mindom: minimum of dominant frequency measured across acoustic signal
 maxdom: maximum of dominant frequency measured across acoustic signal
 dfrange: range of dominant frequency measured across acoustic signal
 modindx: modulation index. Calculated as the accumulated absolute difference between
adjacent measurements of fundamental frequencies divided by the frequency range
NOTE:modindex and duration was excluded from the data set .
The primary objective of this project is to build and evaluate machine learning models that can
accurately predict the gender of an individual based on the provided voice features. The main
approach employed for this task is logistic regression .
After data preprocessing and exploring relationship between the features with each other and
further visualisations Logistic regression is employed as a machine learning model to predict gender
based on voice features. The dataset is split into training (80%)and testing sets(20%). The logistic
regression model is trained on the training set, and predictions are made on the test set. Model
performance is evaluated using a confusion matrix, accuracy calculation, and logistic regression
summary.
Threshold Optimization
Different threshold values are explored to find the optimal logistic regression model. The model with
the highest accuracy is selected, and the corresponding threshold is identified.
	



### 2.)Predicting And Analysing the Severity of Road Traffic Accidents 
The data I used in this project was collected from the UK government's official statistics on road traffic accidents. The dataset included information about the accidents, the vehicles involved, and the casualties.

The first step in the data preprocessing was to clean the data. I removed any irrelevant columns and dealt with missing values. For categorical variables, I used label encoding to convert them into numerical values that could be used in my machine learning model. For numerical variables, I used standard scaling to ensure that all features had the same scale.

### Exploratory Data Analysis
*I performed exploratory data analysis to understand the data better and identify any patterns or trends. I visualized the distribution of the severity of injuries and the correlation between different features. This helped me understand which features might be important in predicting the severity of injuries.*

### Model Building

I divided the dataset into a training set and a test set. I chose a Random Forest Classifier as my model due to its ability to handle both categorical and numerical data, and its robustness to overfitting. I trained the model on the training set.

### Model Evaluation
I evaluated the model's performance using the test set. I used metrics like accuracy, precision, recall, and F1 score to assess the model's performance. I also performed cross-validation to ensure that my model was not overfitting the data.

### Model Optimization
*To improve the model's performance, I performed hyperparameter tuning using GridSearchCV. I also checked the importance of the features in the model, which gave me insights into which factors were most influential in predicting the severity of injuries.*

### Conclusion
This project demonstrated how machine learning can be used to predict the severity of road traffic accidents. The model I built can be used by traffic authorities and policymakers to understand the factors that contribute to the severity of accidents and develop strategies to reduce their impact. Future work could involve incorporating more data, such as weather conditions and road conditions, to improve the model's accuracy.










 ### 3.)Credit Card Fraud Detection:

 
In this data science project, we address the critical issue of credit card fraud detection using machine learning and Python. With the increasing prevalence of online transactions, fraudsters continually seek ways to exploit vulnerabilities. We analyze a real-world credit card transaction dataset and employ various anomaly detection algorithms such as Isolation Forest, Local Outlier Factor, and One-Class SVM to identify fraudulent activities. The project focuses on achieving a balance between precision and recall to minimize both false positives and false negatives, ensuring that genuine transactions are not mistakenly flagged as fraudulent. By successfully detecting and preventing credit card fraud, this project serves as a powerful tool to protect customers and financial institutions from potential financial losses.
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.









### 4.)Predicting The Onset Of Diabetes Using Diagnostic Measures 


The Pima Indian Diabetes Dataset is a collection of medical records for 768 Pima Indian women. The records include information on age, weight, height, blood pressure, and glucose levels. The dataset has been used to study the risk factors for diabetes and to develop machine learning models to predict diabetes risk.
Variables used here Pregnancies denoting Number of pregnancies ,Glucose denoting 2-hour plasma glucose concentration during an oral glucose tolerance test, Blood Pressure denotes Blood Pressure Diastolic)(mm Hg),SkinThickness  denotes  Thickness of skin and Insulin  denotes 2-Hour Serum Insulin (mu U/ml), DiabetesPedigreeFunction  which is a function (2-hour plasma glucose concentration during an oral glucose tolerance test),BMI which is Body Mass Index (weight in kg/(height in meters)^2),Age denoting  Persons age(in years) and finally
Outcome which denotes  diabetes status (1: diabetes, 0: no diabetes)
The target variable is specified as "outcome", where 1 indicates a positive result of the diabetes test and 0 indicates a negative result.
A machine learning model is to be developed that can predict whether individuals have diabetes or not when their features are given. 
Missing Values Check:
When we checked for missing values in the data set, we got the answer that there is no missing value. However, we observed a value of 0 in values such as Glucose, BloodPressure, SkinThickness, Insulin, and BMI in our dataset. These values cannot be 0. So we will replace these 0 values with NaN.

EDA
The bar graph tells us that the behaviors of those with diabetes are around 35%, while those without diabetes are around 65%.
Histogram for pregnancies, SkinThickness ,Insulin, DiabetsPedigreeFunction ,Age is positively skewed, for glucose it is negatively skewed and   symmetric for BloodPressure, BMI
For people without diabetes (Outcome 0):On average, they have around 3.3 pregnancies,Their average glucose level is about 110,Their average blood pressure is approximately 68,The average thickness of their skin is about 19.7,They have an average insulin level of roughly 68.8,Their average BMI (body mass index) is around 30.3,The average value of a diabetes-related metric is approximately 0.43,Their average age is about 31.2 years

For people with diabetes (Outcome 1): On average, they have around 4.9 pregnancies. Their average glucose level is higher, about 141.3.Their average blood pressure is also higher, around 70.8.The average thickness of their skin is a bit more, about 22.2.They have a higher average insulin level, about 100.3.Their average BMI is notably higher, around 35.1.The average value of the diabetes-related metric is slightly higher, about 0.55. and their average age is higher as well, about 37.1 years
There are no missing values in any of the columns. This is beneficial because it means that all the data we have in our Dataset is complete and doesn't have any missing information in any of the columns. This can make our analysis more accurate and reliable since we have complete information for each data point.
The bar chart will have bars that represent each column in your DataFrame. The height of each bar represents the proportion of missing values in that column. If a column has no missing values, its bar will be full. If a column has many missing values, its bar will be shorter. This visual representation helps us to quickly understand which columns have missing data.

The heatmap representation is a grid where each cell corresponds to a column and a row. If a cell is blank, it means there are no missing values for that column and row combination. If a cell is colored, it indicates missing values. The color intensity or shade may represent the proportion of missing values.From heatmap we find a high correlation between SkinThickness and insulin

Modelling:
We have used 3 Machine Learning Models :Random Forest Classifier,Logistic Regression Model,XGBOOST classifier.
Accuracy of randomforest classifier model is 88.3% that is the RandomForest model correctly predicted the outcome for around 88.3 instances out of 100 instances

Accuracy of logistic regression model is 77.92% that is the  model  correctly predicted the outcome for around 77.92instances out of 100 instances

Accuracy of XGBOOST classifier is 89.6% that is the  model  correctly predicted the outcome for around 89.6 instances out of 100 instances
Highest accuracy of prediction comes from XGBOOST classifier whereas the lowest comes from Logistic Regression Model

Possible reasons for better performance of XGBOOST classifier:

 XGBoost is an ensemble model that combines multiple weak learners. This can improve prediction accuracy by aggregating the strengths of individual models .The dataset contains complex relationships between the features and the target variable. XGBoost's ability to capture nonlinear relationships through decision trees  helps it model these complexities more effectively compared to other . XGBoost's ability to capture such interactions  leads to improved accuracy. XGBoost's built-in regularization techniques (like L1 and L2 regularization) helps prevent overfitting. In a dataset with limited samples like the Pima dataset, overfitting can be a concern, and XGBoost's regularization leads to better generalization. XGBoost has a wide range of hyperparameters that can be tuned to optimize performance for a specific dataset. This extensive tuning capability can lead to better performance on the Pima dataset. 

The lower accuracy of logistic regression compared to random forest and XGBoost classifiers on the Pima Indian diabetes dataset could be due to several factors:

Logistic regression assumes a linear relationship between features and the target variable. If the relationship in the data is non-linear, as is often the case with medical data, logistic regression might struggle to capture the complexity. Random forests and XGBoost are more capable of capturing non-linear relationships due to their ensemble nature and ability to create decision boundaries that are more flexible. : Random forests and XGBoost are ensemble methods, meaning they combine multiple base models (decision trees) to make a final prediction. This ensemble nature allows them to reduce overfitting and improve generalization by combining the strengths of multiple models. Logistic regression is a single linear model and might overfit or underfit the data more easily. 
Scope for further study:
The dataset can be used to identify factors that are associated with an increased risk of developing diabetes. This information can be used to develop interventions to reduce the risk of diabetes in high-risk individuals.It can be used to identify new targets for drug development. For example, researchers may be able to identify genes or proteins that are involved in the development of diabetes and develop drugs that target these molecules.The dataset can be used to identify effective strategies for preventing diabetes. 







