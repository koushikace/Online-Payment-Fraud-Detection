Online Payment Fraud Detection
Project Overview
This project aims to build and evaluate machine learning models for detecting fraudulent transactions in an online payment dataset. With the increasing volume of digital transactions, identifying and preventing fraud is crucial for financial institutions and e-commerce platforms. This repository contains the data analysis, preprocessing, and machine learning model training pipeline to predict fraudulent transactions.

Dataset
The core dataset for this project is new_data.csv. This file contains simulated financial transaction data from a mobile money service. It includes various features related to the transactions, such as:

step: Represents a unit of time in the real world. 1 step is 1 hour.

type: Type of online transaction (e.g., CASH_OUT, PAYMENT, DEBIT, TRANSFER, CASH_IN).

amount: The amount of the transaction.

nameOrig: Customer ID of the initiator of the transaction.

oldbalanceOrg: Initial balance before the transaction.

newbalanceOrg: New balance after the transaction.

nameDest: Customer ID of the recipient of the transaction.

oldbalanceDest: Initial balance of the recipient before the transaction.

newbalanceDest: New balance of the recipient after the transaction.

isFraud: This is the target variable; 1 if the transaction is fraudulent, 0 otherwise.

isFlaggedFraud: Indicates if the transaction was flagged by the system (often for large transfers).

Note on Dataset Size: The new_data.csv file is quite large (approx. 470 MB). To manage this on GitHub, Git Large File Storage (Git LFS) is used. This ensures that the repository remains clonable while handling large data files efficiently.

Data Exploration & Preprocessing
The onilepayment.ipynb notebook details the following data exploration and preprocessing steps:

Initial Data Loading: Using pandas to load and get a quick overview (.head(), .info(), .describe()) of the dataset.

Categorical and Numerical Feature Identification: Separating variables by their data types (object, integer, float) to understand the dataset structure.

Exploratory Data Analysis (EDA):

Transaction Type Distribution: Visualizing the counts of different transaction types using sns.countplot.

Fraud Distribution: Checking the balance of fraudulent vs. non-fraudulent transactions using data['isFraud'].value_counts().

Time Step Distribution: Plotting the distribution of step using sns.displot to understand transaction timing patterns.

Correlation Heatmap: Visualizing the correlation matrix of numerical features (after converting categorical features using pd.factorize) to identify relationships between variables using seaborn.heatmap.

Feature Engineering and Encoding:

One-Hot Encoding: Converting the categorical type column into numerical format using pd.get_dummies to prepare it for machine learning models.

Feature and Target Separation:

Defining X (features) by dropping the target variable (isFraud) and other non-informative/redundant columns (type - as it's now encoded, nameOrig, nameDest).

Defining y (target) as the isFraud column.

Data Splitting: Dividing the dataset into training (X_train, y_train) and testing (X_test, y_test) sets with a test_size of 30% and random_state=42 for reproducibility.

Modeling
This project evaluates the performance of several classification models:

Logistic Regression: A linear model for binary classification.

XGBoost Classifier (XGBClassifier): A powerful gradient boosting framework known for its speed and performance.

Random Forest Classifier (RandomForestClassifier): An ensemble learning method that builds multiple decision trees.

The models are trained on the preprocessed training data, and their performance is evaluated using the following metric:

ROC AUC Score (roc_auc_score): This metric is particularly useful for imbalanced datasets (common in fraud detection) as it measures the classifier's ability to distinguish between classes.

Model Training & Evaluation Process
For each model:

The model is initialized and trained (fit) on the X_train and y_train datasets.

Training Accuracy (ROC AUC): Predictions (predict_proba) are made on the training set, and the ROC AUC score is calculated.

Validation Accuracy (ROC AUC): Predictions are made on the X_test (validation) set, and the ROC AUC score is calculated to assess generalization performance.

Confusion Matrix
A Confusion Matrix is plotted for one of the models (specifically, models[1], which is the XGBoost Classifier based on the code provided) using ConfusionMatrixDisplay.from_estimator to visualize its classification performance (True Positives, False Positives, True Negatives, False Negatives) on the test set.

Key Libraries Used
numpy

pandas

matplotlib.pyplot

seaborn

scikit-learn (for train_test_split, LogisticRegression, RandomForestClassifier, roc_auc_score, ConfusionMatrixDisplay)

xgboost (XGBClassifier)

How to Run the Project Locally
To replicate the analysis and run the notebook on your machine:

Clone the Repository:

git clone https://github.com/koushikace/Online-Payment-Fraud-Detection.git

Navigate to the Project Directory:

cd Online-Payment-Fraud-Detection

Ensure Git LFS is Installed:
If you don't have Git LFS, install it from https://git-lfs.github.com and then run git lfs install in your terminal. This is crucial for downloading the large new_data.csv file.

git lfs install

Create a Virtual Environment (Recommended):

python -m venv venv

Activate the Virtual Environment:

Windows:

.\venv\Scripts\activate

macOS/Linux:

source venv/bin/activate

Install Dependencies:
You can install the necessary libraries using pip. It's recommended to create a requirements.txt file (e.g., pip freeze > requirements.txt after installing them) and then install with pip install -r requirements.txt. Otherwise, install them manually:

pip install numpy pandas matplotlib seaborn scikit-learn xgboost

Download LFS Files: After cloning, Git LFS should automatically download the new_data.csv file. If not, you can force it:

git lfs pull

Open the Jupyter Notebook:

jupyter notebook onilepayment.ipynb

This will open the notebook in your web browser, where you can execute the cells to see the data analysis, model training, and results.

Future Work and Improvements
Feature Engineering: Explore more advanced feature engineering techniques, such as creating interaction terms or lagged features, which are often beneficial in time-series-like datasets.

Handling Imbalance: Investigate other techniques for handling class imbalance beyond what's shown (e.g., oversampling the minority class, undersampling the majority class, or using specialized loss functions).

Hyperparameter Tuning: Optimize the hyperparameters for each model using techniques like GridSearchCV or RandomizedSearchCV to achieve better performance.

Anomaly Detection: Explore unsupervised anomaly detection algorithms as an alternative approach to fraud detection.

Deployment: Consider building a simple API or web application (e.g., using Flask or Streamlit) to deploy the trained model for real-time fraud prediction.

Deep Learning Models: Investigate the applicability of deep learning models (e.g., LSTMs, Feed-forward Neural Networks) for capturing complex patterns in transaction data.

Contact
If you have any questions or suggestions, feel free to reach out:

GitHub: koushikace

License
This project is licensed under the MIT License - see the LICENSE file for details.