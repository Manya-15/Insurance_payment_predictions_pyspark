# Insurance Premium Payment Behavior Analysis using PySpark

This project aims to analyze customer data from an insurance company to understand patterns in premium payment behavior and predict the likelihood of customers paying premiums on time. The solution is implemented using **Apache Spark** with its **MLlib** machine learning library.

## 📂 Dataset Overview

Two datasets were used:

- `train.csv`: Contains labeled data used for training the models.
- `test.csv`: Contains unlabeled data for prediction.

**Key Features:**
- `perc_premium_paid_by_cash_credit`: % premium paid via cash/credit
- `Income`: Monthly income of the customer
- `Count_3-6_months_late`, `Count_6-12_months_late`, `Count_more_than_12_months_late`: Count of late payments
- `application_underwriting_score`, `no_of_premiums_paid`, `sourcing_channel`, `residence_area_type`
- `target`: 1 if premium paid on time, 0 otherwise

## 🔧 Technologies Used

- Apache PySpark (Spark SQL, DataFrame API, MLlib)
- Python (Pandas, Matplotlib, Seaborn)
- Google Colab (with Google Drive integration)

## 📊 Data Analysis

- **Scatter Plot**: Explored the relationship between income and premium payment mode.
- **Bar Chart**: Average income per target class (paid on time vs. not paid).
- **Crosstab**: Premium behavior segmented by urban vs. rural customers.
- **Correlation Analysis**: Used Pearson correlation on numeric features.
- **Missing Values**: Detected and addressed missing data in both train and test datasets.

## 🤖 Machine Learning Models

Multiple classifiers were built and evaluated using PySpark MLlib:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Gradient Boosted Tree Classifier**
5. **Naive Bayes**
6. **Support Vector Machine (LinearSVC)**

Each model was trained on preprocessed data, and predictions were generated on the test set.

### Evaluation Metrics

- Accuracy
- Confusion Matrix
- Area Under ROC Curve (AUC)
- ROC Curves plotted for comparison across models

## 📌 Insights

- Customers earning less than ₹1.8L/month showed higher default risk.
- High-income customers paid lower percentages via cash/credit — possibly using alternative channels.
- Rural customers generally had better payment behavior than expected.
- Income, count of late payments, and sourcing channel were significant predictors.

## 📁 Directory Structure
```text
├── insurancePyspark.ipynb # Main notebook
├── train.csv # Training data
├── test.csv # Test data

```

## 🚀 Getting Started

1. Upload the `train.csv` and `test.csv` files to your Google Drive.
2. Mount Google Drive in your Colab session.
3. Run the `insurancePyspark.ipynb` notebook step-by-step.
4. View ROC curves and compare model results.


> This project is an educational case study to demonstrate the capabilities of PySpark in handling large-scale data processing and classification tasks in the insurance domain.
