# Diabetes Classification Model Evaluation

This repository contains a Jupyter notebook that demonstrates how to train and evaluate models for **diabetes classification**. The notebook loads and cleans the **diabetes dataset**, applies **SMOTE** for class balancing, and uses machine learning models like **Random Forest** and **XGBoost** for prediction. The models are evaluated using **classification metrics** like **recall**, **ROC AUC**, and **classification report**.

## Overview

The notebook covers the following steps:
1. **Data Loading and Cleaning**:
   - The **diabetes dataset** is loaded from a CSV file.
   - Missing values are handled and unnecessary columns are removed.
2. **Data Preprocessing**:
   - The data is split into training and testing sets.
   - **SMOTE** is applied to handle class imbalance by oversampling the minority class.
3. **Model Training**:
   - **Random Forest** and **XGBoost** models are trained for classification.
4. **Model Evaluation**:
   - The performance of each model is evaluated using metrics such as **recall**, **ROC AUC**, and the **classification report**.

## Files in this Repository

- **`diabetes_model_evaluation.ipynb`**: The main Jupyter notebook where the **diabetes dataset** is loaded, cleaned, and models are trained and evaluated.

## Features

- **Data Preprocessing**: Includes data cleaning, handling missing values, and applying **SMOTE** for class balancing.
- **Model Training**: Implements **Random Forest** and **XGBoost** models for diabetes classification.
- **Model Evaluation**: Evaluates models using metrics like **recall**, **ROC AUC**, and the **classification report**.
- **SMOTE**: Uses **SMOTE** to handle class imbalance in the dataset, which is crucial for improving the model's performance in real-world scenarios.

## Requirements

To run this notebook, you will need the following Python libraries:
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Imbalanced-learn**
- **Matplotlib**

You can install them using `pip`:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib
