# 🏥 Medical Insurance Cost Prediction Using Machine Learning

This project predicts the estimated **medical insurance cost** for a patient based on demographic and lifestyle details.  
It includes the complete workflow from **data exploration → preprocessing → model building → evaluation → tuning → deployment via Streamlit UI.**

---

## Project Overview

Health insurance pricing varies based on parameters like **age, BMI, smoking habits, dependents, and region**.  
The goal of this project is to build a machine learning model that can **accurately estimate insurance charges** using patient data.

This project includes:

- Exploratory Data Analysis (EDA)
- Handling missing values and outliers
- Encoding categorical variables & scaling numerical features
- Training and evaluating multiple algorithms
- Hyperparameter tuning
- Saving the best model using `pickle`
- Building a Streamlit web application for real-time prediction

---

## Dataset Information

| Column | Description |
|--------|------------|
| age | Age of the insured person |
| sex | Gender (male/female) |
| bmi | Body Mass Index |
| children | Number of dependents |
| smoker | Whether the person smokes |
| region | Residential region in the US |
| charges | **Target variable**: medical insurance cost |

---

## Exploratory Data Analysis (EDA)

### Key Insights:

- **Smokers have significantly higher insurance charges** compared to non-smokers.
- Charges increase with **age and BMI**.
- Region shows **minimal correlation** with cost.
- Distribution of `charges` is **right-skewed**, so log transformation improved model performance.

### Visualizations Used:

- Histogram plots  
- Count plots  
- Correlation heatmap  
- Boxplots (outlier inspection)

---

## Machine Learning Models Tested

| Model | Status |
|--------|--------|
| Linear Regression | Tested |
| Decision Tree Regressor | Tested |
| KNN Regressor | Tested |
| Support Vector Regressor (SVR) | Tested |
| Random Forest Regressor | Tested |
| **Gradient Boosting Regressor** | Best Model |
| GridSearchCV | Hyperparameter tuning applied |

The **tuned Gradient Boosting Regressor** achieved the best results based on RMSE and R² score.

---

## Model Performance Summary

| Model | Train RMSE | Test RMSE | R² Score | Overfitting |
|-------|------------|-----------|----------|-------------|
| **Gradient Boosting (Tuned)** | Best | Best | Highest | No |

---

## Feature Engineering & Preprocessing

- Label encoding applied to: `sex`, `smoker`
- One-hot encoding applied to: `region`
- Numerical features scaled using **StandardScaler**
- Target variable optionally log-transformed for better learning
- Final trained model and scaler exported using pickle:

