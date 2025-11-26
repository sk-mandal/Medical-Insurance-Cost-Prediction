# ---------------------------------------------------------
# Medical Insurance Cost Prediction - Regression Project
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

df = pd.read_csv("Medical_Insurance_cost_prediction.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nShape of the dataset (rows, columns):", df.shape)

print("\nColumn data types:")
print(df.dtypes)

print("\nGeneral dataset info:")
print(df.info())

print("\nSummary statistics (numeric features):")
print(df.describe())

# Distribution of numerical columns
numeric_columns = ["age", "bmi", "children", "charges"]

print("\nDistribution of numerical features:")

plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()


# Count plots for categorical columns
categorical_columns = ["sex", "smoker", "region"]

print("\nDistribution of categorical features:")

plt.figure(figsize=(12, 6))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(1, 3, i)
    sns.countplot(data=df, x=col)
    plt.title(f"Count Plot of {col}")
plt.tight_layout()
plt.show()


# Correlation Heatmap (numeric variables only)
numeric_df = df.select_dtypes(include=[np.number])

print("\nCorrelation heatmap for numeric features:")

plt.figure(figsize=(6, 4))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Features Only)")
plt.show()

# Checking for missing values
print("\nChecking missing values:")
print(df.isnull().sum())


# Boxplots for outlier detection (numeric features only)
numeric_columns = ["age", "bmi", "children", "charges"]

plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()


# Detect outlier counts using IQR method
print("\nOutlier count using IQR method:")

for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()

    print(f"{col}: {outliers} outliers detected")


# Make a copy of the original dataset
processed_df = df.copy()


# Encode categorical columns
# Binary columns first
processed_df["sex"] = processed_df["sex"].map({"male": 1, "female": 0})
processed_df["smoker"] = processed_df["smoker"].map({"yes": 1, "no": 0})

# One-hot encode region (multiple categories)
processed_df = pd.get_dummies(processed_df, columns=["region"], drop_first=True)


# Checking skewness before scaling
print("\nSkewness before scaling:")
print(processed_df.skew())


# log-transform target if highly skewed
# We will check threshold first
if processed_df["charges"].skew() > 1:
    print("\nTarget variable 'charges' is highly skewed. Log transformation recommended.")
    processed_df["charges"] = np.log1p(processed_df["charges"])
else:
    print("\nSkew is acceptable. No log transform applied.")


# Split features and target
X = processed_df.drop("charges", axis=1)
y = processed_df["charges"]

# Scaling (only numeric predictors)
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nPreprocessing completed.")
print("Shape of final feature set:", X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Training and testing sets created.")

# Defining Models in Dictionary
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}

# Train, Predict and Evaluate using Loop
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)

    # Overfitting check: compare train and test RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

    results.append([name, train_rmse, rmse, r2])

    print(f"\n{name} model evaluation completed.")


print("\nAll models successfully trained.")

# Create a DataFrame from collected results
comparison_df = pd.DataFrame(results, columns=[
    "Model", "Train_RMSE", "Test_RMSE", "R2_Score"
])

# Add Overfitting flag (Train_RMSE much lower than Test_RMSE)
comparison_df["Overfitting"] = np.where(
    comparison_df["Train_RMSE"] < (comparison_df["Test_RMSE"] * 0.7), "Yes", "No"
)

# Sort by highest R2 Score (best model first)
comparison_df = comparison_df.sort_values(by="R2_Score", ascending=False)

print("\nModel Comparison Table:\n")
print(comparison_df)

plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="R2_Score", data=comparison_df)
plt.xticks(rotation=45)
plt.title("Model Performance Comparison (R2 Score)")
plt.show()

# Parameter grid for Random Forest
rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10]
}

rf = RandomForestRegressor(random_state=42)

rf_grid = GridSearchCV(
    estimator=rf,
    param_grid=rf_params,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

# Parameter grid for Gradient Boosting
gb_params = {
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200, 300],
    "max_depth": [2, 3, 5]
}

gb = GradientBoostingRegressor(random_state=42)

gb_grid = GridSearchCV(
    estimator=gb,
    param_grid=gb_params,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

gb_grid.fit(X_train, y_train)

print("\nBest Parameters Found:")
print("\nRandom Forest:", rf_grid.best_params_)
print("Gradient Boosting:", gb_grid.best_params_)

# Evaluate best tuned models
models_tuned = {
    "Random Forest Tuned": rf_grid.best_estimator_,
    "Gradient Boosting Tuned": gb_grid.best_estimator_,
}

results_tuned = []

for name, model in models_tuned.items():
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2 = r2_score(y_test, y_pred_test)

    results_tuned.append([name, rmse_train, rmse_test, r2])

print("\nTuned Model Results:")
print(pd.DataFrame(results_tuned, columns=["Model", "Train_RMSE", "Test_RMSE", "R2_Score"]))

# Convert tuned results into DataFrame
tuned_df = pd.DataFrame(results_tuned, columns=["Model", "Train_RMSE", "Test_RMSE", "R2_Score"])

# Add indicator for tuned models
tuned_df["Model"] = tuned_df["Model"] + " (Tuned)"

# Combine original and tuned results
final_results = pd.concat([comparison_df, tuned_df], ignore_index=True)

# Recalculate overfitting label for tuned models
final_results["Overfitting"] = np.where(
    final_results["Train_RMSE"] < (final_results["Test_RMSE"] * 0.7), "Yes", "No"
)

# Sort by highest R2 Score
final_results = final_results.sort_values(by="R2_Score", ascending=False)

print("\n--- Final Model Comparison Table ---\n")
print(final_results)

# Save the best model
best_model = gb_grid.best_estimator_   # Gradient Boosting Tuned (final model)

with open("best_insurance_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

# Save the scaler used during preprocessing
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully.")
