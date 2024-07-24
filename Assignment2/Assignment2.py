'''
@author: Muhammad Bilal Naseer
Instructions: Enter the following command in the terminal to run this program [python Assignment2.py]
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from prettytable import PrettyTable


# Reading the data from the files and storing them in the following dataframes.
df = pd.read_csv('A2data.tsv', sep='\t')
df.drop(columns=['InstanceID'], inplace=True)

# Split features and target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Baseline regression model (simple linear regression)
baseline_model = LinearRegression()

# Cross-validation (e.g., 5-fold CV)
baseline_scores = cross_val_score(baseline_model, X, y, cv=5, scoring='neg_mean_squared_error')

baseline_model.fit(X, y)
y_pred_baseline = baseline_model.predict(X)

# Calculate RMSE
baseline_rmse = (-baseline_scores.mean()) ** 0.5
std_deviation = baseline_scores.std()

print("Baseline RMSE:", baseline_rmse)
print("Standard deviation of cross-validation scores:", std_deviation)

model1 = SVR()

model2 = RandomForestRegressor()

# Evaluate model performance
model1_scores = cross_val_score(model1, X, y, cv=5, scoring='neg_mean_squared_error')
model2_scores = cross_val_score(model2, X, y, cv=10, scoring='neg_mean_squared_error')

model1.fit(X, y)
y_pred_baseline = baseline_model.predict(X)

model2.fit(X, y)
y_pred_baseline = baseline_model.predict(X)

# Calculate RMSE for alternative models
model1_rmse = (-model1_scores.mean()) ** 0.5
model2_rmse = (-model2_scores.mean()) ** 0.5
std1 = model1_scores.std()
std2 = model2_scores.std()

print("SVR  RMSE:", model1_rmse,"\nRandom Forest Regressor RMSE:", model2_rmse)
print("\nSVR standard deviation", std1,"\nRandom Forest Regressor standard deviation",std2)

# Create a box plot to compare RMSE of all models
plt.figure(figsize=(10, 6))
plt.boxplot([baseline_scores, model1_scores, model2_scores], labels=['Baseline', 'SVR', 'Random Forest Regressor'])
plt.title('Comparison of Regression Models')
plt.xlabel('Models')
plt.show()

# Scatterplot of Predicted vs. Actual Values
plt.figure(figsize=(10, 6))
plt.scatter(y, model1.predict(X), color='green', label='Model 1')
plt.scatter(y, model2.predict(X), color='orange', label='Model 2')
plt.scatter(y, baseline_model.predict(X), color='red', label='Baseline')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line for reference
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. Actual Values')
plt.legend()
plt.show()


# Create a PrettyTable object
table = PrettyTable()

# Add columns
table.field_names = ["Models ", "Average RMSE", "Std. Deviation"]

# Add rows
table.add_row(["Linear Regression", 1.90, "±1.01"])
table.add_row(["SVR Model", 0.65, "±0.15"])
table.add_row(["Random Forest Regressor", 0.66, "±0.19"])

# Print the table
print(table)