'''
@author: Muhammad Bilal Naseer
Instructions: Enter the following command in the terminal to run this program [python Assignment3.py]
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from prettytable import PrettyTable
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc


# Reading the data from the files and storing them in the following dataframes.
df = pd.read_csv('A3_TrainData.tsv', sep='\t')
test_data = pd.read_csv('A3_TestData.tsv', sep='\t')

# Explore class distribution
class_distribution = df['label'].value_counts()
print(class_distribution)

# Prepare data and target
X = df.drop('label', axis=1)  # Replace 'target_column' with your actual target column name
y = df['label']

baseModel = LogisticRegression(random_state=42, max_iter=1000)

# Stratified K-Fold for cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = cross_val_score(baseModel, X, y, cv=cv, scoring='accuracy')

print("Mean Accuracy of Logistic Regression Model:", np.mean(results))

svm_model = SVC()

# Define  parameter grid
svm_param_grid = {'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']}

stratified_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Perform grid search for SVM with a smaller dataset
svm_grid_search = GridSearchCV(svm_model, svm_param_grid, cv=stratified_cv, scoring='accuracy', n_jobs=-1)
svm_grid_search.fit(X, y)

print("SVM - Best Hyperparameters:", svm_grid_search.best_params_)
print("Best accuracy for SVM:", svm_grid_search.best_score_)

rf_model = RandomForestClassifier()

# Define the parameter grid for Random Forest
rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]}

# Perform grid search for Random Forest
rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=stratified_cv, scoring='accuracy')
rf_grid_search.fit(X, y)

print("Random Forest - Best Hyperparameters:", rf_grid_search.best_params_)
print("Best accuracy for Random Forest:", rf_grid_search.best_score_)

# Define a function to plot precision-recall curve and ROC curve
def plot_curves(model, X, y, name):
    # Fit the model
    model.fit(X, y)
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y, model.predict_proba(X)[:, 1])
    pr_auc = auc(recall, precision)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y, model.predict_proba(X)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Plot precision-recall curve
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{name} Precision-Recall curve: AP={pr_auc:.2f}')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} ROC curve')
    plt.legend(loc="lower right")
    plt.show()

# Initialize the best models per method with the best hyper-parameter settings
best_logreg = LogisticRegression(max_iter=1000, random_state=42)
best_rf = RandomForestClassifier(**rf_grid_search.best_params_, random_state=42)
best_svm = SVC(**svm_grid_search.best_params_, random_state=42, probability=True)

# Plot precision-recall and ROC curves for the best models per method
plot_curves(best_logreg, X, y, "Logistic Regression")
plot_curves(best_rf, X, y, "Random Forest")
plot_curves(best_svm, X, y, "SVM")

# Compare the mean validation scores of the models
mean_validation_scores = {
    "Logistic Regression": np.mean(results),
    "Random Forest": np.mean(rf_grid_search.cv_results_['mean_test_score']),
    "SVM": np.mean(svm_grid_search.cv_results_['mean_test_score'])
}

best_model_name = max(mean_validation_scores, key=mean_validation_scores.get)
best_model = None

# Select the best model
if best_model_name == "Logistic Regression":
    best_model = best_logreg
elif best_model_name == "Random Forest":
    best_model = best_rf
elif best_model_name == "SVM":
    best_model = best_svm

print("Best Model Overall:", best_model_name)

# Create the final model using the best hyper-parameters and all of the training data
best_model.fit(X, y)

# Use the final model to predict the likelihood to belong to class 1 for the test instances
predictions = best_model.predict_proba(test_data)[:, 1]

# Write the predicted values to a file
with open('A3_predictions_202046892.txt', 'w') as f:
    for prediction in predictions:
        f.write(f'{prediction}\n')
