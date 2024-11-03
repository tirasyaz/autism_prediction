import pandas as pd
import numpy as np
import streamlit as st
df = pd.read_csv('https://raw.githubusercontent.com/tirasyaz/autism_prediction/refs/heads/main/Toddler%20Autism%20dataset%20July%202018.csv')

df.head()

st.write(df)

# Detecting any missing values
df.isna().sum()

df = pd.read_csv('https://raw.githubusercontent.com/amirulerfn/jie43202/refs/heads/main/Toddler%20Autism%20dataset%20July%202018.csv')

# List of columns to remove
columns_to_remove = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who completed the test']

# Drop the columns
df = df.drop(columns_to_remove, axis=1)

# Save the modified DataFrame if needed
df.to_csv('modified_dataset.csv', index=False)

df.head()

import pandas as pd

df1 = pd.read_csv('modified_dataset.csv')
bools1 = ['Class/ASD Traits ']

for col in bools1:
  df1[col] = df1[col].replace({'Yes': 1, 'No': 0})

# Save the modified DataFrame
df1.to_csv('modified_dataset.csv', index=False)

# Now df1 is modified with Yes/No converted to 1/0
df1.head()

# Calculate mean and standard deviation
mean = df1.mean()
std_dev = df1.std()

# Set a threshold for outliers
threshold = 3
lower_bound = mean - threshold * std_dev
upper_bound = mean + threshold * std_dev

# Find outliers
outliers = df[(df1 < lower_bound) | (df1 > upper_bound)]

print("Outliers:")
print(outliers)

df2 = pd.read_csv('modified_dataset.csv')
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, PowerTransformer, Normalizer

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

X = df2.drop('Class/ASD Traits ',axis=1)
y = df2.iloc[:,-2]
X.columns = X.columns.astype(str)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models
models = [
    RandomForestClassifier(n_estimators=50),
    SGDClassifier(),
    SVC(C=1, kernel='rbf', degree=3, gamma='scale')
]
names = ['Random Forest', 'SGD Classifier', 'SVM']

# Create an empty dictionary to store the results
results = {'Model': [], 'Accuracy': [], 'Sensitivity': [], 'Specificity': [], 'Mean Score': []}

# Plot settings
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for counter, model in enumerate(models):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate confusion matrix values
    cm = metrics.confusion_matrix(y_test, y_pred)

    # Calculate accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)

    # Handle multiclass confusion matrix (same logic from original code)
    sensitivity = np.diag(cm) / np.sum(cm, axis=1)
    specificity = []
    for i in range(cm.shape[0]):
        temp = np.delete(cm, i, axis=0)
        temp = np.delete(temp, i, axis=1)
        tn = np.sum(temp)
        fp = np.sum(cm[i, :]) - cm[i, i]
        specificity.append(tn / (tn + fp))

    # Calculate mean scores
    mean_sensitivity = np.mean(sensitivity)
    mean_specificity = np.mean(specificity)
    mean_score = (accuracy + mean_sensitivity + mean_specificity) / 3

    # Store the results in the dictionary
    results['Model'].append(names[counter])
    results['Accuracy'].append(accuracy)
    results['Sensitivity'].append(mean_sensitivity)
    results['Specificity'].append(mean_specificity)
    results['Mean Score'].append(mean_score)

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[counter])
    axes[counter].set_title(f'{names[counter]} Confusion Matrix')
    axes[counter].set_xlabel('Predicted')
    axes[counter].set_ylabel('Actual')

# Create a dataframe from the results dictionary
scores = pd.DataFrame(results)

# Sort dataframe by mean score
sorted_df2 = scores.sort_values(by='Mean Score', ascending=False)

# Show plot
plt.tight_layout()
plt.show()

# Display sorted scores
sorted_df2

st.pyplot(plt.gcf())


