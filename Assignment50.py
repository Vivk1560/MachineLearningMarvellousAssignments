#==========================================================================================================================
# Assignment : Breast Cancer Prediction
#
# Objective:
# To implement a Logistic Regression model to predict whether a breast tumor
# is Malignant or Benign using the Breast Cancer Wisconsin dataset.
#
# Dataset:
# Breast Cancer Wisconsin Dataset (loaded using load_breast_cancer from sklearn)
#
# Features (Independent Variables):
# 30 real-valued features representing various measurements of breast cancer tumors.
#
# Target (Dependent Variable):
# 0 - Malignant
# 1 - Benign
#
# Data Processing:
# - Load and inspect the dataset
# - Check for missing values and duplicate records
# - Analyze the correlation between the variables
# - Visualize feature correlations using a heatmap
# - Separate independent and dependent features
# - Split the data into training and testing sets
# - Standardize the independent features using StandardScaler
#
# Machine Learning Algorithm:
# Logistic Regression
#
# Model Evaluation:
# - Accuracy
# - Confusion Matrix
# - Precision
# - Recall
# - F1-Score
# - Display Logistic Regression coefficients
#
#==========================================================================================================================

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#--------------------------------------------------------------------------------------------------------------------------
# Step 1 : Load Data
#--------------------------------------------------------------------------------------------------------------------------

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Target"] = data.target
print("Initial Entries From The Dataset Are:")
print(df.head())
print()
print("Shape of the dataset is:")
print(df.shape)
print()
print("Description of Dataset is:")
print(df.describe())
print()
print("Information of dataset:")
print(df.info())
print()
print("Columns of dataset:")
print(df.columns)

#--------------------------------------------------------------------------------------------------------------------------
# Step 2 : Process Data
#--------------------------------------------------------------------------------------------------------------------------

print(df.head())
print()
print("Checking for missing values....")
print(df.isna().sum())
print()
print("Number of duplicate records:")
print(df.duplicated().sum())
print()
print("Correlation Matrix is:")
print(df.corr())
print()
print("Displaying Feature Correlation Heatmap:")
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------------------------------------------------------
# Step 3 : Separate Dependent And Independent Features
#--------------------------------------------------------------------------------------------------------------------------

X = df.drop(columns=["Target"])
Y = df["Target"]
print()
print("Independent Features:")
print(X.head())
print()
print("Dependent Features:")
print(Y.head())

#--------------------------------------------------------------------------------------------------------------------------
# Step 4 : Split training and testing data
#--------------------------------------------------------------------------------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=42, test_size=0.5)

#--------------------------------------------------------------------------------------------------------------------------
# Step 5 : Scaling the Independent Features
#--------------------------------------------------------------------------------------------------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#--------------------------------------------------------------------------------------------------------------------------
# Step 6 : Model Creation and Training
#--------------------------------------------------------------------------------------------------------------------------

model = LogisticRegression()
model = model.fit(X_train,Y_train)

#--------------------------------------------------------------------------------------------------------------------------
# Step 7 : Model Testing
#--------------------------------------------------------------------------------------------------------------------------

Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test,Y_pred)

#--------------------------------------------------------------------------------------------------------------------------
# Step 8 : Displaying Results
#--------------------------------------------------------------------------------------------------------------------------

print()
for i in range(len(Y_test)):
    print(
        f"Record {i + 1}: "
        f"Predicted = {Y_pred[i]}, "
        f"Expected = {Y_test.iloc[i]}"
    )
features = X.columns
print()
print("Coefficients of the model are:")
for feature, coefficient in zip(features, model.coef_[0]):
    print(f"{feature}: {coefficient}")
print()
print("Intercept of the model is:")
print(model.intercept_)

#--------------------------------------------------------------------------------------------------------------------------
# Step 9 : Model Performance Display
#--------------------------------------------------------------------------------------------------------------------------

print()
print("Model Performance:")
print("Accuracy :",(accuracy*100))
print()
print("Classification Report:")
print(classification_report(Y_test,Y_pred))
print()
print("Confusion Matrix :")
print(confusion_matrix(Y_test,Y_pred))
ConfusionMatrixDisplay.from_predictions(
    Y_test,
    Y_pred,
    display_labels=data.target_names
)
plt.title("Confusion Matrix")
plt.show()

#--------------------------------------------------------------------------------------------------------------------------
# Step 10 : Observations and Conclusion
#--------------------------------------------------------------------------------------------------------------------------

print()
print("Observations and Conclusion:")
print("The dataset contains 569 records and 30 independent features.")
print("No missing values or duplicate records were found.")
print("The features were standardized using StandardScaler.")
print("Feature correlation analysis shows that several features are strongly correlated.")
print(f"The Logistic Regression model achieved an accuracy of {accuracy * 100:.2f}%.")
print("The classification report shows high precision, recall and F1-score for both classes.")
print("The confusion matrix shows that the model correctly classified most of the test records.")
print("Therefore, the trained Logistic Regression model performed well on the given test data.")