#--------------------------------------------------------------------------------------------------------------------------
# Assignment 56 : Fraudulent Transaction Detection
#--------------------------------------------------------------------------------------------------------------------------
#
# Objective:
# To detect potentially fraudulent financial transactions using various
# machine learning classification algorithms and ensemble learning techniques.
#
# Dataset:
# Fraudulent_Transaction_Detection.csv
#
# Dataset Features:
# 1. TransactionAmount
# 2. TransactionHour
# 3. AccountAgeMonths
# 4. PreviousTransactions
# 5. LocationDifferenceKm
# 6. DeviceType
# 7. FailedLoginAttempts
#
# Target Variable:
# Fraud
# 0 -> Normal Transaction
# 1 -> Fraudulent Transaction
#
# Data Processing:
# - Checked dataset shape, description, information and column names.
# - Checked for missing values and duplicate records.
# - Generated a correlation matrix and feature correlation heatmap.
# - Separated independent features (X) and dependent feature (Y).
# - Split the dataset into training and testing sets using stratified sampling.
# - Applied StandardScaler to scale the independent features.
#
# Models Implemented:
#
# 1. Logistic Regression
#    - Used as a linear classification model.
#
# 2. Decision Tree Classifier
#    - Used with max_depth=3 to control tree complexity.
#
# 3. K-Nearest Neighbors (KNN)
#    - Used with n_neighbors=7.
#
# 4. Soft Voting Classifier
#    - Combines Logistic Regression, Decision Tree and KNN.
#    - Uses predicted probabilities for final voting.
#
# 5. Hard Voting Classifier
#    - Combines Logistic Regression, Decision Tree and KNN.
#    - Uses majority voting for the final prediction.
#
# 6. Bagging Classifier
#    - Uses Decision Tree as the base estimator.
#    - Trains multiple estimators on bootstrap samples.
#    - Independent estimators can be trained in parallel using n_jobs=-1.
#
# 7. AdaBoost Classifier
#    - Uses shallow Decision Trees as base estimators.
#    - Builds models sequentially and gives more importance to observations
#      that were incorrectly classified by previous learners.
#
# 8. Random Forest Classifier
#    - Uses an ensemble of multiple Decision Trees.
#    - Introduces randomness while building individual trees and combines
#      their predictions.
#
# Model Evaluation:
# - Accuracy is calculated for all models and used for the initial comparison.
# - Classification Report is displayed for every model, containing:
#       Precision
#       Recall
#       F1 Score
# - Confusion Matrix is displayed for every model.
# - A graphical comparison of model accuracies is also displayed.
#
# Final Comparison:
# The models are compared based on their classification performance to identify
# the best-performing model for fraudulent transaction detection.
#
#--------------------------------------------------------------------------------------------------------------------------

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier,AdaBoostClassifier,BaggingClassifier,RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns


#--------------------------------------------------------------------------------------------------------------------------
# Step 1 : Load Data
#--------------------------------------------------------------------------------------------------------------------------

df = pd.read_csv("Fraudulent_Transaction_Detection.csv")
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
df.info()
print()
print("Columns of dataset:")
print(df.columns)

#--------------------------------------------------------------------------------------------------------------------------
# Step 2 : Process Data
#--------------------------------------------------------------------------------------------------------------------------

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

X = df.drop(columns=["Fraud"])
Y = df["Fraud"]
print()
print("Independent Features:")
print(X.head())
print()
print("Dependent Features:")
print(Y.head())

#--------------------------------------------------------------------------------------------------------------------------
# Step 4 : Split training and testing data
#--------------------------------------------------------------------------------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=42, test_size=0.5,stratify=Y)

#--------------------------------------------------------------------------------------------------------------------------
# Step 5 : Scaling the Independent Features
#--------------------------------------------------------------------------------------------------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#--------------------------------------------------------------------------------------------------------------------------
# Step 6 : Model Creation and Training
#--------------------------------------------------------------------------------------------------------------------------

model_logistic = LogisticRegression(max_iter=1000)
model_logistic = model_logistic.fit(X_train,Y_train)
model_decision = DecisionTreeClassifier(random_state=42, max_depth=3)
model_decision = model_decision.fit(X_train,Y_train)
model_knn = KNeighborsClassifier(n_neighbors=7)
model_knn = model_knn.fit(X_train,Y_train)
model_soft = VotingClassifier(
    estimators=[
        ("logistic",model_logistic),
        ("decision_tree",model_decision),
        ("knn",model_knn)
    ],
    voting = "soft"
)
model_soft = model_soft.fit(X_train,Y_train)
model_hard = VotingClassifier(
    estimators=[
        ("logistic",model_logistic),
        ("decision_tree",model_decision),
        ("knn",model_knn)
    ],
    voting = "hard"
)
model_hard = model_hard.fit(X_train,Y_train)
model_bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    n_jobs=-1,
    random_state=42
)
model_bagging = model_bagging.fit(X_train,Y_train)
model_adaBoost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
)
model_adaBoost = model_adaBoost.fit(X_train,Y_train)
model_random = RandomForestClassifier(
    n_estimators=1000,
    random_state=42
)
model_random = model_random.fit(X_train,Y_train)

#--------------------------------------------------------------------------------------------------------------------------
# Step 7 : Model Testing
#--------------------------------------------------------------------------------------------------------------------------

Y_pred_logistic = model_logistic.predict(X_test)
Y_pred_decision = model_decision.predict(X_test)
Y_pred_knn = model_knn.predict(X_test)
Y_pred_soft = model_soft.predict(X_test)
Y_pred_hard = model_hard.predict(X_test)
Y_pred_bagging = model_bagging.predict(X_test)
Y_pred_boosting = model_adaBoost.predict(X_test)
Y_pred_random = model_random.predict(X_test)

#--------------------------------------------------------------------------------------------------------------------------
# Step 8 : Calculate Accuracies
#--------------------------------------------------------------------------------------------------------------------------

acc_decision = accuracy_score(Y_test,Y_pred_decision)
acc_logistic = accuracy_score(Y_test,Y_pred_logistic)
acc_knn = accuracy_score(Y_test,Y_pred_knn)
acc_soft = accuracy_score(Y_test,Y_pred_soft)
acc_hard = accuracy_score(Y_test,Y_pred_hard)
acc_bagg = accuracy_score(Y_test,Y_pred_bagging)
acc_boost = accuracy_score(Y_test,Y_pred_boosting)
acc_random = accuracy_score(Y_test,Y_pred_random)

#--------------------------------------------------------------------------------------------------------------------------
# Step 9 : Display Results and Compare
#--------------------------------------------------------------------------------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Soft Voting Classifier",
        "Hard Voting Classifier",
        "Bagging Classifier",
        "AdaBoost Classifier",
        "Random Forest Classifier"
    ],
    "Accuracy": [
        acc_logistic,
        acc_decision,
        acc_knn,
        acc_soft,
        acc_hard,
        acc_bagg,
        acc_boost,
        acc_random
    ]
})
results = results.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
print("----------------------------------------------------------------------")
print("                    MODEL ACCURACY COMPARISON")
print("----------------------------------------------------------------------")
for index, row in results.iterrows():
    print(f"{index + 1}. {row['Model']:<25} : {row['Accuracy']:.4f} ({row['Accuracy'] * 100:.2f}%)")
print("----------------------------------------------------------------------")
best_model = results.iloc[0]
print(f"Best Performing Model : {best_model['Model']}")
print(f"Best Accuracy         : {best_model['Accuracy']:.4f} ({best_model['Accuracy'] * 100:.2f}%)")
print("----------------------------------------------------------------------")

#--------------------------------------------------------------------------------------------------------------------------
# Step 10 : Displaying Classification Reports and Confusion Matrices
#--------------------------------------------------------------------------------------------------------------------------

models = {
    "Logistic Regression": Y_pred_logistic,
    "Decision Tree": Y_pred_decision,
    "K-Nearest Neighbors": Y_pred_knn,
    "Soft Voting Classifier": Y_pred_soft,
    "Hard Voting Classifier": Y_pred_hard,
    "Bagging Classifier": Y_pred_bagging,
    "AdaBoost Classifier": Y_pred_boosting,
    "Random Forest Classifier": Y_pred_random
}

for model_name, Y_pred in models.items():

    print()
    print("----------------------------------------------------------------------")
    print(f"                    {model_name.upper()}")
    print("----------------------------------------------------------------------")

    print("Classification Report:")
    print(classification_report(
        Y_test,
        Y_pred,
        target_names=["Normal Transaction", "Fraudulent Transaction"]
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, Y_pred))

#--------------------------------------------------------------------------------------------------------------------------
# Step 11 : Displaying Various Model Accuracies Graphically
#--------------------------------------------------------------------------------------------------------------------------

plt.bar(results["Model"],results["Accuracy"]*100)
plt.xlabel("Models")
plt.ylabel("Accuracies")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

