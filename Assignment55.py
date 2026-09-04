# --------------------------------------------------------------------------------------------------------------------------
# Assignment 55 : Customer Loan Approval Prediction Using Ensemble Learning
# --------------------------------------------------------------------------------------------------------------------------
# Description:
# This program predicts customer loan approval using multiple classification algorithms
# and compares their performance with Soft and Hard Voting Classifiers.
#
# Models Used:
# - Logistic Regression
# - Decision Tree Classifier
# - K-Nearest Neighbors (KNN)
# - Soft Voting Classifier
# - Hard Voting Classifier
#
# The dataset is first explored and processed, followed by train-test splitting and
# feature scaling. Individual models and ensemble models are then trained, tested,
# and evaluated using accuracy scores.
#
# Objective:
# To compare individual classification models with ensemble learning techniques
# and identify the best-performing model based on test accuracy.
# --------------------------------------------------------------------------------------------------------------------------

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns


#--------------------------------------------------------------------------------------------------------------------------
# Step 1 : Load Data
#--------------------------------------------------------------------------------------------------------------------------

df = pd.read_csv("Customer_Loan_Approval.csv")
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

X = df.drop(columns=["LoanApproved"])
Y = df["LoanApproved"]
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
model_decision = DecisionTreeClassifier(random_state=42, max_depth=7)
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

#--------------------------------------------------------------------------------------------------------------------------
# Step 7 : Model Testing
#--------------------------------------------------------------------------------------------------------------------------

Y_pred_logistic = model_logistic.predict(X_test)
Y_pred_decision = model_decision.predict(X_test)
Y_pred_knn = model_knn.predict(X_test)
Y_pred_soft = model_soft.predict(X_test)
Y_pred_hard = model_hard.predict(X_test)

#--------------------------------------------------------------------------------------------------------------------------
# Step 8 : Calculate Accuracies
#--------------------------------------------------------------------------------------------------------------------------

acc_decision = accuracy_score(Y_test,Y_pred_decision)
acc_logistic = accuracy_score(Y_test,Y_pred_logistic)
acc_knn = accuracy_score(Y_test,Y_pred_knn)
acc_soft = accuracy_score(Y_test,Y_pred_soft)
acc_hard = accuracy_score(Y_test,Y_pred_hard)

#--------------------------------------------------------------------------------------------------------------------------
# Step 9 : Display Results and Compare
#--------------------------------------------------------------------------------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Soft Voting Classifier",
        "Hard Voting Classifier"
    ],
    "Accuracy": [
        acc_logistic,
        acc_decision,
        acc_knn,
        acc_soft,
        acc_hard
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
# Step 10 : Displaying Various Model Accuracies Graphically
#--------------------------------------------------------------------------------------------------------------------------

plt.bar(results["Model"],results["Accuracy"]*100)
plt.xlabel("Models")
plt.ylabel("Accuracies")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------------------------------------
# Verification:
# A separate Logistic Regression model is trained from scratch to verify that the 100% accuracy
# obtained by the original Logistic Regression model is not affected by the cloning behavior of
# VotingClassifier. The independently trained model is evaluated on the same test set for comparison.
# --------------------------------------------------------------------------------------------------------------------------

model_logistic2 = LogisticRegression(max_iter=1000)
model_logistic2 = model_logistic2.fit(X_train,Y_train)
y_pred = model_logistic2.predict(X_test)
print("Accuracy of a separate logistic regression model :",accuracy_score(Y_test,y_pred))