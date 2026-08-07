import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("student_performance_ml.csv")
model = DecisionTreeClassifier(max_depth=5)
featured_cols = ["StudyHours","Attendance","PreviousScore","AssignmentsCompleted","SleepHours"]
X = df[featured_cols]
Y = df["FinalResult"]
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model = model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Expected Answers:")
print(y_test)
print("Predicted Answers:")
print(y_pred)
old_accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy of old model is:{old_accuracy*100}%")

df["PerformanceIndex"] = (df["StudyHours"] * 2) + df["Attendance"]
modelX = DecisionTreeClassifier(max_depth=5)
featured_cols = ["StudyHours","Attendance","PreviousScore","AssignmentsCompleted","SleepHours","PerformanceIndex"]
X = df[featured_cols]
Y = df["FinalResult"]
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
modelX = modelX.fit(X_train,y_train)
y_pred = modelX.predict(X_test)
new_accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy of new model is:{new_accuracy*100}%")

if new_accuracy > old_accuracy:
    print("Accuracy improved after adding PerformanceIndex.")
elif new_accuracy < old_accuracy:
    print("Accuracy decreased after adding PerformanceIndex.")
else:
    print("Accuracy remained the same after adding PerformanceIndex.")