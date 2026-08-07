import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
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
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy of model with SleepHours is:{accuracy*100}%")


model2 = DecisionTreeClassifier(max_depth=5)
featured_cols = ["StudyHours","Attendance","PreviousScore","AssignmentsCompleted"]
X = df[featured_cols]
Y = df["FinalResult"]
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model2 = model2.fit(X_train,y_train)
y_pred = model2.predict(X_test)
accuracy2 = accuracy_score(y_test,y_pred)
print(f"Accuracy of model without SleepHours is:{accuracy2*100}%")