import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("student_performance_ml.csv")
featured_cols = ["StudyHours","Attendance","PreviousScore","AssignmentsCompleted","SleepHours"]
X = df[featured_cols]
Y = df["FinalResult"]

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
model1 = DecisionTreeClassifier(max_depth=5)
model1 = model1.fit(X_train,y_train)
y_pred = model1.predict(X_test)
accuracy1 = accuracy_score(y_test,y_pred)

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=10)
model2 = DecisionTreeClassifier(max_depth=5)
model2 = model2.fit(X_train,y_train)
y_pred = model2.predict(X_test)
accuracy2 = accuracy_score(y_test,y_pred)

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model3 = DecisionTreeClassifier(max_depth=5)
model3 = model3.fit(X_train,y_train)
y_pred = model3.predict(X_test)
accuracy3 = accuracy_score(y_test,y_pred)

print(f"Testing Accuracy (randomstate = 0)    : {accuracy1*100:.2f}%")
print(f"Testing Accuracy (randomstate = 10)    : {accuracy2*100:.2f}%")
print(f"Testing Accuracy (randomstate = 42) : {accuracy3*100:.2f}%")