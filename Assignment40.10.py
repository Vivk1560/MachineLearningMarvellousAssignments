import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("student_performance_ml.csv")
model = DecisionTreeClassifier(max_depth=None)
featured_cols = ["StudyHours","Attendance","PreviousScore","AssignmentsCompleted","SleepHours"]
X = df[featured_cols]
Y = df["FinalResult"]
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model = model.fit(X_train,y_train)
y_pred = model.predict(X_test)
testaccuracy = accuracy_score(y_test,y_pred)
y_train_pred = model.predict(X_train)
trainaccuracy = accuracy_score(y_train,y_train_pred)
print(f"Testing Accuracy of the model is: {testaccuracy*100}%")
print(f"Training Accuracy of the model is: {trainaccuracy*100}%")
if trainaccuracy>testaccuracy+0.05:
    print("Model Is Overfitting")
elif trainaccuracy<testaccuracy-0.05:
    print("Model Is Underfitting")
else:
    print("Model is Well Fitted")