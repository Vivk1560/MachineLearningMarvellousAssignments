import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("student_performance_ml.csv")
featured_cols = ["StudyHours","Attendance","PreviousScore","AssignmentsCompleted","SleepHours"]
X = df[featured_cols]
Y = df["FinalResult"]
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

model1 = DecisionTreeClassifier(max_depth=1)
model1 = model1.fit(X_train,y_train)
y_pred = model1.predict(X_test)
accuracy1 = accuracy_score(y_test,y_pred)


model2 = DecisionTreeClassifier(max_depth=3)
model2 = model2.fit(X_train,y_train)
y_pred = model2.predict(X_test)
accuracy2 = accuracy_score(y_test,y_pred)

model3 = DecisionTreeClassifier()
model3 = model3.fit(X_train,y_train)
y_pred = model3.predict(X_test)
accuracy3 = accuracy_score(y_test,y_pred)

print(f"Testing Accuracy (max_depth=1)    : {accuracy1*100:.2f}%")
print(f"Testing Accuracy (max_depth=3)    : {accuracy2*100:.2f}%")
print(f"Testing Accuracy (max_depth=None) : {accuracy3*100:.2f}%")

#All three models achieved the same testing accuracy. 
#This indicates that the dataset is simple enough to be classified correctly even with a shallow decision tree.
#Increasing the tree depth did not improve the testing accuracy for this dataset.