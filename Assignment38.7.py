import pandas as pd
import matplotlib.pyplot as plt

datapath = "student_performance_ml.csv"
df = pd.read_csv(datapath)
passStudents = df[df["FinalResult"] == 1]
failStudents = df[df["FinalResult"] == 0]
plt.scatter(passStudents["StudyHours"],passStudents["PreviousScore"],color = "green", label = "Pass")
plt.scatter(failStudents["StudyHours"],failStudents["PreviousScore"],color = "red", label = "Fail")
plt.title("Study Hours vs Previous Score")
plt.xlabel("Study Hours")
plt.ylabel("Previous Score")
plt.legend()
plt.grid(True)
plt.show()