import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("student_performance_ml.csv")
passStudents = df[df["FinalResult"] == 1]
failStudents = df[df["FinalResult"] == 0]
plt.scatter(passStudents["SleepHours"],passStudents["FinalResult"],color="green",label="Pass",alpha=0.7,s=80)
plt.scatter(failStudents["SleepHours"],failStudents["FinalResult"],color="red",label="Fail",alpha=0.7,s=80)
plt.title("Sleep Hours vs Final Result")
plt.xlabel("Sleep Hours")
plt.ylabel("Final Result")
plt.yticks([0, 1], ["Fail", "Pass"])
plt.legend()
plt.grid(True)
plt.show()