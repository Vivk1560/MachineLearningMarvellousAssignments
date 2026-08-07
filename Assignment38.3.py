import pandas as pd

datapath = "student_performance_ml.csv"
df = pd.read_csv(datapath)
print("Average Study Hours Of Students are:",df["StudyHours"].mean())
print("Average Attendance of students is:",df["Attendance"].mean())
print("Maximum from PreviousScore is:",df["PreviousScore"].max())
print("Minimum Sleep Hours are:",df["SleepHours"].min())