import pandas as pd
import matplotlib.pyplot as plt

datapath = "student_performance_ml.csv"
df = pd.read_csv(datapath)
plt.hist(df["StudyHours"], bins=8,edgecolor = "black",rwidth=0.5) #as studyhrs ranges from 1 to 8.5 we have created 8 parts in the graph
plt.title("Histogram of Study Hours")
plt.xlabel("Study Hours")
plt.ylabel("Number of Students")
plt.show()