import pandas as pd
import matplotlib.pyplot as plt

datapath = "student_performance_ml.csv"
df = pd.read_csv(datapath)
plt.boxplot(df["Attendance"])
plt.title("Boxplot of Attendance")
plt.ylabel("Attendance")
plt.show()
#The boxplot shows that the Attendance values range from 60 to 96. 
#The median attendance is approximately 80
#indicating that half of the students have attendance above this value and half below it. 
#The middle 50% of attendance values lie roughly between 70 and 89. 
#No outliers are present, as there are no points outside the whiskers.