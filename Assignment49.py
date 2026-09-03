#=========================================================================
# Assignment 49
#=========================================================================
# Description:
# This program demonstrates basic statistical calculations, feature
# scaling, Euclidean distance calculation, and classification metrics.
#
# Topics Covered:
# 1. Calculate mean of a given dataset.
# 2. Calculate variance and standard deviation manually and using NumPy.
# 3. Apply StandardScaler for feature scaling.
# 4. Calculate Euclidean distance between two points before and after
#    applying feature scaling.
# 5. Calculate True Positives (TP), True Negatives (TN), False Positives
#    (FP), and False Negatives (FN).
# 6. Generate a classification report using scikit-learn.
#
# Libraries Used:
# - NumPy
# - Math
# - Scikit-learn
#=========================================================================

import numpy as np
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import euclidean_distances
from sklearn.metrics import classification_report

#-------------------------------------------------------------------------
# Question 1
#-------------------------------------------------------------------------

data = np.array([6,7,8,9,10,11,12])
print("Data is :",data)
avg = np.mean(data)
print("Mean of the data is :",avg)

#-------------------------------------------------------------------------
# Question 2
#-------------------------------------------------------------------------

summation = 0
for i in data:
    summation += (i-avg)**2
variance = summation/len(data)
print(f"Variance of data : {data} is : {variance}")
stdDev = sqrt(variance)
print(f"Standard Deviation is :",stdDev)
print("Variance using numpy of data is :",np.var(data))
print("Standard Deviation using numpy of data is :",np.std(data))

#-------------------------------------------------------------------------
# Question 3
#-------------------------------------------------------------------------

scaler = StandardScaler()
dataX = np.array([
    [25,20000],
    [30,40000],
    [35,80000]
])
dataX_scaled = scaler.fit_transform(dataX)
print("Data before scaling is:")
for i in dataX:
    print(i)
print("Scaled Data is:")
for i in dataX_scaled:
    print(i)

#-------------------------------------------------------------------------
# Question 4
#-------------------------------------------------------------------------

P1 = [[25,20000]]
P2 = [[35,80000]]
distance_before_scaling = euclidean_distances(P1,P2)
print(f"Distance between points {P1} and {P2} before scaling is : {distance_before_scaling}")
scalerX = StandardScaler()
scalerX.fit(dataX)
P1 = scalerX.transform(P1)
P2 = scalerX.transform(P2)
distance_after_scaling = euclidean_distances(P1,P2)
print(f"Distance between points {P1} and {P2} after scaling is : {distance_after_scaling}")

#-------------------------------------------------------------------------
# Question 8
#-------------------------------------------------------------------------

actual = np.array([1,1,1,1,0,0,0,0])
predicted = np.array([1,1,0,1,0,1,0,0])
TP = 0
TN = 0
FP = 0
FN = 0
for act,pred in zip(actual,predicted):
    if(act == 0 and pred == 0):
        TN += 1
    elif(act == 1 and pred ==1):
        TP += 1
    elif(act==0 and pred ==1):
        FP += 1
    elif(act==1 and pred==0):
        FN += 1

print(f"True Positives(TP) : {TP}")
print(f"True Negatives(TN) : {TN}")
print(f"False Positives(FP) : {FP}")
print(f"False Negatives(FN) : {FN}")

#-------------------------------------------------------------------------
# Question 9
#-------------------------------------------------------------------------

print("Classification Report:")
print(classification_report(actual, predicted))