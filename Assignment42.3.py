from sklearn.neighbors import KNeighborsClassifier

def main():
    X = [
        [2, 60],
        [5, 80],
        [6, 85],
        [1, 50]
    ]
    Y = ["Fail", "Pass", "Pass", "Fail"]
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, Y)
    studyHours = int(input("Enter Study Hours: "))
    attendance = int(input("Enter Attendance: "))
    prediction = model.predict([[studyHours, attendance]])
    print(f"Predicted Result: {prediction[0]}")

if __name__ == "__main__":
    main()