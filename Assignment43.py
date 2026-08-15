from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def CheckAccuracy(X, Y, k):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=0.20, random_state=42
    )
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    return accuracy


def main():
    df = pd.read_csv("MarvellousInfosystems_PlayPredictor.csv")
    feature_columns = ["Wether","Temperature"]
    print("Starting entries of dataset are:")
    print(df.head())
    print("Information about data:")
    print(df.info())
    print("Shape of dataset:")
    print(df.shape)
    if df.isnull().values.any():
        print("Null values found. Removing rows...")
        df = df.dropna()
    else:
        print("No null values found.")
    wether_encoder = LabelEncoder()
    df["Wether"] = wether_encoder.fit_transform(df["Wether"])
    temp_encoder = LabelEncoder()
    df["Temperature"] = temp_encoder.fit_transform(df["Temperature"])
    play_encoder = LabelEncoder()
    df["Play"] = play_encoder.fit_transform(df["Play"])
    X = df[feature_columns]
    Y = df["Play"]
    print(X)
    print(Y)
    model = KNeighborsClassifier(n_neighbors=3)
    model = model.fit(X,Y)
    print("Model Trained!")
    accuracy = CheckAccuracy(X,Y,3)
    print("Accuracy of model is:", accuracy * 100, "%")
    print("We can see it's predictions...")
    w = input("Enter Weather (Sunny,Overcast,Rainy): ").lower()
    if w in ["sunny","overcast","rainy"]:
        w = w.capitalize()
        w = wether_encoder.transform([w])[0]
    else:
        print("Invalid input for weather!")
        return
    
    t = input("Enter Temperature (Hot,Mild,Cool): ").lower()
    if t in ["hot","mild","cool"]:
        t = t.capitalize()
        t = temp_encoder.transform([t])[0]
    else:
        print("Invalid input for temperature!")
        return
    input_data = pd.DataFrame([[w,t]],columns=feature_columns)
    output = model.predict(input_data)
    output = play_encoder.inverse_transform(output)
    print("Model's Prediction for user input is:",output[0])
    
if __name__ == "__main__":
    main()