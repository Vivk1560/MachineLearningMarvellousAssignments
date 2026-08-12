from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def main():
    df = pd.read_csv("WinePredictor.csv")
    features = [
                "Alcohol",
                "Malic acid",
                "Ash",
                "Alcalinity of ash",
                "Magnesium",
                "Total phenols",
                "Flavanoids",
                "Nonflavanoid phenols",
                "Proanthocyanins",
                "Color intensity",
                "Hue",
                "OD280/OD315 of diluted wines",
                "Proline"
            ]
    X = df[features]
    Y = df["Class"]
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=42)
    model1 = DecisionTreeClassifier(max_depth=5,random_state=42)
    model1 = model1.fit(X_train,y_train)
    y_pred = model1.predict(X_test)
    accuracy1 = accuracy_score(y_test,y_pred)
    print(f"Accuracy of the model Decision Tree Classifier and max_depth=5 is: {accuracy1*100}%")

    model2 = DecisionTreeClassifier(max_depth=3,random_state=42)
    model2 = model2.fit(X_train,y_train)
    y_pred = model2.predict(X_test)
    accuracy2 = accuracy_score(y_test,y_pred)
    print(f"Accuracy of the model Decision Tree Classifier and max_depth=3 is: {accuracy2*100}%")

    model3 = DecisionTreeClassifier(max_depth=7,random_state=42)
    model3 = model3.fit(X_train,y_train)
    y_pred = model3.predict(X_test)
    accuracy3 = accuracy_score(y_test,y_pred)
    print(f"Accuracy of the model Decision Tree Classifier and max_depth=7 is: {accuracy3*100}%")

    modelR1 = RandomForestClassifier(n_estimators=5,random_state=42)
    modelR1 = modelR1.fit(X_train,y_train)
    y_pred = modelR1.predict(X_test)
    accuracy4 = accuracy_score(y_test,y_pred)
    print(f"Accuracy of the model Random Forest Classifier and n_estimators=5 is: {accuracy4*100}%")

    modelR2 = RandomForestClassifier(n_estimators=7,random_state=42)
    modelR2 = modelR2.fit(X_train,y_train)
    y_pred = modelR2.predict(X_test)
    accuracy5 = accuracy_score(y_test,y_pred)
    print(f"Accuracy of the model Random Forest Classifier and n_estimators=7 is: {accuracy5*100}%")

    modelR3 = RandomForestClassifier(n_estimators=9,random_state=42)
    modelR3 = modelR3.fit(X_train,y_train)
    y_pred = modelR3.predict(X_test)
    accuracy6 = accuracy_score(y_test,y_pred)
    print(f"Accuracy of the model Random Forest Classifier and n_estimators=9 is: {accuracy6*100}%")

if __name__ == "__main__":
    main()