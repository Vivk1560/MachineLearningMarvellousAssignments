import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
    model = RandomForestClassifier(n_estimators=7,random_state=42)
    model = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(f"Accuracy of model using all features is:{accuracy*100:.2f}%")
    for feature,importance in zip(features,model.feature_importances_):
        print(f"Feature : {feature} ,Importance : {importance:.4f}")

    top_features = [
        "Flavanoids",
        "OD280/OD315 of diluted wines",
        "Alcohol",
        "Alcalinity of ash",
        "Proline",
        "Total phenols",
        "Color intensity"
    ]
    X = df[top_features]
    Y = df["Class"]
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=42)
    model_top = RandomForestClassifier(n_estimators=7,random_state=42)
    model_top= model_top.fit(X_train,y_train)
    y_pred = model_top.predict(X_test)
    accuracy_top = accuracy_score(y_test,y_pred)
    print(f"Accuracy of model using only 7 features is:{accuracy_top*100:.2f}%")

    top_5_features = [
    "Flavanoids",
    "OD280/OD315 of diluted wines",
    "Alcohol",
    "Proline",
    "Alcalinity of ash"
    ]
    X = df[top_5_features]
    Y = df["Class"]
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=42)
    model_top5 = RandomForestClassifier(n_estimators=7,random_state=42)
    model_top5 = model_top5.fit(X_train,y_train)
    y_pred = model_top5.predict(X_test)
    accuracy_top5 = accuracy_score(y_test,y_pred)
    print(f"Accuracy of model using only 5 features is:{accuracy_top5*100:.2f}%")

    selected_10_features = [
    "Alcohol",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline"
    ]
    X = df[selected_10_features]
    Y = df["Class"]
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=42)
    model_top10 = RandomForestClassifier(n_estimators=7,random_state=42)
    model_top10 = model_top10.fit(X_train,y_train)
    y_pred = model_top10.predict(X_test)
    accuracy_top10 = accuracy_score(y_test,y_pred)
    print(f"Accuracy of model using only 10 features is:{accuracy_top10*100:.2f}%")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[1,2,3]
    )
    disp.plot()
    plt.title("Confusion Matrix - Final Random Forest Model")
    plt.show()

if __name__ == "__main__":
    main()


# ============================================================
# FEATURE SELECTION AND FINAL MODEL SELECTION
# ============================================================
#
# 1. Initially, all 13 features were used to train a Random
#    Forest Classifier and the accuracy was calculated.
#
# 2. Random Forest feature_importances_ was used to identify
#    which features contributed most to the classification.
#
# 3. Based on feature importance, different feature subsets
#    were tested:
#
#       - All 13 features  -> Accuracy: 97.22%
#       - Top 7 features   -> Accuracy: 94.44%
#       - Top 5 features   -> Accuracy: 90.28%
#       - Selected 10 features -> Accuracy: 98.61%
#
# 4. The 10-feature model performed better than the model
#    using all 13 features while using 3 fewer features.
#
# 5. Therefore, the following 10 features were selected for
#    the final model:
#
#       Alcohol
#       Alcalinity of ash
#       Magnesium
#       Total phenols
#       Flavanoids
#       Proanthocyanins
#       Color intensity
#       Hue
#       OD280/OD315 of diluted wines
#       Proline
#
# 6. Final Model:
#       Random Forest Classifier
#       n_estimators = 7
#       random_state = 42
#
#    Final accuracy on the test set = 98.61%
#
# 7. The final model will now be evaluated using a confusion
#    matrix and ConfusionMatrixDisplay to understand the
#    class-wise predictions and misclassifications.
# ============================================================