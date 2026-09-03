class UserDefinedSimpleLinearRegression:

    def __init__(self):
        self.slope = None
        self.intercept = None
    
    def fit(self,X : list, Y : list):
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same number of values")
        if len(X) == 0:
            raise ValueError("X and Y cannot be empty")
        meanX = sum(X)/len(X)
        meanY = sum(Y)/len(Y)
        summation = 0
        denominator = 0
        for i in range(len(X)):
            summation += (X[i]-meanX)*(Y[i]-meanY)
            denominator += (X[i]-meanX) ** 2
        if denominator == 0:
            raise ValueError("Cannot calculate slope when all X values are the same")
        self.slope = summation/denominator
        self.intercept = meanY - (meanX*self.slope)
        return self

    def predict(self,X:list):
        if self.slope is None or self.intercept is None:
            raise ValueError("Model has not been fitted yet")
        predictions = list()
        for x in X:
            predictions.append((self.slope*x)+self.intercept)
        return predictions

    @staticmethod
    def mse(Y_true:list,Y_pred:list):
        if(len(Y_true)!=len(Y_pred)):
            raise ValueError("Y_true and Y_pred are of different lengths!")
        if len(Y_true) == 0:
                    raise ValueError("Y cannot be empty")
        summation = 0
        for i in range(len(Y_true)):
            summation += (Y_true[i]-Y_pred[i])**2
        meanse = summation/len(Y_true)
        return meanse

    @staticmethod
    def meanAbsoluteError(Y_true:list,Y_pred:list):
        if(len(Y_true)!=len(Y_pred)):
            raise ValueError("Y_true and Y_pred are of different lengths!")
        if len(Y_true) == 0:
            raise ValueError("Y cannot be empty")
        summations = 0
        for i in range(len(Y_true)):
            summations += abs(Y_true[i]-Y_pred[i])
        mae = summations/len(Y_true)
        return mae

    @staticmethod
    def r2(Y_true:list,Y_pred:list):
        if(len(Y_true)!=len(Y_pred)):
            raise ValueError("Y_true and Y_pred are of different lengths!")
        if len(Y_true) == 0:
            raise ValueError("Y cannot be empty")
        ss_res = 0
        ss_tot = 0
        n = len(Y_true)
        meanY = sum(Y_true)/n
        for i in range(n):
            ss_res += (Y_true[i]-Y_pred[i])**2
            ss_tot += (Y_true[i]-meanY)**2
        r2Score = 1 - (ss_res/ss_tot)
        return r2Score
    
if __name__ == "__main__":
    X = [1,2,3,4,5]
    Y = [3,4,2,4,5]
    model = UserDefinedSimpleLinearRegression()
    model = model.fit(X,Y)
    Y_pred = model.predict(X)
    for i in range(len(Y_pred)):
        print(f"Predicted Value : {Y_pred[i]}")
        print(f"Actual Value : {Y[i]}")
    print("MAE of training data :",UserDefinedSimpleLinearRegression.meanAbsoluteError(Y,Y_pred))
    print("MSE of training data :",UserDefinedSimpleLinearRegression.mse(Y,Y_pred))
    print("R2 score of training data :",UserDefinedSimpleLinearRegression.r2(Y,Y_pred))




    
    