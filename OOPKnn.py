from math import sqrt
class KNeighboursClassifier:

    def __init__(self,k=3):
        if(isinstance(k,int)):
            if(k>0):
                self.k = k
                self.trainingPoints = None
            else:
                raise ValueError("k cannot be 0 or smaller than 0")
        else:
            raise ValueError("Invalid data type of k, k should be an integer")    

    @staticmethod
    def calcEuclideanDistance(X,Y):
        x1 = X["co-ordinates"][0]
        x2 = Y["co-ordinates"][0]
        y1 = X["co-ordinates"][1]
        y2 = Y["co-ordinates"][1]
        result = sqrt((((x2-x1)**2)+((y2-y1)**2)))
        return result

    @staticmethod
    def calc(existingPoints,newPoint):
        result = list()
        for data in existingPoints:
            res = KNeighboursClassifier.calcEuclideanDistance(data,newPoint)
            ans = {"point":data["point"],"label":data["label"],"distanceWithNewPoint":res}
            result.append(ans)
        sorted_result = sorted(result,key=lambda d: d["distanceWithNewPoint"])
        return sorted_result


    def vote(self,existingPoints,newPoint):
        newDict = dict()
        dataList = KNeighboursClassifier.calc(existingPoints,newPoint)
        nearest = dataList[:self.k]
        for n in nearest:
            label = n["label"]
            newDict[label] = newDict.get(label,0)+1
        maximum = max(newDict,key=lambda d : newDict.get(d))
        carry = newDict.get(maximum)
        sameVoteCount = list()
        for data in newDict:
            if(newDict[data]==carry):
                sameVoteCount.append(data)
        if(len(sameVoteCount)==1):
            return (sameVoteCount[0],nearest)
        else:
            for n in nearest:
                if(n["label"] in sameVoteCount):
                    return (n["label"],nearest)

    def fit(self,trainingPoints):
        if(self.k<=len(trainingPoints)):
            self.trainingPoints = trainingPoints
            return self
        else:
            raise ValueError("k is larger than the available training data")

    def predict(self,newPoints):
        if(self.trainingPoints is None):
            raise RuntimeError("First You Need To Fit The Model With Valid Data")
        if(isinstance(newPoints,dict)):
            return self.vote(self.trainingPoints,newPoints)
        elif(isinstance(newPoints,list)):
            result = list()
            for i in newPoints:
                ans = self.vote(self.trainingPoints,i)
                result.append(ans)
            return result
        else:
            raise ValueError("Invalid Data Given!")



        
          

            







