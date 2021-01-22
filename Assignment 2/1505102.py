import numpy as np

featureCount = 0
classCount = 0
#layerCount = 4
classes = []
trainFeature = None
trainClass = None
testFeature = None
testClass = None

trainFileName = "trainNN.txt"
testFileName = "testNN.txt"

def sigmoid(n):
    return 1 / (1 + np.exp(-n))

def derivativeSigmoid(n):
    return sigmoid(n) * (1 - sigmoid(n))

def readDataset():
    global featureCount, classCount, classes, testFileName, trainFileName, trainFeature, trainClass, testFeature, testClass

    # read train dataset
    features = []
    trainFeature = []
    trainClass = []

    trainFile = open(trainFileName)
    lines = trainFile.readlines()
    dataCount = len(lines)

    data = lines[0].split()
    featureCount = len(data) - 1

    for i in range(dataCount):
        featureValues = lines[i].split()
        temp = featureValues[: featureCount]
        features.append(np.array(temp, dtype=float))
        currentClass = int(featureValues[featureCount])
        classes.append(currentClass)

        if classCount < currentClass:
            classCount = currentClass

    trainFeature = np.array(features).T
    trainClass = np.zeros((classCount, dataCount))

    for i in range(dataCount):
        trainClass[classes[i] - 1][i] = 1

    # read test dataset
    testFeature = []
    testClass = []
    features = []
    testFile = open(testFileName)
    lines = testFile.readlines()
    dataCount = len(lines)

    for i in range(dataCount):
        featureValues = lines[i].split()
        temp = featureValues[: featureCount]
        features.append(np.array(temp, dtype=float))
        testClass.append(featureValues[featureCount])

    testFeature = np.array(features).T

if __name__ == "__main__":
    readDataset()