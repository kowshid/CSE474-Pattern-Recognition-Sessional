import numpy as np

featureCount = 0
classCount = 0
datasetSize = 0

dataset = []
classes = []

def ReadDataset(file):
    f = open(file, 'r')
    global dataset, classes, datasetSize, featureCount, classCount
    featureCount, classCount, datasetSize = f.readline().split()
    featureCount = int(featureCount)
    classCount = int(classCount)
    datasetSize = int(datasetSize)
    # print("ds  ",datasetSize)

    lines = f.readlines()
    # print(len(lines), datasetSize, lines[0])
    for i in range(datasetSize):
        # print(i+1)
        data = lines[i].split()
        x = data[: featureCount]
        x.append(1.0)
        dataset.append(np.array(x, dtype=float))
        classes.append(int(data[featureCount]))

    # print(classes[0])
    # dataset = dataset.astype('float64')
    # classes = classes.astype('int32')


class Pocket:
    def __init__(self):
        self.w = np.zeros(featureCount + 1)
        self.wP = np.zeros(featureCount + 1)
        self.lr = 0.5
        self.highestAccuracy = 0

    def train(self):
        global dataset, datasetSize, classes, featureCount, classCount
        maxIteration = 1000
        itr = 0

        while True:
            misclassifiedData = []
            misclassifiedCount = 0

            for i in range(datasetSize):
                x = np.array(dataset[i])
                givenClass = classes[i]

                product = np.dot(self.w, x)

                if givenClass == 1 and product < 0:
                    misclassifiedData.append(x * -1.0)
                    misclassifiedCount += 1
                elif givenClass == 2 and product >= 0:
                    misclassifiedData.append(x)
                    misclassifiedCount += 1
            if self.highestAccuracy < (datasetSize - misclassifiedCount):
                self.highestAccuracy = (datasetSize - misclassifiedCount)
                self.wP = self.w

            if misclassifiedCount == 0:
                print("training done at " + str(i + 1) + "th iteration\n")
                break

            else:
                total = sum(misclassifiedData)
                self.w = self.w - self.lr * total

            itr += 1
            if itr >= maxIteration:
                #print("Could not converge\n")
                break

    def test(self, file):
        global classes

        correctPrediction = 0
        wrongPredictions = []
        wrongPredictionIdx = []
        actual = []
        predicted = []
        wrongPredictionCount = 0
        out = open("outputPocket.txt", "w")
        out.write("Testing Basic Perceptron Algorithm\n")

        f = open(file, "r")
        lines = f.readlines()
        testData = len(lines)

        for i in range(testData):
            data = lines[i].split()
            givenClass = int(data[featureCount])
            for j in range(featureCount):
                data[j] = float(data[j])

            data[featureCount] = 1.0
            data = np.array(data)
            # line = lines[i].split()
            # data = line[: featureCount]
            # givenClass = int(line[featureCount])
            # data.append(1.0)
            # #data.append(np.array(data, dtype=float))

            product = np.dot(self.wP, data)

            if product > 0:
                predictedClass = 1
            else:
                predictedClass = 2

            if givenClass == predictedClass:
                correctPrediction += 1
            else:
                #print(predictedClass, givenClass)
                print(i)
                wrongPredictions.append(data)
                wrongPredictionIdx.append(i)
                actual.append(givenClass)
                predicted.append(predictedClass)
                wrongPredictionCount += 1

        accuracy = (correctPrediction / testData) * 100

        out.write("Wrong predictions: " + str(wrongPredictionCount) + "\n")
        out.write("sample no.\tfeature values\tactual\tpredict\n")
        for i in range(wrongPredictionCount):
            out.write(str(wrongPredictionIdx[i]) + " ")
            out.write(str(wrongPredictions[i]) + " ")
            out.write(str(actual[i]) + " ")
            out.write(str(predicted[i]))
            out.write("\n")

        out.write("Correctly classified: " + str(correctPrediction) + "\n")
        out.write("Accuracy: " + str(accuracy))
        print(str(accuracy))

ReadDataset("trainLinearlyNonSeparable.txt")
p = Pocket()
p.train()
p.test("testLinearlyNonSeparable.txt")