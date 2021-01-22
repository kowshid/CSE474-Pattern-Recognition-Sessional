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


class RewardAndPunish:
    def __init__(self):
        self.w = np.zeros(featureCount + 1)
        self.lr = 0.5

    def train(self):
        global dataset, datasetSize, classes, featureCount, classCount
        maxIteration = 10000
        itr = 0

        while True:
            misclassifiedData = []
            misclassifiedCount = 0
            change = False

            for i in range(datasetSize):
                x = np.array(dataset[i])
                givenClass = classes[i]

                product = np.dot(self.w, x)

                if givenClass == 1 and product < 0:
                    self.w = self.w + self.lr * x
                    misclassifiedData.append(x * -1.0)
                    misclassifiedCount += 1
                    change = True
                elif givenClass == 2 and product >= 0:
                    self.w = self.w - self.lr * x
                    misclassifiedData.append(x)
                    misclassifiedCount += 1
                    change = True

            if not change:
                print("training done at " + str(i + 1) + "th iteration\n")
                break

            # itr += 1
            # if itr >= maxIteration:
            #     print("Could not converge\n")
            #     break

    def test(self, file):
        global classes

        correctPrediction = 0
        wrongPredictions = []
        wrongPredictionIdx = []
        actual = []
        predicted = []
        wrongPredictionCount = 0
        out = open("outputRewardAndPunish.txt", "w")
        out.write("Testing Reward and Punishment Algorithm\n")

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

            product = np.dot(self.w, data)

            if product > 0:
                predictedClass = 1
            else:
                predictedClass = 2

            if givenClass == predictedClass:
                correctPrediction += 1
            else:
                # print(predictedClass, givenClass)
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

ReadDataset("trainLinearlySeparable.txt")
operation = RewardAndPunish()
operation.train()
operation.test("testLinearlySeparable.txt")