import matplotlib.pyplot as plt
import sys
import math
import random

sys.setrecursionlimit(99999)

dataset = []
clusterId = []
colorCount = 12
colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
          for i in range(colorCount)]


def getDistance(a, b):
    dist = 0.0
    dist += (a[0] - b[0]) ** 2
    dist += (a[1] - b[1]) ** 2
    dist = math.sqrt(dist)
    # print(dist)

    return dist


def readDataset(datasetFile):
    global dataset
    input = open(datasetFile, "r")
    lines = input.readlines()

    for line in lines:
        x, y = map(float, line.split())
        dataset.append([x, y])

    # print(dataset)
    # plt.figure(2)
    # for i in range(len(dataset)):
    #     # if clusterId[i] >= 0:
    #     # colour = colors[clusterId[i]]
    #     plt.scatter(dataset[i][0], dataset[i][1], color='#66ff99')
    # plt.show()


def estimateEPS(k):
    global eps
    fourthNeighbourDist = []
    for i in range(len(dataset)):
        distArr = []
        for j in range(len(dataset)):
            if i == j:
                continue
            distArr.append(getDistance(dataset[i], dataset[j]))

        distArr.sort()
        # print(len(distArr))
        fourthNeighbourDist.append(distArr[k - 1])

    fourthNeighbourDist.sort()
    # print(len(fourthNeighbourDist))

    plt.plot(fourthNeighbourDist, color='#66ff99', linewidth=3)
    plt.grid()
    plt.show()

    eps = input("Estimated eps from graph?\n")
    eps = float(eps)

    return eps


def DFS(src, id, eps):
    global clusterId
    clusterId[src] = id

    for i in range(len(dataset)):
        if clusterId[i] == -1 and getDistance(dataset[src], dataset[i]) < eps:
            DFS(i, clusterId[src], eps)


def DBSCAN(eps, k):
    global clusterId
    corePointIdx = []
    for i in range(len(dataset)):
        count = 0
        for j in range(len(dataset)):
            if i == j:
                continue
            dist = getDistance(dataset[i], dataset[j])
            if dist < eps:
                count += 1  # count of points around you

        if count >= k:
            corePointIdx.append(i)
            # corepoint index are being append
            # tried to append the points instead, but it requires searching later

    clusterId = [-1] * len(dataset)
    cid = -1
    # root = random.choice(dataset)

    for i in corePointIdx:
        if clusterId[i] == -1:
            cid += 1
            DFS(i, cid, eps)
        else:
            continue
    cid += 1

    print("Clusters Formed:", cid)

    plt.figure(2)
    for i in range(len(dataset)):
        if clusterId[i] >= 0:
            colour = colors[clusterId[i]]
            plt.scatter(dataset[i][0], dataset[i][1], color=colour)
    plt.show()

    return cid


def kMeansRandomPoint(clusterCount):
    xMax = - math.inf
    xMin = math.inf
    yMax = - math.inf
    yMin = math.inf

    for i in range(len(dataset)):
        if xMax < dataset[i][0]:
            xMax = dataset[i][0]
        if xMin > dataset[i][0]:
            xMin = dataset[i][0]
        if yMax < dataset[i][1]:
            yMax = dataset[i][1]
        if yMin > dataset[i][1]:
            yMin = dataset[i][1]

    # print(xMin, xMax, yMin, yMax)

    centroids = []
    for i in range(clusterCount):
        xRangeMin = xMin + (xMax - xMin) * i / clusterCount
        xRangeMax = xMin + (xMax - xMin) * (i + 1) / clusterCount
        yRangeMin = yMin + (yMax - yMin) * i / clusterCount
        yRangeMax = yMin + (yMax - yMin) * (i + 1) / clusterCount
        x = random.uniform(xRangeMax, xRangeMin)
        y = random.uniform(yRangeMax, yRangeMin)

        centroids.append([x, y])

    # print(centroids)
    a = 0

    while a < 500:
        distanceFromCentroid = [math.inf] * len(dataset)
        tempClusterId = [-1] * len(dataset)
        clusters = [[] for x in range(clusterCount)]
        newCentroids = []
        print(a, "th iteration")

        for i in range(len(dataset)):
            for j in range(clusterCount):
                dist = getDistance(centroids[j], dataset[i])
                if distanceFromCentroid[i] > dist:
                    distanceFromCentroid[i] = dist
                    tempClusterId[i] = j

        for i in range(len(dataset)):
            if tempClusterId[i] >= 0:
                clusters[tempClusterId[i]].append(dataset[i])

        for i in range(clusterCount):
            # print(clusters[i])
            avgX = 0.0
            avgY = 0.0

            for j in range(len(clusters[i])):
                avgX += clusters[i][j][0]
                avgY += clusters[i][j][1]

            avgX = avgX / len(clusters[i])
            avgY = avgY / len(clusters[i])

            newCentroids.append([avgX, avgY])

        flag = True
        for i in range(clusterCount):
            dist = getDistance(centroids[i], newCentroids[i])
            if dist != 0:
                flag = False

        if flag:
            break

        for i in range(clusterCount):
            centroids[i][0] = newCentroids[i][0]
            centroids[i][1] = newCentroids[i][1]

        a += 1

    plt.figure(3)
    for i in range(len(dataset)):
        if tempClusterId[i] >= 0:
            colour = colors[tempClusterId[i]]
            plt.scatter(dataset[i][0], dataset[i][1], color=colour)

    for centroid in centroids:
        plt.scatter(centroid[0], centroid[1], color='#000000', marker='s', linewidths=5)
    plt.show()


def kMeansRandomIdx(clusterCount):
    centroids = []

    for i in range(clusterCount):
        minIdx = i * (len(dataset)) / clusterCount
        maxIdx = (i + 1) * (len(dataset)) / clusterCount
        idx = random.randint(int(minIdx), int(maxIdx))
        # idx = random.randint(0, len(dataset) - 1)
        centroids.append(dataset[idx])

    a = 0

    while a < 500:
        distanceFromCentroid = [math.inf] * len(dataset)
        tempClusterId = [-1] * len(dataset)
        clusters = [[] for x in range(clusterCount)]
        newCentroids = []

        print(a, "th iteration")
        for i in range(len(dataset)):
            for j in range(clusterCount):
                dist = getDistance(centroids[j], dataset[i])
                if distanceFromCentroid[i] > dist:
                    distanceFromCentroid[i] = dist
                    tempClusterId[i] = j

        for i in range(len(dataset)):
            if tempClusterId[i] >= 0:
                clusters[tempClusterId[i]].append(dataset[i])

        # print(a)
        for i in range(clusterCount):
            # print(clusters[i])
            avgX = 0.0
            avgY = 0.0

            for j in range(len(clusters[i])):
                avgX += clusters[i][j][0]
                avgY += clusters[i][j][1]

            avgX = avgX / len(clusters[i])
            avgY = avgY / len(clusters[i])

            newCentroids.append([avgX, avgY])

        flag = True
        for i in range(clusterCount):
            dist = getDistance(centroids[i], newCentroids[i])
            if dist != 0:
                flag = False

        if flag:
            break

        for i in range(clusterCount):
            centroids[i][0] = newCentroids[i][0]
            centroids[i][1] = newCentroids[i][1]

        a += 1

    plt.figure(3)
    for i in range(len(dataset)):
        if tempClusterId[i] >= 0:
            colour = colors[tempClusterId[i]]
            plt.scatter(dataset[i][0], dataset[i][1], color=colour)

    for centroid in centroids:
        plt.scatter(centroid[0], centroid[1], color='#000000', marker='s', linewidths=4)
    plt.show()


def main():
    random.seed(102)
    k = 4
    readDataset("./data/blobs.txt")
    estimateEPS(k)
    cluster = DBSCAN(eps, k)
    kMeansRandomIdx(cluster)
    # kMeansRandomPoint(cluster)


if __name__ == "__main__":
    main()
