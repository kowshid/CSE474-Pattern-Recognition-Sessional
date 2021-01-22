import matplotlib.pyplot as plt
import sys
import math
import random

sys.setrecursionlimit(99999)

dataset = []
clusterId = []
# highest number of color needed for plots
colorCount = 12
# generating color code randomly
colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
          for i in range(colorCount)]

# euclidian distance for 2 given points
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

# estimatings eps for DBSCAN, k nearest neighbors distance id plotted
def estimateEPS(k):
    global eps
    fourthNeighbourDist = []
    for i in range(len(dataset)):
        # distArr will include the distances from a given point to all other points
        distArr = []
        for j in range(len(dataset)):
            if i == j:
                continue
            distArr.append(getDistance(dataset[i], dataset[j]))

        # sorted the distances and storing 4th neighbor distance in fourthNeighbourDist
        distArr.sort()
        # print(len(distArr))
        fourthNeighbourDist.append(distArr[k - 1])

    fourthNeighbourDist.sort()
    # print(len(fourthNeighbourDist))

    plt.figure(0)
    plt.plot(fourthNeighbourDist, color='#66ff99', linewidth=3)
    plt.ylabel("Distance of k-th neighbor")
    plt.xlabel("points id")
    plt.grid()
    plt.show()

    # estimating eps from graph
    eps = input("Estimated eps from graph?\n")
    eps = float(eps)

    return eps

# DFS is needed for separating core and boundary points in clusters
def DFS(src, id, eps):
    global clusterId
    clusterId[src] = id

    for i in range(len(dataset)):
        if clusterId[i] == -1 and getDistance(dataset[src], dataset[i]) < eps:
            DFS(i, clusterId[src], eps)


def DBSCAN(eps, k):
    global clusterId
    # only storing the index of core points from dataset
    corePointIdx = []
    for i in range(len(dataset)):
        count = 0
        for j in range(len(dataset)):
            if i == j:
                continue
            dist = getDistance(dataset[i], dataset[j])
            if dist < eps:
                count += 1
                # count of points around a a given points within eps

        # corepoint index are being append
        # tried to append the points instead, but it requires searching later
        if count >= k:
            corePointIdx.append(i)

    # every point will have a cluster ID, -1 for noise points
    clusterId = [-1] * len(dataset)
    # total cluster count
    cid = -1
    # root = random.choice(dataset)

    # running DFS on corepoints
    # it will include core and boundary points and discard noise
    for i in corePointIdx:
        if clusterId[i] == -1:
            cid += 1
            DFS(i, cid, eps)
        else:
            continue
    cid += 1

    print("Clusters Formed:", cid)

    plt.figure(1)
    for i in range(len(dataset)):
        if clusterId[i] >= 0:
            colour = colors[clusterId[i]]
            plt.scatter(dataset[i][0], dataset[i][1], color=colour)
    plt.show()

    return cid


def kMeansRandomPoint(clusterCount):
    centroids = []
    # generating points on uniform intervals for centroids
    # generated points may not be in dataset
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

    for i in range(clusterCount):
        xRangeMin = xMin + (xMax - xMin) * i / clusterCount
        xRangeMax = xMin + (xMax - xMin) * (i + 1) / clusterCount
        yRangeMin = yMin + (yMax - yMin) * i / clusterCount
        yRangeMax = yMin + (yMax - yMin) * (i + 1) / clusterCount
        x = random.uniform(xRangeMax, xRangeMin)
        y = random.uniform(yRangeMax, yRangeMin)

        centroids.append([x, y])

    # print(centroids)

    # max iteration check
    a = 0

    while a < 500:
        # calculating distance of each points from centroids, minimum is stored
        distanceFromCentroid = [math.inf] * len(dataset)
        # distance from centroid will determine the cluster id
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

        # appending to corresponding clusters to plot
        for i in range(len(dataset)):
            if tempClusterId[i] >= 0:
                clusters[tempClusterId[i]].append(dataset[i])

        # determining new centroids to update
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

        # if all the centroids in a iteration remains unchanged, break loop
        flag = True
        for i in range(clusterCount):
            dist = getDistance(centroids[i], newCentroids[i])
            if dist != 0:
                flag = False

        if flag:
            break

        # updating the centroids
        for i in range(clusterCount):
            centroids[i][0] = newCentroids[i][0]
            centroids[i][1] = newCentroids[i][1]

        a += 1

    plt.figure(2)
    for i in range(len(dataset)):
        if tempClusterId[i] >= 0:
            colour = colors[tempClusterId[i]]
            plt.scatter(dataset[i][0], dataset[i][1], color=colour)

    for centroid in centroids:
        plt.scatter(centroid[0], centroid[1], color='#000000', marker='s', linewidths=5)
    plt.show()


def kMeansRandomIdx(clusterCount):
    # generating points on uniform intervals for centroids
    # generated points may not be in dataset
    centroids = []

    # picking random index from uniform interval in dataset
    # and appending that to centroids
    for i in range(clusterCount):
        minIdx = i * (len(dataset)) / clusterCount
        maxIdx = (i + 1) * (len(dataset)) / clusterCount
        idx = random.randint(int(minIdx), int(maxIdx))
        # idx = random.randint(0, len(dataset) - 1)
        centroids.append(dataset[idx])

    # max iteration check
    a = 0

    while a < 500:
        # calculating distance of each points from centroids, minimum is stored
        distanceFromCentroid = [math.inf] * len(dataset)
        # distance from centroid will determine the cluster id
        tempClusterId = [-1] * len(dataset)
        clusters = [[] for x in range(clusterCount)]
        newCentroids = []

        # print(a, "th iteration")
        for i in range(len(dataset)):
            for j in range(clusterCount):
                dist = getDistance(centroids[j], dataset[i])
                if distanceFromCentroid[i] > dist:
                    distanceFromCentroid[i] = dist
                    tempClusterId[i] = j

        # appending to corresponding clusters to plot
        for i in range(len(dataset)):
            if tempClusterId[i] >= 0:
                clusters[tempClusterId[i]].append(dataset[i])

        # print(a)
        # determining new centroids to update
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

        # if all the centroids in a iteration remains unchanged, break loop
        flag = True
        for i in range(clusterCount):
            dist = getDistance(centroids[i], newCentroids[i])
            if dist != 0:
                flag = False

        if flag:
            break

        # updating the centroids
        for i in range(clusterCount):
            centroids[i][0] = newCentroids[i][0]
            centroids[i][1] = newCentroids[i][1]

        a += 1

    print("Iteration needed = ", a)
    plt.figure(2)
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
    readDataset("./data/moons.txt")
    estimateEPS(k)
    cluster = DBSCAN(eps, k)
    kMeansRandomIdx(cluster)
    # kMeansRandomPoint(cluster)
    

if __name__ == "__main__":
    main()


"""
EPS
bisecting: 0.65
blob: 0.6
moon: 0.08
"""