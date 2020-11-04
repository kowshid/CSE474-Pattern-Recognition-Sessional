import numpy as np
import cv2
import math
import time

M = 0
N = 0
I = 0
J = 0
p = 15
prevRow = 0
prevCol = 0
inputInGrayScale = []
outputFrames = []
frameCount = 0
search = 0

def readVideo():
    global inputInGrayScale
    cap = cv2.VideoCapture('input.mov')
    ret = True

    while ret:
        ret, frame = cap.read()
        if ret:
            frameGrayScale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            inputInGrayScale.append(frameGrayScale)

    cap.release()

def writeVideo():
    global outputFrames, prevRow, prevCol
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    outputVideo = cv2.VideoWriter("2D Logarithmic.mov", fourcc, 60, (J, I))

    for frame in outputFrames:
        outputVideo.write(frame)

    outputVideo.release()
    outputFrames = []
    prevRow = 0
    prevCol = 0

# work on per frame
def exhaustiveSearch(frame, ref):
    global M, N, I, J, p, outputFrames, prevRow, prevCol, search

    frameImg = frame
    frame = frame.astype(int)
    ref = ref.astype(int)

    minDist = math.inf
    rowIdx = 0
    colIdx = 0

    # denotes first search, need to search whole area (all sub-matrix)
    if (prevRow == 0) and (prevCol == 0):
        rowStart = 0
        rowEnd = I - M + 1
        colStart = 0
        colEnd = J - N + 1
    # not the first search, need to search neighbouring area
    else:
        rowStart = prevRow - p
        rowEnd = prevRow + p
        colStart = prevCol - p
        colEnd = prevCol + p

    for i in range(rowStart, rowEnd):
        for j in range(colStart, colEnd):
            if (i < 0) or (i > I - M) or (j < 0) or (j > J - N):
                None
            else:
                search += 1
                subMatrix = frame[i: i + M, j: j + N]
                diff = (subMatrix - ref)
                diff *= diff
                dist = np.sum(diff)

                if dist < minDist:
                    rowIdx = i
                    colIdx = j
                    minDist = dist

    #print(minDist, rowIdx, colIdx)

    #need to draw a red rectangle around
    frameRGB = cv2.cvtColor(frameImg, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(frameRGB, (colIdx, rowIdx), (colIdx + N, rowIdx + M), (0, 0, 255), 1)
    # print(frameRGB)
    outputFrames.append(frameRGB)

    prevRow = rowIdx
    prevCol = colIdx

    # return testRGB
    # cv2.imshow("Exhaustive Search Output Image", testRGB)
    # cv2.waitKey(0)

def log2D(frame, ref):
    global M, N, I, J, p, outputFrames, prevRow, prevCol, search

    frameImg = frame
    frame = frame.astype(int)
    ref = ref.astype(int)

    minDist = math.inf
    rowIdx = 0
    colIdx = 0
    # denotes first search, need to search whole area (all sub-matrix)
    # exhaustive search
    if (prevRow == 0) and (prevCol == 0):
        for i in range(I - M + 1):
            for j in range(J - N + 1):
                subMatrix = frame[i: i + M, j: j + N]
                diff = (subMatrix - ref)
                diff *= diff
                dist = np.sum(diff)

                if (dist < minDist):
                    rowIdx = i
                    colIdx = j
                    minDist = dist
        print("jvj")
        prevRow = rowIdx
        prevCol = colIdx
    # not the first search, need to search neighbouring area
    else:
        pTemp = p
        while True:
            print("hbdacn")
            k = math.ceil(math.log2(pTemp))
            d = 2 ** (k - 1)
            print(k, d)

            if d < 1:
                break

            X = [prevCol + d, prevCol, prevCol - d]
            Y = [prevRow + d, prevRow, prevRow - d]

            for i in Y:
                for j in X:
                    if (i < 0) or (i > I - M) or (j < 0) or (j > J - N):
                        None
                    else:
                        search += 1
                        subMatrix = frame[i: i + M, j: j + N]
                        diff = (subMatrix - ref)
                        diff *= diff
                        dist = np.sum(diff)

                        if dist < minDist:
                            rowIdx = i
                            colIdx = j
                            minDist = dist

            prevRow = rowIdx
            prevCol = colIdx
            pTemp = pTemp/2

    # need to draw a red rectangle around
    frameRGB = cv2.cvtColor(frameImg, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(frameRGB, (colIdx, rowIdx), (colIdx + N, rowIdx + M), (0, 0, 255), 1)
    # print(frameRGB)
    outputFrames.append(frameRGB)

def main():
    global M, N, I, J, p, frameCount, inputInGrayScale, outputFrames, prevRow, prevCol, search

    # loading reference image in grayscale using 0
    ref = cv2.imread("reference.jpg", 0)
    # test = cv2.imread("test.jpg", 0)
    readVideo()

    refsize = np.shape(ref)
    M = refsize[0]
    N = refsize[1]

    testsize = np.shape(inputInGrayScale[0])
    I = testsize[0]
    J = testsize[1]

    frameCount = len(inputInGrayScale)

    print("shape of input ", (I, J))
    print("shape of reference ", (M, N))
    print("total frames: ", frameCount)

    # search = 0
    # start = time.time()
    # for i in range(frameCount):
    #     exhaustiveSearch(inputInGrayScale[i], ref)
    # end = time.time()
    # writeVideo()
    # avgSearchExhaustive = search/frameCount
    # timeExhaustive = end - start
    # print("Avg Search per frame for Exhaustive Search", avgSearchExhaustive)
    # print("Time taken by Exhaustive Search ", timeExhaustive, "sec")

    search = 0
    start = time.time()
    for i in range(frameCount):
        log2D(inputInGrayScale[i], ref)
    end = time.time()
    writeVideo()
    avgSearch2DLog = search / frameCount
    time2DLog = end - start
    print("Avg Search per frame for Exhaustive Search", avgSearch2DLog)
    print("Time taken by Exhaustive Search ", time2DLog, "sec")

if __name__ == "__main__":
    main()