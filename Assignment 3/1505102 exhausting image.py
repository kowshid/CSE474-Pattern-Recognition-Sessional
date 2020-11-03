import numpy as np
import cv2
import math

def exhaustive(test, ref):
    testImg = test
    test = test.astype(int)
    ref = ref.astype(int)

    testsize = np.shape(test)
    I = testsize[0]
    J = testsize[1]

    refsize  = np.shape(ref)
    M = refsize[0]
    N = refsize[1]

    print(I, J, M, N)

    minDist = math.inf
    rowIdx = 0
    colIdx = 0

    for i in range(I - M + 1):
        for j in range(J - N + 1):
            subMatrix = test[i : i + M, j : j + N]
            diff = (subMatrix - ref)
            diff *= diff
            dist = np.sum(diff)

            if(dist < minDist):
                rowIdx = i
                colIdx = j
                minDist = dist

    print(minDist, rowIdx, colIdx)

    #need to draw a red rectangle around

    testRGB = cv2.cvtColor(testImg, cv2.COLOR_GRAY2RGB)
    #cv2.rectangle(testRGB, (rowIdx, colIdx), (rowIdx + N, colIdx + M), (0, 0, 255), 1)
    cv2.rectangle(testRGB, (colIdx, rowIdx), (colIdx + N, rowIdx + M), (0, 0, 255), 1)

    cv2.imshow("Exhaustive Search Output Image", testRGB)
    cv2.waitKey(0)

def main():
    ref = cv2.imread("reference.jpg", 0)
    test = cv2.imread("test.jpg", 0)

    exhaustive(test, ref)

if __name__ == "__main__":
    main()