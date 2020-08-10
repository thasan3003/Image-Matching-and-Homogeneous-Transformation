# ==================================================
# Md. Tahmid Hasan
# Web: tahmidhasan3003
# ==================================================

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

inf = 999999999
sigma = .75               # Used for Gaussian
pi = 3.1416               # PI in mathematics
harrisK = 0.04            # Value of k in harris corner detection
nmsKSize = 5              # Non maximum supression kernel size
maxNoOfKeyPoints = 50     # Maximum number of key points
resizedDim = (400, 600)   # Image resize dimension


# Derivation and other values of an image
def derivation(image):
    image = np.float32(image)
    kernelSize = 3   # Sobel kernel size: 3x3
    offset = int(kernelSize/2)
    height = image.shape[0]
    width = image.shape[1]
    magnitude = np.zeros((height,width), dtype=np.float32)
    orientation = np.zeros((height,width), dtype=np.float32)

    sobel_x = np.array([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]])
    sobel_y = np.array([[-1,-2,-1],
                        [0,0,0],
                        [1,2,1]])
    
    # Calculate X and Y derivative
    dx = cv2.filter2D(image,-1,sobel_x)
    dy = cv2.filter2D(image,-1,sobel_y)
    
    # Calculate magnitude and orientation
    for row in range(offset, height-offset):
        for col in range(offset, width-offset):
            sum1 = dx[row][col]
            sum2 = dy[row][col]
            
            magnitude[row][col] = math.sqrt((sum1**2)+(sum2**2))
            orientation[row][col] = (math.atan2(sum2,sum1)*180/pi)+180   # 0 to 360 degree
    
    print('Derivation: Done!')
    dxx = dx**2
    dyy = dy**2
    dxy = dx*dy
        
    return dxx, dyy, dxy, magnitude, orientation


# Generating Gaussian kernel
def generateGaussianKernel():
    kernelSize = int(2*3*sigma)+1
    offset = int(kernelSize/2)
    gaussianKernel = np.zeros((kernelSize,kernelSize), dtype = np.float32)
    
    doubleSigmaSquare = 2*sigma*sigma
    sum1 = 0
    
    for x in range(-offset,offset+1):
        for y in range(-offset,offset+1):
            tempNum = x*x+y*y
            gaussianKernel[x+offset][y+offset] = (math.exp(-tempNum/doubleSigmaSquare))/(pi*doubleSigmaSquare)
            sum1 += gaussianKernel[x+offset][y+offset]
    gaussianKernel = (1.0/sum1)*gaussianKernel
    
    return gaussianKernel


# Applying Gaussian filter & finding corner response applying threshold
def gaussianFilter(dxx, dyy, dxy, gaussianKernel):
    kernelSize = gaussianKernel.shape[0]
    offset = int(kernelSize/2)
    maxValue = -inf
    minValue = inf
    height = dxx.shape[0]
    width = dxx.shape[1]
    cornerResponse = np.zeros((height,width), dtype = np.float32)
    
    # Apply Gaussian filter
    Sxx = cv2.filter2D(dxx,-1,gaussianKernel)
    Syy = cv2.filter2D(dyy,-1,gaussianKernel)
    Sxy = cv2.filter2D(dxy,-1,gaussianKernel)
    
    # Calculate corner response
    for row in range(offset, height-offset):
        for col in range(offset, width-offset):
            sum1 = Sxx[row][col]
            sum2 = Syy[row][col]
            sum3 = Sxy[row][col]

            det = (sum1 * sum2) - (sum3**2)   # Determinant
            trace = sum1 + sum2               # Trace

            r = det - harrisK*(trace**2)      # Corner response
            cornerResponse[row][col] = r
            
            if maxValue < r:
                maxValue = r
            if minValue > r:
                minValue = r
            
    print('Apply Gaussian & calculate corner response: Done!')
    
    return Sxx, Syy, Sxy, cornerResponse, minValue, maxValue


# Apply threshold on corner response value
def applyThreshold(cornerResponse, cornerThres):
    height = cornerResponse.shape[0]
    width = cornerResponse.shape[1]
    thresholdResponse = np.zeros((height,width), dtype = np.float32)
    count = 0
    
    for row in range(height):
        for col in range(width):
            temp = cornerResponse[row][col]
            if (temp > cornerThres):
                thresholdResponse[row][col] = temp
                count += 1
    
    print('Total point (after applying threshold): ', count)
    return thresholdResponse


# Adaptive Non Maximum Supression for picking max-threshold in a region
def maxPicker(thresholdResponse, cornerThres):
    height = thresholdResponse.shape[0]
    width = thresholdResponse.shape[1]
    kernelSize = nmsKSize
    
    while (1):   # Continuous process
        NMSThresholdResponse = np.zeros((height,width), dtype=float)
        offset = int(kernelSize/2)
        kernelRange = np.arange(-offset,offset+1,1)
        increment = (2*offset)+1
        count = 0
        finalKeyPointList = []
        
        for row in range(offset, height-offset, increment):
            for col in range(offset, width-offset, increment):
                maxValue = thresholdResponse[row][col]
                maxValRow = row
                maxValCol = col
                for i in kernelRange:
                    for j in kernelRange:
                        temp = thresholdResponse[row+i][col+j]
                        if (maxValue < temp):
                            maxValue = temp
                            maxValRow = row+i
                            maxValCol = col+j

                if (maxValue > cornerThres):
                    NMSThresholdResponse[maxValRow][maxValCol] = 255 # maxValue
                    finalKeyPointList.append([maxValue,maxValRow,maxValCol])   # Value, row and column
                    count += 1
                    
        if(count <= maxNoOfKeyPoints):   # Fixed number of key points
            break
        else:
            kernelSize += 4
            
    #print('Final kernel size of Adaptive NMS: ', kernelSize)
    print('Total point (after applying non-maximum supression): ', count)
    
    return NMSThresholdResponse, finalKeyPointList


# Marking the key points
def markPoints(inImage, finalKeyPointList):
    outImage = inImage.copy()
    radious = 3
    thickness = 2
    color = (255,0,0)   # Blue in BGR
    
    for item in finalKeyPointList:
        row = item[1]
        col = item[2]
        cv2.circle(outImage, (col, row), radious, color, thickness)   # Draw circle 
    
    return outImage


# Image generating from a matrix
def generateImage(inArray):
    maxValue = -inf
    minValue = inf
    height = inArray.shape[0]
    width = inArray.shape[1]
    outArray = np.zeros((height,width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            temp = inArray[i][j]
            if maxValue < temp:
                maxValue = temp
            if minValue > temp:
                minValue = temp
                
    maxValue -= minValue
    
    for i in range(height):
        for j in range(width):
            temp = inArray[i][j] - minValue
            outArray[i][j] = int((255.0/maxValue)*temp)   # Scale all value: 0 to 255
    
    return outArray


# Harris corner detection technique
def cornerDetect(mainImage, imageNo):
    print('Working on image ', imageNo, '...')
    cv2.imwrite('OutputImages/1_'+str(imageNo)+'_1_mainImage.jpg', mainImage)
    grayImage = cv2.cvtColor(mainImage, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('OutputImages/1_'+str(imageNo)+'_2_grayImage.jpg', grayImage)
    
    # Derivation
    dxx, dyy, dxy, magnitude, orientation = derivation(grayImage)
    
    # Apply Gaussian & Calculate Corner Response
    gaussianKernel = generateGaussianKernel()
    Sxx, Syy, Sxy, cornerResponse, minValue, maxValue = gaussianFilter(dxx, dyy, dxy, gaussianKernel)
    
    # Set Threshold Value
    #cornerThres = minValue + ((maxValue - minValue)*.5)
    cornerThres = 100000000
    print ('Threshold Value: ', cornerThres)
    
    # Apply Threshold
    thresholdResponse = applyThreshold(cornerResponse, cornerThres)
    outImage = generateImage(thresholdResponse)
    cv2.imwrite('OutputImages/1_'+str(imageNo)+'_3_thresholdResponse.jpg',outImage)
    
    # Non Maximum Supression
    NMSThresholdResponse, finalKeyPointList = maxPicker(thresholdResponse, cornerThres)
    #NMSThresholdResponse = maxPicker2(thresholdResponse, pointsList)
    outImage = generateImage(NMSThresholdResponse)
    cv2.imwrite('OutputImages/1_'+str(imageNo)+'_4_NMSResult.jpg',outImage)

    # Marking Corners
    markedPoints = markPoints(mainImage, finalKeyPointList)
    cv2.imwrite('OutputImages/1_'+str(imageNo)+'_5_keyPoints.jpg',markedPoints)
    
    return magnitude, orientation, finalKeyPointList , markedPoints


# SIFT like descriptor
def descriptor(mag, ori, finalKeyPointList):
    descList = []
    
    for points in finalKeyPointList:   # Each point in finalKeyPointList
        # Left-top corner of the 16x16 patch of key point
        startRow = points[1]-7
        startCol =  points[2]-7
        # Container of values
        hist = [0 for x in range(128)]
        
        # Outer quardant
        for row in range(4):
            for col in range(4):
                # Inner quardant
                for inRow in range(4):
                    for inCol in range(4):
                        # Current position
                        posR = 4*row+inRow+startRow   # Row
                        posC = 4*col+inCol+startCol   # Col
                        posI = 32*row + 8*col         # Index of hist

                        tempOri = ori[posR][posC]
                        tempMag = mag[posR][posC]
                        
                        # Distribute the magnitude value between degrees according to distance.
                        # Nearest degree has the largest portion
                        if (tempOri < 45):
                            tempMag = (45-tempOri)/45*tempMag
                            hist[posI+0] += tempMag
                            hist[posI+1] += (mag[posR][posC]-tempMag)
                        elif (tempOri >= 45 and tempOri < 90):
                            tempMag = (90-tempOri)/45*tempMag
                            hist[posI+1] += tempMag
                            hist[posI+2] += (mag[posR][posC]-tempMag)
                        elif (tempOri >= 90 and tempOri < 135):
                            tempMag = (135-tempOri)/45*tempMag
                            hist[posI+2] += tempMag
                            hist[posI+3] += (mag[posR][posC]-tempMag)
                        elif (tempOri >= 135 and tempOri < 180):
                            tempMag = (180-tempOri)/45*tempMag
                            hist[posI+3] += tempMag
                            hist[posI+4] += (mag[posR][posC]-tempMag)
                        elif (tempOri >= 180 and tempOri < 225):
                            tempMag = (225-tempOri)/45*tempMag
                            hist[posI+4] += tempMag
                            hist[posI+5] += (mag[posR][posC]-tempMag)
                        elif (tempOri >= 225 and tempOri < 270):
                            tempMag = (270-tempOri)/45*tempMag
                            hist[posI+5] += tempMag
                            hist[posI+6] += (mag[posR][posC]-tempMag)
                        elif (tempOri >= 270 and tempOri < 315):
                            tempMag = (315-tempOri)/45*tempMag
                            hist[posI+6] += tempMag
                            hist[posI+7] += (mag[posR][posC]-tempMag)
                        elif (tempOri >= 315):
                            tempMag = (360-tempOri)/45*tempMag
                            hist[posI+7] += tempMag
                            hist[posI+0] += (mag[posR][posC]-tempMag)
        
        # Normalization (Euclidean)
        sqSum = 0
        for x in hist:
            sqSum += (x**2)   # Squared sum
        sqSum = math.sqrt(sqSum)
        for i in range(128):
            hist[i] = hist[i]/sqSum
        
        # Add to descriptor list. First part is position and second part is descriptor values: [[row, col], hist]
        descList.append([[startRow+7, startCol+7], hist])
    
    return descList


# Plot 16 bar chart from every 8 values of a descriptor
def plotGraph(desc):
    #print(desc)
    #print('Pos: ',desc[0])
    
    xCo = [i for i in range(8)]   # x-coordinates
    
    for i in range(16):
        # y-coordinates
        yCo = [j for j in desc[1][8*i:8*i+8]]   # Each iteration index: 0...7, 8...15, 16...23, ...., 120...127
        # labels for bars 
        tick_label = [j for j in range(0,360,45)]    # 0, 45, 90, ..., 315

        # plotting a bar chart 
        #plt.figure()
        plt.bar(xCo, yCo, tick_label = tick_label)
        #plt.bar(xCo, yCo, tick_label = tick_label, width = 0.8, color = ['red', 'green'])
        plt.ylim(0, 1)
        plt.xlabel('Orientations')
        plt.ylabel('Magnitudes')
        #plt.title(print('SIFT like descriptor of a keypoint: Quardant ',i+1))
        plt.savefig('OutputImages/2_SIFT_Quardant_'+str(i+1)+'.png', bbox_inches='tight')
        plt.close()


# Plot whole descriptor in a bar chart
def plotGraph2(desc):
    #print(desc)
    #print('Pos: ',desc[0])
    
    xCo = [i for i in range(1,129)]   # x-coordinates
    yCo = desc[1]                     # y-coordinates
    #plt.figure()
    plt.bar(xCo, yCo)
    plt.ylim(0, 1)
    plt.xlabel('Orientations')
    plt.ylabel('Magnitudes')
    #plt.title(print('SIFT like descriptor of a keypoint'))
    plt.savefig('OutputImages/2_SIFT_Keypoint_1.png', bbox_inches='tight')
    plt.close()


# Matching using Euclidean distance
def matchEuclid(descList, descList2, markedPoints, markedPoints2, selectPointNo):
    color = (0, 255, 255)   # Yellow color in BGR 
    thickness = 1           # Line thickness
    count = 0
    offset = markedPoints.shape[1]
    choosenPairList = []
    firstSecondBest = []
    
    # Combine two matrix
    horizontalCombined = cv2.hconcat([markedPoints, markedPoints2])
    
    for item1 in descList:
        minDist = inf
        minDist2 = inf
        startPoint = (item1[0][1], item1[0][0])   # (x, y) = (col, row) & stored as [row, col]
        endPoint = (0, 0)
        endPoint2 = (0, 0)
        
        for item2 in descList2:
            # Euclidean distance
            dist = 0
            for x in range(128):
                dist += ((item1[1][x]-item2[1][x])**2)
            dist = math.sqrt(dist)
            
            if(minDist > dist):   # Best match
                # 2nd best
                minDist2 = minDist
                endPoint2 = endPoint
                # 1st best
                minDist = dist
                endPoint = (offset+item2[0][1], item2[0][0])   # (x, y) = (col, row) & stored as [row, col]
            elif(minDist2 > dist):   # 2nd best
                minDist2 = dist
                endPoint2 = (offset+item2[0][1], item2[0][0])  # (x, y) = (col, row) & stored as [row, col]
        
        # Store [start point, 1st best, 2nd best] = [[x,y],[v1,x1,y1],[v2,x2,y2]]
        firstSecondBest.append([[startPoint[0], startPoint[1]], [minDist, endPoint[0]-offset, endPoint[1]], [minDist2, endPoint2[0]-offset, endPoint2[1]]])
        
        # Thresholding on distance
        thres = 0.6
        if(minDist < thres):
            count += 1
            if count in selectPointNo:   # Manually choosen
                # Store as [x1, y1, x2, y2]
                choosenPairList.append([startPoint[0], startPoint[1], endPoint[0]-offset, endPoint[1]])
                
            cv2.line(horizontalCombined, startPoint, endPoint, color, thickness)
            
    print ('Matching using Euclidean distance: Done!')
    print ('Total matched (Euclidean distance under ', thres,'): ', count)
    
    return horizontalCombined, choosenPairList, firstSecondBest


# Matching using Cosine Similarity
def matchCosSimilarity(descList, descList2, markedPoints, markedPoints2):
    color = (0, 255, 255)   # Yellow
    thickness = 1           # Line thickness
    count = 0
    offset = markedPoints.shape[1]
    
    # Combine two matrix
    horizontalCombined = cv2.hconcat([markedPoints, markedPoints2])
    
    for item1 in descList:
        maxSim = -inf
        startPoint = (item1[0][1], item1[0][0])   # (x, y) = (col, row)
        endPoint = (0, 0)
        
        for item2 in descList2:
            # Cosine Similarity
            a = item1[1]
            b = item2[1]
            simScore = (np.dot(a, b)) / (math.sqrt(np.dot(a, a))*math.sqrt(np.dot(b, b)))   # using dot product
            
            if(maxSim < simScore):   # Best match
                maxSim = simScore
                endPoint = (offset+item2[0][1], item2[0][0])
        
        # Thresholding on similarity score
        thres = 0.75
        if(maxSim > thres):
            count += 1
            cv2.line(horizontalCombined, startPoint, endPoint, color, thickness)
            
    print ('Matching using Cosine similarity: Done!')
    print ('Total matched (Cosine similarity above ', thres,'): ', count)
    
    return horizontalCombined


# Matching using Nearest Neighbor Distance Ratio (NNDR)
def matchNNDR(descList, descList2, markedPoints, markedPoints2, firstSecondBest):
    color = (0, 255, 255)   # Yellow
    thickness = 1           # Line thickness
    count = 0
    offset = markedPoints.shape[1]
    
    # Combine two matrix
    horizontalCombined = cv2.hconcat([markedPoints, markedPoints2])
    
    for item in firstSecondBest:
        # NNDR
        # item = [[x,y],[v1,x1,y1],[v2,x2,y2]]
        ratio = item[1][0]/item[2][0]
        startPoint = (item[0][0], item[0][1])        # (x, y)
        endPoint = (item[1][1] + offset, item[1][2])   # (x1 + offset, y1)
        
        # Thresholding on ratio
        thres = .85
        if(ratio < thres):
            count += 1
            color = (0, 255, 255)   # Yellow
        else:
            color = (0, 0, 255)     # Red
        
        cv2.line(horizontalCombined, startPoint, endPoint, color, thickness)
            
    print ('Matching using Nearest Neighbor Distance Ratio: Done!')
    print('Total matched (NNDR under ', thres,'): ', count)
    
    return horizontalCombined


# Mark choosen pair
def markChoosenPair(markedPoints, markedPoints2, choosenPairList):
    color = (0, 255, 255)   # Yellow
    thickness = 2           # Line thickness
    offset = markedPoints.shape[1]
    
    # Combine two matrix
    horizontalCombined = cv2.hconcat([markedPoints, markedPoints2])
    
    for item in choosenPairList:
        # item = [x1, y1, x2, y2]
        startPoint = (item[0], item[1])
        endPoint = (item[2] + offset, item[3])
        cv2.line(horizontalCombined, startPoint, endPoint, color, thickness)
        
    return horizontalCombined


# Generating matrix A in AH=0
def generateAMatrix(choosenPairList):
    listLength = len(choosenPairList)
    matA = np.zeros((listLength*2, 9), dtype=np.int32)
    
    print('Choosen points:')
    for i in range(listLength):
        x = choosenPairList[i][0]
        y = choosenPairList[i][1]
        xp = choosenPairList[i][2]
        yp = choosenPairList[i][3]
        print('(x, y): (', x, ', ', y, ')\t(xp, yp): (', xp, ', ', yp, ')')
        
        matA[i*2] = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
        matA[i*2+1] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
        
    return matA


# Calculating matrix H in AH=0 using SVD
def findHMatrix(matA):
    u, s, vT = np.linalg.svd(matA)
    matH = vT[-1]                         # last row of vT = last column of v
    matH = np.reshape(matH,(3,3))
    matH = matH/matH[2][2]
    
    return matH


# Apply H to transform an image
def applyHMatrix(inImage, matH):
    # Finding transformed position of 4 corners of input image
    xc = []
    yc = []
    xyc = np.matmul(matH,np.array([1,1,1]))
    xc.append(int(xyc[0]/xyc[2]+0.5))
    yc.append(int(xyc[1]/xyc[2]+0.5))
    xyc = np.matmul(matH,np.array([inImage.shape[1],1,1]))
    xc.append(int(xyc[0]/xyc[2]+0.5))
    yc.append(int(xyc[1]/xyc[2]+0.5))
    xyc = np.matmul(matH,np.array([1,inImage.shape[0],1]))
    xc.append(int(xyc[0]/xyc[2]+0.5))
    yc.append(int(xyc[1]/xyc[2]+0.5))
    xyc = np.matmul(matH,np.array([inImage.shape[1],inImage.shape[0],1]))
    xc.append(int(xyc[0]/xyc[2]+0.5))
    yc.append(int(xyc[1]/xyc[2]+0.5))
    
    # Define transformed matrix size & offset
    xc = np.array(xc)   # Convert list into array
    yc = np.array(yc)
    xOffset = min(xc)
    tMatWidth = max(xc)
    yOffset = min(yc)
    tMatHeight = max(yc)
    tMatWidth = tMatWidth-xOffset
    tMatHeight = tMatHeight-yOffset
    transformedMatrix = np.zeros((tMatHeight, tMatWidth, 3), dtype = np.uint8)   # (tMatHeight, tMatWidth) = (row, col)
    
    # Backward warping
    matH = np.linalg.inv(matH)   # Inverse of matrix H
    
    for row in range(transformedMatrix.shape[0]):
        for col in range(transformedMatrix.shape[1]):
            xyc = np.matmul(matH,np.array([col+xOffset,row+yOffset,1]))
            tempX = xyc[0]/xyc[2]
            tempY = xyc[1]/xyc[2]
            x = int(tempX)
            y = int(tempY)
            
            if (x >= 0 and y >= 0 and (x+1) < inImage.shape[1] and (y+1) < inImage.shape[0]):
                # Bilinear interpolation
                a = tempX - x
                b = tempY - y
                tempV = (1-a)*(1-b)*inImage[y][x]
                tempV += (a*(1-b)*inImage[y][x+1])
                tempV += ((1-a)*b*inImage[y+1][x])
                tempV += (a*b*inImage[y+1][x+1])
                transformedMatrix[row][col] = (tempV+0.5)
    
    return transformedMatrix


# Finding difference between two images of same size
def findDifference(inImage, transformedImage):
    height = inImage.shape[0]
    width = inImage.shape[1]
    differenceMatrix = np.zeros((height,width), dtype = np.uint8)
    inImage = cv2.cvtColor(inImage, cv2.COLOR_BGR2GRAY)
    transformedImage = cv2.cvtColor(transformedImage, cv2.COLOR_BGR2GRAY)
    
    for row in range(height):
        for col in range(width):
            differenceMatrix[row][col] = abs(int(inImage[row][col])-int(transformedImage[row][col]))
            
    return differenceMatrix


# Main part
def main():
    # Main part 1: Keypoint (Corner)
    print ('Problem 1: Harris Corner Detection\n')
    # Image 1
    mainImage1 = cv2.imread('Images/NotreDame1.jpg')
    mainImage1 = cv2.resize(mainImage1, resizedDim, interpolation = cv2.INTER_AREA)
    magnitude1, orientation1, finalKeyPointList1, markedPoints1 = cornerDetect(mainImage1, 1)
    print ()
    
    # Image 2
    mainImage2 = cv2.imread('Images/NotreDame2.jpg')
    mainImage2 = cv2.resize(mainImage2, resizedDim, interpolation = cv2.INTER_AREA)
    magnitude2, orientation2, finalKeyPointList2, markedPoints2 = cornerDetect(mainImage2, 2)

    print ('\nProblem 1 status: Done!\n')

    # Main part 2: SIFT Like Descriptor
    print ('\nProblem 2: SIFT like descriptor')
    # Image 1
    descList1 = descriptor(magnitude1, orientation1, finalKeyPointList1)
    #plotGraph(descList1[0])    # Descriptor into 16 parts
    plotGraph2(descList1[0])   # Whole descriptor

    # Image 2
    descList2 = descriptor(magnitude2, orientation2, finalKeyPointList2)
    
    print ('\nProblem 2 status: Done!\n')


    # Main part 3: Matching
    print ('\nProblem 3: Matching\n')
    # Euclidean distance
    selectPointNo = [2,3,12,21]
    matchedImage, choosenPairList, firstSecondBest = matchEuclid(descList1, descList2, markedPoints1, markedPoints2, selectPointNo)
    cv2.imwrite('OutputImages/3_1_matching_Euclid.jpg', matchedImage)
    print ()
    # Cosine similarity
    matchedImage2 = matchCosSimilarity(descList1, descList2, markedPoints1, markedPoints2)
    cv2.imwrite('OutputImages/3_2_matching_CosSim.jpg', matchedImage2)
    print ()
    # Nearest Neighbor Distance Ratio (NNDR)
    matchedImage3 = matchNNDR(descList1, descList2, markedPoints1, markedPoints2, firstSecondBest)
    cv2.imwrite('OutputImages/3_3_matching_NNDR.jpg', matchedImage3)
    
    print ('\nProblem 3 status: Done!\n')


    # Main part 4: Generate A
    print ('\nProblem 4: Generate Matrix A\n')
    # Marking pairs
    markedPairImage = markChoosenPair(markedPoints1, markedPoints2, choosenPairList)
    cv2.imwrite('OutputImages/4_markedChoosenPair.jpg', markedPairImage)

    # Generate matrix A
    matA = generateAMatrix(choosenPairList)
    print('Matrix A:\n', matA)
    
    print ('\nProblem 4 status: Done!\n')


    # Main part 5: Calculate H
    print ('\nProblem 5: Calculate Matrix H\n')
    matH = findHMatrix(matA)
    print('Matrix H:\n',matH)
    
    print ('\nProblem 5 status: Done!\n')


    # Main part 6: Apply H
    print ('\nProblem 6: Apply H to Transform\n')
    # Applying H
    transformedImage = applyHMatrix(mainImage1, matH)
    cv2.imwrite('OutputImages/6_1_transformedImage.jpg',transformedImage)

    # Concatenation
    transformedImage = cv2.resize(transformedImage, resizedDim, interpolation = cv2.INTER_AREA)
    threeConcate = cv2.hconcat([mainImage1, mainImage2, transformedImage])
    cv2.imwrite('OutputImages/6_2_threeConcate.jpg',threeConcate)

    # Differenciation
    # Difference between 2nd image and transformed image
    differenceImage = findDifference(mainImage2, transformedImage)
    cv2.imwrite('OutputImages/6_3_differenceImage2TransformedImage.jpg',differenceImage)

    print ('\nProblem 6 status: Done!\n')

    print('\nAll are done! Check results...')
    

# Code starts from here
if __name__ == '__main__':
    main()
