from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cityblock
import matplotlib.pyplot as plot

from numpy import mean


# Problem2 By Claudia Trafton
# Aid to help solve problem 2 of Homework 1 for CIS700, Security in Machine Learning.
# Please do not use without citation. Please write it on your own if you are taking this class.

# Classifiers with no printing
def manhattanDistanceClassificationNoPrint(point):
    manhattanDist_c1 = cityblock(point, meanPointc1)
    manhattanDist_c2 = cityblock(point, meanPointc2)
    if manhattanDist_c1 < manhattanDist_c2:
        return 1
    else:
        return 2


# Wrapper function that wraps whichever distance measure I try
def cosineDistanceClassificationNoPrint(point):
    cosineDist_c1 = cosine(point, meanPointc1)
    cosineDist_c2 = cosine(point, meanPointc2)
    if cosineDist_c1 < cosineDist_c2:
        return 1
    else:
        return 2


# Wrapper function that wraps manhattan
def euclideanDistanceClassificationNoPrint(point):
    euclideanDist_c1 = euclidean(point, meanPointc1)
    euclideanDist_c2 = euclidean(point, meanPointc2)
    if euclideanDist_c1 < euclideanDist_c2:
        return 1
    else:
        return 2


# reverse engineer data points, and verify if classification method gave a correct result
# try and classify existing data points and see if they were correctly classified.
# I'll use the method that got the most accurate results
def getAccuracyRatingsForClassifications(data):
    manhattanCorrect = 0
    cosineCorrect = 0
    euclidCorrect = 0
    DATA_SIZE = 1000.000  # did this to avoid truncation of decimal places
    print("Data size: " + str(DATA_SIZE))
    count1 = 0
    count2 = 0
    for vector in data:
        classifier = vector[3]
        point = [float(vector[1]), float(vector[2])]

        manhattan = manhattanDistanceClassificationNoPrint(point)
        if manhattan == classifier:
            manhattanCorrect += 1

        cos = cosineDistanceClassificationNoPrint(point)
        if cos == classifier:
            cosineCorrect += 1

        euclid = euclideanDistanceClassificationNoPrint(point)
        if euclid == classifier:
            euclidCorrect += 1

    manhattanCorrect = float(manhattanCorrect) / DATA_SIZE
    cosineCorrect = float(cosineCorrect) / DATA_SIZE
    euclidCorrect = float(euclidCorrect) / DATA_SIZE

    print("percentages correct: ")
    print("Manhattan: " + str.format('{0:.6f}', manhattanCorrect))
    print("Cosine: " + str.format('{0:.6f}', cosineCorrect))
    print("Euclid: " + str.format('{0:.6f}', euclidCorrect))


# helper functions
# Classify a given point into either c1 or c2 depending on distance
def classifyForAllMethods(x, y):
    point = [x, y]
    print()
    print("------TRYING MANHATTAN-------")
    print()
    manhattanDistanceClassification(point)
    print()
    print("------TRYING COSINE-------")
    print()
    cosineDistanceClassification(point)
    print()
    print("------TRYING EUCLIDEAN-------")
    print()
    euclideanDistanceClassification(point)


# Wrapper function that wraps manhattan
def manhattanDistanceClassification(point):
    manhattanDist_c1 = cityblock(point, meanPointc1)
    print("Manhattan distance for c1: ")
    print(manhattanDist_c1)

    manhattanDist_c2 = cityblock(point, meanPointc2)
    print("Manhattan distance for c2: ")
    print(manhattanDist_c2)
    if manhattanDist_c1 < manhattanDist_c2:
        print(point)
        print(str(point) + " belongs to c1")
        return manhattanDist_c1
    else:
        print(point)
        print(str(point) + " belongs to c2")
        return manhattanDist_c2


# Wrapper function that wraps whichever distance measure I try
def cosineDistanceClassification(point):
    cosineDist_c1 = cosine(point, meanPointc1)
    print("cosine distance for c1: ")
    print(cosineDist_c1)

    cosineDist_c2 = cosine(point, meanPointc2)
    print("cosine distance for c2: ")
    print(cosineDist_c2)
    if cosineDist_c1 < cosineDist_c2:
        print(point)
        print(str(point) + " belongs to c1")
        return cosineDist_c1
    else:
        print(point)
        print(str(point) + " belongs to c2")
        return cosineDist_c2


# Wrapper function that wraps manhattan
def euclideanDistanceClassification(point):
    euclideanDist_c1 = euclidean(point, meanPointc1)
    print("Euclidean distance for c1: ")
    print(euclideanDist_c1)

    euclideanDist_c2 = euclidean(point, meanPointc2)
    print("Euclidean distance for c2: ")
    print(euclideanDist_c2)
    if euclideanDist_c1 < euclideanDist_c2:
        print(point)
        print(str(point) + " belongs to c1")
        return euclideanDist_c1
    else:
        print(point)
        print(str(point) + " belongs to c2")
        return euclideanDist_c2


# Open the data file
filename = "q2data.txt"
file = open(filename, "r")
data = {}

# I'm being lazy, I just looked at the key of the last entry
DATA_SIZE = 1000

c1_x = []
c1_y = []

c2_x = []
c2_y = []
data = []
for line in file:
    vect = line.split(",")
    if vect[3] == '"c1"\n':
        c1_x.append(float(vect[1]))
        c1_y.append(float(vect[2]))
        vect[3] = 1
    elif vect[3] == '"c2"\n':
        c2_x.append(float(vect[1]))
        c2_y.append(float(vect[2]))
        vect[3] = 2
    data.append(vect)

# remove header
data.pop(0)

# calculate the mean point for each category

# for c1

# remove the header
c1_x.pop(0)
c2_x.pop(0)
c1_xMean = mean(c1_x)
c1_yMean = mean(c1_y)
meanPointc1 = [c1_xMean, c1_yMean]
print("c1 mean is: ")
print(meanPointc1)
print()

# for c2
c2_xMean = mean(c2_x)
c2_yMean = mean(c2_y)
meanPointc2 = [c2_xMean, c2_yMean]
print("c2 mean is: ")
print(meanPointc2)
print()

getAccuracyRatingsForClassifications(data)

print("----CLASSIFYING [5.0, -2.0]: --------")
print()
classifyForAllMethods(5.0, -2.0)
print()
print("-------------------------------------")
print("----CLASSIFYING [0.0, -4.0]: --------")
print()
classifyForAllMethods(0.0, -4.0)
print()
print("-------------------------------------")