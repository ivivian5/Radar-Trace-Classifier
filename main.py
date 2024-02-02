#
# main function for A5 Radar Trace Classifier assignment.
# Calls radar trace classifier to classify 10 unidentified objects
# using naive bayesian classification.
#
# CS 131 - Artificial Intelligence
#
# Written by - Vivian Lau vlau02
# Last modified - 12/03/2023

from Radar_Trace_Classifier import RadarTraceClassifier

# begin format
print ("----------------------------------------------------------------------")
print ("Welcome to Vivian's Radar Trace Classifier used for classifying \n" + \
       "unidentified flying objects as birds or airplanes!")
print ("----------------------------------------------------------------------")

param = input("Default or custom parameters? ")

likelihoodFile, trainFile, testFile = None, None, None
likelihoods, trainData, testData = [], [], []
filesFound = True

if param.lower() == "default": # get default files
    try:
        likelihoodFile = open('likelihood.txt', 'r')
        trainFile = open('training.txt', 'r')
        testFile = open('testing.txt', 'r')
    except FileNotFoundError:
        filesFound = False
        print("Error: Files not found, cannot execute default problem.")
else: # requests for custom files
    print("\nPlease enter the file names of custom problem.")
    valid = False
    while not valid:
        likeFileName = input("Enter file name for likelihoods of bird and"+\
                             " airplane velocities: ")
        valid = True
        try:
            likelihoodFile = open(likeFileName, 'r')
        except FileNotFoundError:
            valid = False
            print("Invalid file name, please enter again.")


    valid = False
    while not valid:
        trainFileName = input("Enter file name for training data: ")
        valid = True
        try:
            trainFile = open(trainFileName, 'r')
        except FileNotFoundError:
            valid = False
            print("Invalid file name, please enter again.")

    valid = False
    while not valid:
        testFileName = input("Enter file name for testing data: ")
        valid = True
        try:
            testFile = open(testFileName, 'r')
        except FileNotFoundError:
            valid = False
            print("Invalid file name, please enter again.")

lines = None
cont = True

if filesFound: # read files if found
    likelyLines = likelihoodFile.readlines()
    if len(likelyLines) == 2:
        for line in likelyLines: # likelihoods
            line = line.strip()
            line = [x for x in line.split()] # remove end carriage
            line = [-1 if x == "NaN" else x for x in line] # fixing data format
            line = [float(x) for x in line]
            likelihoods.append(line)
    else:
        cont = False
        
    trainLines = trainFile.readlines()
    if cont and len(trainLines) == 20:
        numTrain = 0
        for line in trainLines: # training data
            numTrain += 1
            line = line.strip()
            line = [x for x in line.split()]
            line = [-1 if x == "NaN" else x for x in line]
            line = [float(x) for x in line]
            trainData.append(line)
    else:
        cont = False
        
    testLines = testFile.readlines()
    if cont and len(testLines) == 10:
        numTest = 0
        for line in testLines: # testing data
            numTest += 1
            line = line.strip()
            line = [x for x in line.split()]
            line = [-1 if x == "NaN" else x for x in line]
            line = [float(x) for x in line]
            testData.append(line)
    else:
        cont = False

if not cont:
    print("Invalid data format in file, please run program again.")
else:
    rtc = RadarTraceClassifier(likelihoods, trainData, testData)
    classifications = rtc.run() # run solver
    
    # print results
    print('\n FINAL LIST')
    for i in range(len(testLines)):
        print("Line", (i+1), ": ", classifications[i])
    print('\n')
    
    countBad = 0
    answerKey = ['b','b','b','a','a','b','a','a','a','b']
    for i in range(10):
        if answerKey[i] != classifications[i]:
            countBad += 1
    print("Accuracy:",100-(countBad/10)*100,'%\n')

# end format
print ("----------------------------------------------------------------------")
print ("Thanks for using Vivian's Radar Trace Classifier.")
print ("----------------------------------------------------------------------")
    