#
# Using Naive Bayesian Classification for 
# A5 Radar Trace Classifier assignment.
#
# CS 131 - Artificial Intelligence
#
# Written by - Vivian Lau vlau02
# Last modified - 12/03/2023

import copy # to make changes to a list without modifying original
import numpy as np # for calculating mean and std dev
import math # for calculating natural log
import itertools # for making lists filled with 0s

# global variables
TRANS_PROB = 0.9
WINDOW_SIZE = 150

class RadarTraceClassifier:
    # Stores given likelihoods, training data, and testing data
    # Assumes the given data are 2d float lists
    def __init__(self, likelihoodsIn, trainDataIn, testDataIn):
        self.likelihood = likelihoodsIn
        self.trainData = trainDataIn
        self.testData = testDataIn
        
    # Returns list of classifications for each set of velocities
    # where one line in given test data is a set
    # 'a' represents plane and 'b' represents bird
    # order of classifications corresponds to line ordering in test data
    def run(self):
        self.makeNewLikelihoods() # make new likelihood function on std dev
        
        classifications = []
        
        # run model on testing data and classify each set of testing data
        count = 0
        for testSet in self.testData:
            count += 1
            print("Classifying Line",count)
            classifications.append(self.classifier(testSet))
            
        return classifications

    # Returns identification of bird or plane based on which has higher
    # probability given a set of stats consisting of the mean and std dev
    # of velocities in a window
    # 'a' represents plane and 'b' represents bird
    def classifier(self, testSet):
        stats = self.getStats(testSet)
        planeProb = 0.5 # starts with equal chances of being plane or bird
        birdProb = 0.5
        classification = 'a'
        first = True # different for the first velocity bc no priors
        
        # for each window, calculate probability of being bird or plane
        # based on previous probability and stats of current window
        for i in range(len(stats[0])):
            curMean = stats[0][i] # get mean of window from stats
            curStdDev = stats[1][i] # get std dev of window from stats
            
            # apply transition probability to previous probability
            newPlaneProb = (planeProb * TRANS_PROB) + (birdProb* (1-TRANS_PROB))
            newBirdProb = (birdProb * TRANS_PROB) + (planeProb * (1-TRANS_PROB)) 
            
            if first: # only for first velocity where there are no priors
                newPlaneProb = planeProb # do not apply transition probability
                newBirdProb = birdProb
                first = False
            
            # calculate new likelihoods based on current stats in window
            planeLikelihood, birdLikelihood = 0, 0
            if (round(curMean*2) <= len(self.likelihood[0]) and \
                round(curStdDev*2) <= len(self.likelihood[1])) :
                planeLikelihood = self.likelihood[1][round(curMean*2)] \
                                  * self.likelihood[3][round(curStdDev*2)]
                birdLikelihood = self.likelihood[0][round(curMean*2)] \
                                 * self.likelihood[2][round(curStdDev*2)]
                #print('\t\tPlane likelihood:', planeLikelihood
                #print('\t\tBird likelihood:', birdLikelihood)

            # calculate new probability using likelihood and previous prior
            newPlaneProb *= planeLikelihood
            newBirdProb *= birdLikelihood

            # normalizes new probabilities
            totalProb = newBirdProb + newPlaneProb
            if totalProb != 0: # avoid division by 0
                planeProb = newPlaneProb / totalProb
                birdProb = newBirdProb / totalProb
        
            # update classification
            print('\t\tChance being Bird:', round(birdProb*100))
            print('\t\tChance being Plane:', round(planeProb*100))
            if birdProb > planeProb: # compare probabilities and classify
                classification =  'b' # bird
            else:
                classificaiton =  'a' # plane
            print('\tCurrent Classification:', classification)
        
        return classification
            
        
    # Uses stored training data to compute new likelihood based on occurrences
    # of std dev of specified window size of velocities for birds and planes
    def makeNewLikelihoods(self):
        birdStdDevs = []
        planeStdDevs = []
        birdMeans = []
        planeMeans = []
        
        # Get all the std dev and means in a specified window for each bird and
        # plane training set
        # Assumes that there is 10 bird data sets followed by 10 plane data sets
        for i in range(0,10):
            birdStdDevs = birdStdDevs + self.getStats(self.trainData[i])[1]
            planeStdDevs = planeStdDevs + self.getStats(self.trainData[i+10])[1]
            birdMeans = birdMeans + self.getStats(self.trainData[i])[0]
            planeMeans = planeMeans + self.getStats(self.trainData[i+10])[0]
        
        # Bin the std devs found in the training data
        # find max std dev bin value
        maxBin = round(max(max(birdStdDevs),max(planeStdDevs))*2)
        # set up new likelihood lists of 1s of applicable size
        self.likelihood.append(list(itertools.repeat(1, maxBin+10)))
        self.likelihood.append(list(itertools.repeat(1, maxBin+10)))
        for stdDev in birdStdDevs: # increment bins of size 0.5 when occurred
            self.likelihood[2][round(stdDev*2)] += 1
        for stdDev in planeStdDevs: # increment bins of size 0.5 when occurred
            self.likelihood[3][round(stdDev*2)] += 1
        # normalize the occurrences by total number of occurences (probability)
        totalBird = len(birdStdDevs)+maxBin
        totalPlane = len(planeStdDevs)+maxBin
        self.likelihood[2] = [i/totalBird for i in self.likelihood[2]]
        self.likelihood[3] = [i/totalPlane for i in self.likelihood[3]]
        
        ''' Scrapped due to using the given likelihoods for the mean instead
        # Bin the means found in the training data
        # find max mean value
        maxBin = round(max(max(birdMeans),max(planeMeans))*2)
        # set up new likelihood lists of 1s of applicable size
        self.likelihood[0] = (list(itertools.repeat(1, maxBin+10)))
        self.likelihood[1] = (list(itertools.repeat(1, maxBin+10)))
        for mean in birdMeans: # increment bins of size 0.5 when occurred
            self.likelihood[0][round(mean*2)] += 1
        for mean in planeMeans: # increment bins of size 0.5 when occurred
            self.likelihood[1][round(mean*2)] += 1
        # normalize the occurrences by total number of occurences (probability)
        totalBird = len(birdMeans)+maxBin
        totalPlane = len(planeMeans)+maxBin
        self.likelihood[0] = [i/totalBird for i in self.likelihood[0]]
        self.likelihood[1] = [i/totalPlane for i in self.likelihood[1]]
        '''

    # Returns mean and std dev of given set of velocities over given window size
    def getStats(self, velocities):
        means = []
        stdDevs = []
        
        # set up calculating variance of the set of velocities
        vCopy = copy.deepcopy(velocities)
        vCopy = [v for v in vCopy if v >= 0] # delete all NaN
        
        for window in range(0, len(vCopy), WINDOW_SIZE):
            vInWindow = []
            
            endWindow = window + WINDOW_SIZE # set end index of window
            # if not enough values left for another window
            if min(window + (2*WINDOW_SIZE), len(vCopy)):
                endWindow = len(vCopy)
            
            for idx in range(window, endWindow): # get all velocities in window
                vInWindow.append(vCopy[idx])
            means.append(np.mean(vInWindow)) # add mean of velocities in window
            stdDevs.append(np.std(vInWindow)) # add std dev of velocity in window
            
        return (means, stdDevs)