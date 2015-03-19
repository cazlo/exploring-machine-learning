
__author__ = "Andrew Paettie"

'''
    The binary single layer perceptron which is used for classification of an unknown number of binary valued attributes
    Uses gradient descent to update weights.

    wi = weight at index i, sigma = 1 / (1 + e ^ -t)
     wi <= wi + (learningRate)(derivative Error wrt. wi)
        <= wi + (learningRare)(delta sigma)(attribute value)(sigma prime)
        <= wi + (learningRate)(actual class - sigma)(attribute value)(sigma)(1-sigma)

'''

import sys
import math


class GDClassifier(object):

    def __init__(self, learningRate, numIterations):
        self.learningRate = learningRate
        self.iterations = numIterations

        self.attributes = []  # the attribute names
        self.attributeWeights = []
        self.classifiers = []  # an array of the possible classifiers
        self.attributeValues = [[]]  # [attributeIndex] = array of possible attribute values
                                     # [attributeIndex][attributeValueIndex] = attribute value
        self.trainingAttributeData = [[]]  # [instanceIndex][attributeIndex] = attribute value for training instance
        self.trainingAttributeClasses = []  # [instanceIndex] = the class of the training instance
        self.testAttributeData = [[]]  # same format as training data
        self.testAttributeClasses = []  # same format as training data

    def readFile(self, filename, isTraining, limit=None):
        f = open(filename, 'r')
        lineNum = 0
        if limit is None:
            limit = sys.maxint
        for line in f:
            if (lineNum > limit) and isTraining:
                break
            splitOnSpaces = line.split()
            if splitOnSpaces.__len__() == 0:
                continue
            elif splitOnSpaces.__len__() == 1:
                continue
            elif lineNum == 0 and isTraining:
                # add attribute definitions
                for index, attribute in enumerate(splitOnSpaces):
                    if index != splitOnSpaces.__len__() - 1: # dont add class to attributes (class is last attribute)
                        self.attributes.append(attribute)
                        self.attributeWeights.append(0.0)

                lineNum += 1
            elif lineNum == 0 and not isTraining:
                # make sure that attributes match up to training set attributes
                for index, atttribute in enumerate(self.attributes):
                    if self.attributes.__len__() > index:
                        if splitOnSpaces[index] != self.attributes[index]:
                            print('WARNING: test attributes do not match training attributes')
                lineNum += 1
            else:
                attrCount = 0
                if isTraining:
                    classesList = self.trainingAttributeClasses
                    dataList = self.trainingAttributeData
                else:
                    classesList = self.testAttributeClasses
                    dataList = self.testAttributeData
                dataList.append([])
                for attrIndex, attribute in enumerate(splitOnSpaces):
                    if attrCount == splitOnSpaces.__len__() - 1:
                        classesList.append(int(attribute))
                        if self.classifiers.count(int(attribute)) == 0:
                            self.classifiers.append(int(attribute))
                    else:
                        dataList[lineNum - 1].append(int(attribute))
                        if self.attributeValues.__len__() <= attrIndex:
                            self.attributeValues.append([])
                        if self.attributeValues[attrIndex].count(int(attribute)) == 0:
                            self.attributeValues[attrIndex].append(int(attribute))
                    attrCount += 1
                lineNum += 1
        dataList.remove([])
        return self

    def learn(self):
        iteration = 0
        while iteration < self.iterations:
            instanceIndex = iteration % self.trainingAttributeData.__len__()
            instanceSigma = self.getSigmoid(instanceIndex, True)
            for weightIndex, weight in enumerate(self.attributeWeights):
                #  modify weights based on gradient descent
                # wi = wi + (learningRate)(dE/dwi)
                #    = wi + (learningRate)(actual class - sigma)(attribute value)(sigma)(1-sigma)
                actualClass = self.trainingAttributeClasses[instanceIndex]
                attrValue = self.trainingAttributeData[instanceIndex][weightIndex]
                weightChange = self.learningRate*(actualClass - instanceSigma)*attrValue*(1-instanceSigma)*instanceSigma
                self.attributeWeights[weightIndex] = weight + weightChange
            msg = "After iteration "+str(iteration+1)+": "
            for attrIndex, attrValue in enumerate(self.attributes):
                msg += "w(" + attrValue + ") = {0:0.4f}, ".format(self.attributeWeights[attrIndex])
            msg += "output: {0:0.4f}"
            print(msg.format(self.getSigmoid(instanceIndex, True)))
            iteration += 1

    def sigmaFunction(self, t):
        return 1 / (1 + math.e ** (0-t))

    def getSigmoid(self, instanceIndex, isTraining):
        instanceList = self.trainingAttributeData if isTraining else self.testAttributeData
        t = 0.0
        for attrIndex, weight in enumerate(self.attributeWeights):
            t += (weight * instanceList[instanceIndex][attrIndex])

        s = self.sigmaFunction(t)
        return s

    def classify(self, isTraining):
        classList = self.trainingAttributeClasses if isTraining else self.testAttributeClasses
        numCorrect = 0.0
        numTotal = 0.0
        for instanceIndex, instanceClass in enumerate(classList):
            classification = self.getClassFromPerceptron(instanceIndex, isTraining)
            if instanceClass == classification:
                numCorrect += 1
            numTotal += 1

        message = "Accuracy on " + ("training " if isTraining else "test ") + "set "
        message += "({0} instances):  {1:.1f}%"
        print message.format(numTotal, numCorrect / float(numTotal) * 100)

    def getClassFromPerceptron(self, instanceIndex, isTraining):
        out = self.getSigmoid(instanceIndex, isTraining)
        if out >= 0.5:
            return 1
        else:
            return 0