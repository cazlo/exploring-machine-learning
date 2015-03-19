__author__ = "Andrew Paettie"

'''
    An implementation of a Naive Bayes classifier.  This will work for binary classification, but as an additional
    challenge I generalized it such that it will work for any number of classes and any number of attributes.
    To facilitate this, when looking at probabilities of attributes and classes, the program really only looks
    at the indices of attributes and classes in their respective arrays.  In this way, there can be a nearly unlimited
    number of classes and attribute values, and the classifier will still learn and use the appropriate probabilities.
    So that probabilities only need to be computed once, the NaiveBayes classifier features a container class for
    probabilities.  This container class is basically just an array of probabilities indexed on the same attribute
    indices arrays in the NaiveBayes object.
'''

import sys

class NaiveBayes(object):

    def __init__(self):
        self.attributes = []  # the attribute names
        self.classifiers = []  # an array of the possible classifiers
        self.attributeValues = [[]]  # [attributeIndex] = array of possible attribute values
                                     # [attributeIndex][attributeValueIndex] = attribute value
        self.trainingAttributeData = [[]]  # [instanceIndex][attributeIndex] = attribute value for training instance
        self.trainingAttributeClasses = []  # [instanceIndex] = the class of the training instance
        self.testAttributeData = [[]]  # same format as training data
        self.testAttributeClasses = []  # same format as training data
        self.probabilities = []  # a list of probability container objects

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
                        classesList.append(attribute)
                        if self.classifiers.count(attribute) == 0:
                            self.classifiers.append(attribute)
                    else:
                        dataList[lineNum - 1].append(attribute)
                        if self.attributeValues.__len__() <= attrIndex:
                            self.attributeValues.append([])
                        if self.attributeValues[attrIndex].count(attribute) == 0:
                            self.attributeValues[attrIndex].append(attribute)
                    attrCount += 1
                lineNum += 1
        if isTraining:
            self.trainingAttributeData.remove([])
        else:
            self.testAttributeData.remove([])
        return self

    # learns probabilities of data
    def learn(self):
        # TODO:
        for classIndex, c in enumerate(self.classifiers):
            pc = self.probabilityContainer(classIndex, self)
            for attrIndex, a in enumerate(self.attributes):
                attrValueIndicies = []  # so that we don't throw index out of bounds
                for attrValueIndex, av in enumerate(self.attributeValues[attrIndex]):
                    attrValueIndicies.append(self.computeConditionalProb(classIndex, attrIndex, attrValueIndex, True))
                pc.attributeProb.insert(attrIndex, attrValueIndicies)
            pc.attributeProb.remove([])  # remove the trailing empty list
            self.probabilities.append(pc)

    # computes probability of a class
    def computeProb(self, classifierIndex, isTraining):
        total = self.trainingAttributeData.__len__() if isTraining else self.testAttributeData.__len__()
        classTotal = 0.0
        for instanceClass in (self.trainingAttributeClasses if isTraining else self.testAttributeClasses):
            if instanceClass == self.classifiers[classifierIndex]:
                classTotal += 1
        if total == 0:
            return 0.0
        else:
            return float(classTotal) / total

    # computes probability of an attribute having a specific value for a class
    # set invertClass to true if we are looking for prob for !class
    def computeConditionalProb(self, classifierIndex, attributeIndex, attributeValueIndex, isTraining, invertClass = False):
        attrTotal = 0.0
        classTotal = 0.0
        instanceClassList = self.trainingAttributeClasses if isTraining else self.testAttributeClasses
        for instanceIndex, instanceClass in enumerate(instanceClassList):
            matchedClass = False
            if instanceClass == self.classifiers[classifierIndex] and not invertClass:
                matchedClass = True
            elif instanceClass != self.classifiers[classifierIndex] and invertClass:
                matchedClass = True
            if matchedClass:
                classTotal += 1
                if isTraining:
                    attrValue = self.trainingAttributeData[instanceIndex][attributeIndex]
                else:
                    attrValue = self.testAttributeData[instanceIndex][attributeIndex]

                if attrValue == self.attributeValues[attributeIndex][attributeValueIndex]:
                    attrTotal += 1
        if classTotal == 0.0:
            return 0.0
        else:
            return float(attrTotal) / classTotal

    # gives the index of an attribute value or -1 if it is not found
    def getAttributeValueIndexFromValue(self, attrValuePreferred, attrIndex):
        for attrValueIndex, attrValue in enumerate(self.attributeValues[attrIndex]):
            if attrValue == attrValuePreferred:
                return attrValueIndex
        return -1

    # classifies instances based on the probablilites learned previously
    def classify(self, isTraining):
        # TODO:
        instanceList = self.trainingAttributeData if isTraining else self.testAttributeData
        classList = self.trainingAttributeClasses if isTraining else self.testAttributeClasses
        numCorrect = 0.0
        numTotal = 0.0
        for instanceIndex, instance in enumerate(instanceList):
            instanceClass = classList[instanceIndex]
            # find max of P( instance = class | instanceData ) for all classes
            # P( instance = class | instanceData) = P(instanceData | instance = class) P(class)
            # we are using naive bayes so P( instanceData | instanceClass) is just a chain of probabilities
            # P(var1 | class) * P(var2 | class) ... * P(class)
            maxProb = 0.0
            classifiedAsIndex = 0
            for classIndex, classValue in enumerate(self.classifiers):
                probClass = self.probabilities[classIndex].classifierProb # P(class)
                for attrIndex, attrValue in enumerate(instance):
                    attrValueIndex = self.getAttributeValueIndexFromValue(attrValue, attrIndex)
                    probClass *= self.probabilities[classIndex].attributeProb[attrIndex][attrValueIndex]
                if probClass > maxProb:
                    classifiedAsIndex = classIndex
                    maxProb = probClass
            if self.classifiers[classifiedAsIndex] == instanceClass:
                numCorrect += 1
            numTotal += 1

        message = "Accuracy on " + ("training " if isTraining else "test ") + "set "
        message += "({0} instances):  {1:.2f}% = {2}/{0}"
        print message.format(numTotal, numCorrect / float(numTotal) * 100, numCorrect)

    # a container which holds probabilities related to a class
    class probabilityContainer():
        def __init__(self, classifierIndex, nb):
            self.classifier = classifierIndex
            self.classifierProb = nb.computeProb(classifierIndex, True)
            self.attributeProb = [[]]  # [attributeIndex][attributeValueIndex] = prob

        def printAllProbs(self, nb):
            classValue = str(nb.classifiers[self.classifier])
            msg = "P(" + classValue +")="+str(self.classifierProb)+" "
            for attrIndex, attrValueList in enumerate(self.attributeProb):
                for attrValueIndex, prob in enumerate(attrValueList):
                    msg += "P("+str(nb.attributes[attrIndex]) +"=" + str(nb.attributeValues[attrIndex][attrValueIndex]) + "|" +str(classValue)+")="+str(prob)+ " "
            print(msg)