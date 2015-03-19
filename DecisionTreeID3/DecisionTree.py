import math
import sys
import main

__author__ = 'Andrew Paettie'

'''
    The binary decision tree object
    Contains attributes and data read from training and testing files
        The file-format is briefly: whitespace delimited attributes where last attribute is class, and each line is an instance
        Since this is a binary decision tree, only values of 1 and 0 are valid for attributes and classes
    This data is used to build a decision tree based on the ID3 algorithm
    The decisionTree object itself contains a subclass: Node
        This is essentially just a container for instances' indicies and the nodes' parents and children
        Storing instances' nodes based on index and class value simplifies the calculation of conditional entropy
'''

class DecisionTree(object):
    # 'Static' stuff
    attributes = []
    trainingAttributeData = [[]]
    trainingAttributeClasses = []
    testAttributeData = [[]]
    testAttributeClasses = []

    root = None

    def __init__(self):
        self.root = DecisionTree.Node()

    # recursively called on children of start node until a leaf node is reached
    def printTree(self, start):
        line = ''
        for i in range(0, start.parents.__len__() - 1):
            line += '| '
        if not start.splitOnName is None and not start.splitOnValue is None:
            line = line + self.attributes[start.splitOnName] + ' = ' + str(start.splitOnValue) + ' : '
            if start.isLeaf():
                line += str(start.nodeOutcome)
                print(line)
            else:
                print(line)
                for child in start.children:
                    self.printTree(child)
        else:
            if start.nodeOutcome is None:
                for child in start.children:
                    self.printTree(child)
            else:
                print(": " + str(start.nodeOutcome))

    # parses a file for instance data and stores it in teh DecisionTree object
    # if limit is supplied, only that many instances will be read
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
                    if (index != splitOnSpaces.__len__() - 1):
                        self.attributes.append(attribute)
                # if dTree.attributes[dTree.attributes.__len__()-1] != 'class':
                # print('Last attribute found is '+ dTree.attributes[dTree.attributes.__len__()-1])
                # print('Removing this assuming it is class descriptor')
                # dTree.attributes.remove(dTree.attributes.__len__()-1)
                # else:
                # dTree.attributes.remove(dTree.attributes.__len__()-1)
                self.attrSplitCandidates = self.attributes
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
                for attribute in splitOnSpaces:
                    if attrCount == splitOnSpaces.__len__() - 1:
                        classesList.append(int(attribute))
                    else:
                        dataList[lineNum - 1].append(int(attribute))
                    attrCount += 1
                lineNum += 1
        return self

    # recursively builds a subtree based on maximizing information gain (by minimizing conditional entropy)
    def learnTree(self, startNode):
        # check for all test instances having same class value
        if startNode.zeroList.__len__() == 0 and startNode.oneList.__len__() == 0:
            return None

        # check for all test instances having same attribute value
        elif startNode.zeroList.__len__() == 0:
            startNode.nodeOutcome = 1
            return startNode
        elif startNode.oneList.__len__() == 0:
            startNode.nodeOutcome = 0
            return startNode

        # if no more attributes to split on make leaf with majority class
        elif startNode.attrSplitCandidates.__len__() <= 0:
            # return node with majority class as label
            if startNode.zeroList.__len__() >= startNode.oneList.__len__():
                startNode.nodeOutcome = 0
            else:
                startNode.nodeOutcome = 1
            return startNode

        # calculate conditional entrophy for splitting on all available attributes
        # minimum entropy -> attribute to split on
        bestAttribute = self.chooseBestAttribute(startNode)

        # create a new sub-tree for each child based on splitting on best attribute
        for classifier in range(0, 2):
            # Setup new child and attributes
            newChild = self.Node()
            newChild.parents = startNode.parents[:]
            newChild.parents.extend([startNode])
            newChild.zeroList = []
            newChild.oneList = []
            newChild.splitOnName = bestAttribute
            newChild.splitOnValue = classifier
            newChild.attrSplitCandidates = startNode.attrSplitCandidates[:]
            newChild.attrSplitCandidates.remove(bestAttribute)
            for instance in startNode.zeroList:
                if self.trainingAttributeData[instance][bestAttribute] == classifier:
                    newChild.zeroList.append(instance)

            for instance in startNode.oneList:
                if self.trainingAttributeData[instance][bestAttribute] == classifier:
                    newChild.oneList.append(instance)

            newChild = self.learnTree(newChild)
            if (newChild != None):# only add non-null subtrees
                startNode.children.append(newChild)
        return startNode

    def getEntropy(self, classList, indiciesToSearch):
        prob0 = 0.0
        prob1 = 0.0
        zerosAndOnes = self.getClassIndexes(classList, indiciesToSearch)
        total = zerosAndOnes[0].__len__() + zerosAndOnes[1].__len__()
        if not total == 0:
            prob0 = zerosAndOnes[0].__len__() / float(total)
            prob1 = zerosAndOnes[1].__len__() / float(total)
        if prob0 == 0.0 and prob1 == 0.0:
            entropy = 0
        elif prob0 == 0.0:
            entropy = - prob1 * math.log(prob1, 2)
        elif prob1 == 0.0:
            entropy = - prob0 * math.log(prob0, 2)
        else:
            entropy = - prob0 * math.log(prob0, 2) - prob1 * math.log(prob1, 2)
        return entropy

    def getConditionalEntropy(self, node, attribute):
        # instance<class>GoingTo<attributeValue>
        instance1GoingTo0 = []
        instance1GoingTo1 = []
        instance0GoingTo0 = []
        instance0GoingTo1 = []
        # find out which instances are moving to what attribute value
        for instance in node.oneList:
            nodeAttrValue = self.trainingAttributeData[instance][attribute]

            if (nodeAttrValue == 0):
                instance1GoingTo0.append(instance)
            else:
                instance1GoingTo1.append(instance)

        for instance in node.zeroList:
            nodeAttrValue = self.trainingAttributeData[instance][attribute]
            if (nodeAttrValue == 0):
                instance0GoingTo0.append(instance)
            else:
                instance0GoingTo1.append(instance)

        total = node.zeroList.__len__() + node.oneList.__len__()
        if total != 0: # total should never actually be 0, but just in case, dont / by 0
            prob0 = float(instance0GoingTo0.__len__() + instance1GoingTo0.__len__()) / total
            prob1 = float(instance0GoingTo1.__len__() + instance1GoingTo1.__len__()) / total
            left = prob0 * self.getEntropy(self.trainingAttributeClasses, instance1GoingTo0 + instance0GoingTo0)
            right = prob1 * self.getEntropy(self.trainingAttributeClasses, instance1GoingTo1 + instance0GoingTo1)
            return left + right
        else:
            return 0

    # find attribute whose split results in the lowest conditional entropy
    def chooseBestAttribute(self, node):
        bestEntropy = 1
        best = node.attrSplitCandidates[0]
        for attribute in node.attrSplitCandidates:
            if self.getConditionalEntropy(node, attribute) < bestEntropy:
                best = attribute
                bestEntropy = self.getConditionalEntropy(node, attribute)
        return best

    # takes as parameter list of indexes to search
    # returns 2d list,
    #   first element is list of all indexes of data classified as 0,
    #   second is list of all indexes of data classified as 1
    def getClassIndexes(self, classList, indiciesToSearch):
        zeros = []
        ones = []

        for index in indiciesToSearch:
            if classList[index] == 0:
                zeros.append(index)
            elif classList[index] == 1:
                ones.append(index)

        return [zeros, ones]

    # recursive call to walk down appropriate path to leaf, finally returning class label of leaf
    def classifyInstance(self, instanceIndex, currNode, isTraining):
        if currNode.isLeaf():
            return currNode.nodeOutcome
        else:
            for child in currNode.children:
                instanceAttrValue = self.trainingAttributeData[instanceIndex][child.splitOnName] if isTraining else \
                    self.testAttributeData[instanceIndex][child.splitOnName]
                if child.splitOnValue == instanceAttrValue:
                    return self.classifyInstance(instanceIndex, child, isTraining)

    # method which classifies instances and prints out the accuracy
    # calls recursive method on each instance to classify each one
    def classifyInstances(self, numToClassify, isTraining):
        numCorrect = 0
        for instanceIndex in range(0, numToClassify):
            if self.root.children.__len__() == 0:
                classifier = self.root.nodeOutcome
                if isTraining and classifier == self.trainingAttributeClasses[instanceIndex]:
                    numCorrect += 1
                elif not isTraining and classifier == self.testAttributeClasses[instanceIndex]:
                    numCorrect += 1
                elif not main.production:
                    print("Failed to classify instance #" + str(instanceIndex))
                    print("    @ classifier : " + str(classifier))
            else:
                attrIndex = self.root.children[0].splitOnName
                instanceAttrValue = self.trainingAttributeData[instanceIndex][attrIndex] if isTraining else \
                    self.testAttributeData[instanceIndex][attrIndex]
                for child in self.root.children:
                    if child.splitOnValue == instanceAttrValue:
                        classifier = self.classifyInstance(instanceIndex, child, isTraining)
                        if isTraining and classifier == self.trainingAttributeClasses[instanceIndex]:
                            numCorrect += 1
                            break
                        elif not isTraining and classifier == self.testAttributeClasses[instanceIndex]:
                            numCorrect += 1
                            break
                        elif not main.production:
                            print("Failed to classify instance #" + str(instanceIndex))
                            print("    @ classifier : " + str(classifier))
                            break

        message = "Accuracy on " + ("training " if isTraining else "test ") + "set "
        message += "({0} instances):  {1:.2f}% = {2}/{3}"
        print message.format(numToClassify, numCorrect / float(numToClassify) * 100, numCorrect, numToClassify)

    # internal class for node of tree
    # basically just a container for holding the instances' indices until they are split or classified
    class Node:

        def __init__(self):
            self.parents = []
            self.children = []

            self.zeroList = []  # a list of the indexes of data which is classified as 0
            self.oneList = []

            self.entropy = None

            self.splitOnName = None  # classname
            self.splitOnValue = None  # class value (0 or 1)

            self.nodeOutcome = None  # the class of the node (if leaf)

            self.attrSplitCandidates = []
            pass

        def isLeaf(self):
            return self.children.__len__() == 0