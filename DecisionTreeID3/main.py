__author__ = 'Andrew Paettie'

'''
    Main runner code for Intro to Machine Learning Homework 1
    The main class is simple, basically just building the binary decision tree object and using it's methods
    The input is 2 files, passed as arguments to the program.
        The first argument is the filename of a file containing training data,
        The second is the name of a file containing test data.
        The file-format is briefly: whitespace delimited attributes where last attribute is class, and each line is an instance
        Since this is a binary decision tree, only values of 1 and 0 are valid for attributes and classes
    The training data is used to build a decision tree based on the ID3 algorithm; this tree is printed to STDOUT
    The decision tree is used to classify both test and training instances.
    The overall accuracy of these classifications is printed to STDOUT
    Additionaly, a boolean flag, production, can be set to indicate whether or not this homework has been turned in.
        If this flag is set to false, debug information will be output, and the program will take an optional argument
        This argument specifies the number of training instances to use to build the decision tree.
        The reason I did this is that this makes it easier to create a 'learning curve', while still meeting the requirement
            of exactly 2 arguments to the program
'''

import sys
import DecisionTree

production = True # change this to false to turn on debug stuff

def printUsage():
    message = "Usage: python main.py </path/to/training file> </path/to/test file>"
    if not production:
        message += " [training set size]"
    print (message)


def main(argv):
    # sanity check
    if production and argv.__len__() != 3: # if we are in production, limit arguments to 2 as per specification
        printUsage()
        exit()
    elif argv.__len__() < 3 or argv.__len__() > 4:
        printUsage()
        exit()

    dTree = DecisionTree.DecisionTree()

    if not production and argv.__len__() == 4:
        sampleSize = int(argv[3])
        # read training and test data
        dTree.readFile(argv[1], True, sampleSize)  # training
        dTree.readFile(argv[2], False, sampleSize)  # testing
    else:
        dTree.readFile(argv[1], True)  # training
        dTree.readFile(argv[2], False)  # testing
    # get lists of indexes of instances which are classed 0 or 1
    onesAndZeros = dTree.getClassIndexes(dTree.trainingAttributeClasses,
                                         range(0, dTree.trainingAttributeClasses.__len__()))
    dTree.root.zeroList = onesAndZeros[0]
    dTree.root.oneList = onesAndZeros[1]
    dTree.root.entropy = dTree.getEntropy(dTree.trainingAttributeClasses,
                                          range(0, dTree.trainingAttributeClasses.__len__()))
    dTree.root.attrSplitCandidates = range(0, dTree.attributes.__len__())
    dTree.root = dTree.learnTree(dTree.root)
    # print tree
    dTree.printTree(dTree.root)
    # classify training instances and display training accuracy
    dTree.classifyInstances(dTree.trainingAttributeClasses.__len__(), True)
    # classify test instances and display test accuracy
    dTree.classifyInstances(dTree.testAttributeClasses.__len__(), False)


if __name__ == "__main__":
    main(sys.argv)
