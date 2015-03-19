
__author__ = 'Andrew Paettie'

'''
    Main runner for the binary gradient descent perceptron.
    1. Read data
    2. Learn data
    3. ???
    4. Skynet
'''

import BinaryGDClassifier
import sys

def printUsage():
    message = "Usage: python main.py </path/to/training file> </path/to/test file> <learningRate> <numIterations>"
    print (message)


def main(argv):
    # sanity check
    if argv.__len__() != 5:  # 5 because the first arg is the command used to run the program
        printUsage()
        exit()

    classifier = BinaryGDClassifier.GDClassifier(float(argv[3]), int(argv[4]))

    classifier.readFile(argv[1], True)    # Training set
    classifier.readFile(argv[2], False)   # Test set
    classifier.learn()
    classifier.classify(True)   # Training set
    classifier.classify(False)  # Test set


if __name__ == "__main__":
    main(sys.argv)
