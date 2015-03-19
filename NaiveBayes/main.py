__author__ = 'Andrew Paettie'

'''
    Main runner code for Intro to Machine Learning Homework 2.  Basically just a demonstration of the
    NaiveBayes Classifier that I built.  Reads 2 files (whose names are supplied as arguments) and learns
    based off of the information in these files.
'''

import sys
import NaiveBayesClassifier

def printUsage():
    message = "Usage: python main.py </path/to/training file> </path/to/test file>"
    print (message)


def main(argv):
    # sanity check
    if argv.__len__() != 3: # 3 because the first arg is the command used to run the program
        printUsage()
        exit()

    classifier = NaiveBayesClassifier.NaiveBayes()
    classifier.readFile(argv[1], True)  # training
    classifier.readFile(argv[2], False)  # testing

    classifier.learn()
    for pc in classifier.probabilities:
        pc.printAllProbs(classifier)

    classifier.classify(True)
    classifier.classify(False)


if __name__ == "__main__":
    main(sys.argv)
