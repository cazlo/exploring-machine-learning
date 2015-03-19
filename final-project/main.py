from __future__ import print_function


import numpy
import sys

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import safe_asarray
from sklearn.pipeline import Pipeline
from sklearn import tree, cross_validation, metrics
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.feature_selection import RFECV, SelectPercentile, chi2
from sklearn.svm import SVC, LinearSVC

import InstanceContainer

__author__ = 'Andrew Paettie'

'''
    Program to find the best classifier/classifer configuration for the given input
'''

def main(argv):

    if argv.__len__() != 4:
        printUsageAndQuit()

    dataContainer = InstanceContainer.DataContainer()
    dataContainer.loadFromFile(argv[1])
    dataContainer.loadTestFromFile(argv[2])
    dataContainer.loadFinalFromFile(argv[3])

    X, y = dataContainer.getXY()
    testX, testy = dataContainer.getTestXY()
    allX, ally = X+testX, y+testy
    finalX = dataContainer.final

    #pre-process the data to remove the statistically worst features
    print("Selecting best 30% of features according to the chi2 algorithm")
    bestFeatures = SelectPercentile(chi2, 30).fit(X, y)
    X_best = bestFeatures.transform(X)
    testX_best = bestFeatures.transform(testX)
    allX_best = bestFeatures.transform(allX)
    finalX_best = bestFeatures.transform(finalX)

    #use a linearsvc to select best features
    #first we need to find optimal parameter for C
    Crange = numpy.logspace(-8, 2, 30)
    svcParams={'C' : Crange}
    svcGrid = GridSearchCV(LinearSVC(penalty="l1", dual=False), param_grid=svcParams, cv=2, n_jobs=-1)
    print("Finding optimal parameter for SVC (used for feature selection)")
    svcGrid.fit(allX_best, ally)
    print("best param choice:", svcGrid.best_params_)
    print("crossval-accuracy:", cross_val_score(svcGrid.best_estimator_, allX_best, safe_asarray(ally), cv=5, n_jobs=-1))

    #now use a tree to classify
    # first lets try messing with the parameters
    treeParams={'max_depth' : [4, 5, 6, 7, 8, 9, 10, 11, 12],
                'min_samples_leaf' : [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
    treeGrid = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=treeParams, cv=4, n_jobs=-1)
    print("Finding optimal parameters for decision tree just for train set")
    treeGrid.fit(svcGrid.transform(X_best), y)
    #print("Decision tree # features:", svcGrid.transform(X).__len__())
    print("best param choices:", treeGrid.best_params_)
    #testPredictions = treeGrid.predict(svcGrid.transform(testX))
    testPredictions = treeGrid.predict(svcGrid.transform(testX_best))
    allPredictions = treeGrid.predict(svcGrid.transform(allX_best))
    print("test accuracy:", metrics.accuracy_score(testy, testPredictions))
    print("overall accuracy:", metrics.accuracy_score(ally, allPredictions))
    print("crossval-accuracy:", cross_val_score(treeGrid.best_estimator_, svcGrid.transform(allX_best), safe_asarray(ally), cv=5, n_jobs=-1))

    finalPrediction = treeGrid.predict(svcGrid.transform(finalX_best))
    print("Final predictions:", finalPrediction)
    f = open('output1.txt', 'w')
    for p in finalPrediction:
        print(p, file=f)
    f.close()

    print("Finding optimal parameters for decision tree just for test set")
    treeGrid.fit(svcGrid.transform(testX_best), testy)
    print("best param choices:", treeGrid.best_params_)
    trainPredictions = treeGrid.predict(svcGrid.transform(X_best))
    allPredictions = treeGrid.predict(svcGrid.transform(allX_best))
    print("train accuracy:", metrics.accuracy_score(y, trainPredictions))
    print("overall accuracy:", metrics.accuracy_score(ally, allPredictions))
    print("crossval-accuracy:", cross_val_score(treeGrid.best_estimator_, svcGrid.transform(allX_best), safe_asarray(ally), cv=5, n_jobs=-1))

    finalPrediction = treeGrid.predict(svcGrid.transform(finalX_best))
    print("Final predictions:", finalPrediction)
    f = open('output2.txt', 'w')
    for p in finalPrediction:
        print(p, file=f)
    f.close()

    print("Finding optimal parameters for decision tree over all known instances (test + train)")
    treeGrid.fit(svcGrid.transform(allX_best), ally)
    #print("Decision tree # features:", svcGrid.transform(X).__len__())
    print("best param choices:", treeGrid.best_params_)
    testPredictions = treeGrid.predict(svcGrid.transform(testX_best))
    trainPredictions = treeGrid.predict(svcGrid.transform(X_best))
    print("test accuracy:", metrics.accuracy_score(testy, testPredictions))
    print("train accuracy:", metrics.accuracy_score(y, trainPredictions))
    print("crossval-accuracy:", cross_val_score(treeGrid.best_estimator_, svcGrid.transform(allX_best), safe_asarray(ally), cv=5, n_jobs=-1))

    finalPrediction = treeGrid.predict(svcGrid.transform(finalX_best))
    print("Final predictions:", finalPrediction)
    f = open('output3.txt', 'w')
    for p in finalPrediction:
        print(p, file=f)
    f.close()

    #lets try a Gaussian Naive Bayes classifier to see if that improves accuracy
    #print("transforming data from tree to naive bayes classifier")
    #gnb = GaussianNB()
    #gnbX = treeGrid.transform(bestFeatures.transform(X))
    #gnbTestX = treeGrid.transform(bestFeatures.transform(testX))
    #trainPredictions = gnb.fit(gnbX, y).predict(gnbTestX)
    #print("train accuracy:", metrics.accuracy_score(testy, trainPredictions))
    #testPredictions = gnb.fit(gnbTestX, testy).predict(gnbX)
    #print("test accuracy:", metrics.accuracy_score(y, testPredictions))
    # that made accuracy worse, so that sucks

    #lets try a perceptron classifier, which basically randomly selects values to control over/under fitting
    # perceptronParams={'penalty' : [None, 'l2', 'l1'],
    #                   'alpha' : Crange}
    # print("transforming data from tree to perceptron")
    # perceptron = GridSearchCV(Perceptron(n_jobs=-1, n_iter=100), param_grid=perceptronParams, cv=2, n_jobs=-1)
    # pX = bestFeatures.transform(X)
    # pTestX = bestFeatures.transform(testX)
    # perceptron.fit(pX, y)
    # print("best param choices:", perceptron.best_params_)
    # testPreds = perceptron.predict(pTestX)
    # trainPreds = perceptron.fit(pTestX, testy).predict(pX)
    # print("test accuracy:", metrics.accuracy_score(testy, testPreds))
    # print("train accuracy:", metrics.accuracy_score(y, trainPreds))
    #that has really bad accuracy so lets not use that

def printUsageAndQuit():
    print("Usage: python main.py <path/to/train.txt> <path/to/prelim-labeled.txt> <path/to/final-unlabled.txt>")
    exit(1)

if __name__ == "__main__":
    main(sys.argv)