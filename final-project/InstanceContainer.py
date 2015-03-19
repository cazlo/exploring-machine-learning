__author__ = 'Andrew Paettie'

class DataContainer(object):
    def __init__(self):
        self.data=[[]]
        self.targets=[]
        self.test=[[]]
        self.testTargets=[]
        self.final=[[]]


    def loadFromFile(self, filename):
        f = open(filename, 'r')
        lineNum = 0
        for line in f:
            splitOnSpaces = line.split()
            if splitOnSpaces.__len__() == 0:
                continue
            elif splitOnSpaces.__len__() == 1:
                continue
            else:
                for attrIndex, attr in enumerate(splitOnSpaces):
                    if attrIndex == splitOnSpaces.__len__()-1:
                        self.targets.append(int(attr))
                    else:
                        if self.data.__len__() <= lineNum:
                            self.data.append([])
                        self.data[lineNum].append(float(attr))
                lineNum += 1

    def loadTestFromFile(self, filename):
        f = open(filename, 'r')
        lineNum = 0
        for line in f:
            splitOnSpaces = line.split()
            if splitOnSpaces.__len__() == 0:
                continue
            elif splitOnSpaces.__len__() == 1:
                continue
            else:
                for attrIndex, attr in enumerate(splitOnSpaces):
                    if attrIndex == splitOnSpaces.__len__()-1:
                        self.testTargets.append(int(attr))
                    else:
                        if self.test.__len__() <= lineNum:
                            self.test.append([])
                        self.test[lineNum].append(float(attr))
                lineNum += 1

    def loadFinalFromFile(self, filename):
        f = open(filename, 'r')
        lineNum = 0
        for line in f:
            splitOnSpaces = line.split()
            if splitOnSpaces.__len__() == 0:
                continue
            elif splitOnSpaces.__len__() == 1:
                continue
            else:
                for attrIndex, attr in enumerate(splitOnSpaces):
                    if self.final.__len__() <= lineNum:
                        self.final.append([])
                    self.final[lineNum].append(float(attr))
                lineNum += 1

    def getXY(self):
        return self.data, self.targets

    def getTestXY(self):
        return self.test, self.testTargets