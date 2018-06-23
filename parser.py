# coding:utf-8
from enum import Enum
from sklearn.svm import SVC
import re
import nltk

TrainFile = "TRAIN_FILE.txt"
TestFile = "TEST_FILE.txt"
OutFile = "my_answer2.txt"

verbs = []

class Relation(Enum):
    CE = 0 # Cause-Effect = 0
    IA = 1 # Instrument-Agency = 1
    PP = 2 # Product-Producer = 2
    CC = 3 # Content-Container = 3
    EO = 4 # Entity-Origin
    ED = 5 # Entity-Destination
    CW = 6 # Component-Whole
    MC = 7 # Member-Collection
    MT = 8 # Message-Topic
    OTHER = 9 # Other

def relation2string(relation):
    ans = "Other"
    if( relation == Relation.CE.value ): ans = "Cause-Effect"
    elif( relation == Relation.IA.value ): ans = "Instrument-Agency"
    elif( relation == Relation.PP.value ): ans = "Product-Producer"
    elif( relation == Relation.CC.value ): ans = "Content-Container"
    elif( relation == Relation.EO.value ): ans = "Entity-Origin"
    elif( relation == Relation.ED.value ): ans = "Entity-Destination"
    elif( relation == Relation.CW.value ): ans = "Component-Whole"
    elif( relation == Relation.MC.value ): ans = "Member-Collection"
    elif( relation == Relation.MT.value ): ans = "Message-Topic"
    return ans

class DataManager:
    def __init__(self):
        self.id = -1
        self.sentence = []
        self.index1 = -1 # index of e1
        self.index2 = -1 # index of e2

    # kernel 2.1
    def localContextKernelVector(self):
        vector = []
        return vector

    # kernel 2.2
    def verbKernelVector(self):
        vector = []
        for w in verbs:
            if w in self.sentence:
                vector.append(1.0)
            else: vector.append(0.0)
        return vector

    # kernel 2.3
    def distanceKernelVector(self):
        vector = [(1.0)/abs(self.index1 - self.index2)]
        return vector

    # kernel 2.4
    def cycKernelVector(self):
        vector = []
        return vector


class TrainingDataManager(DataManager):
    def __init__(self):
        super().__init__()
        self.pos = []
        self.relation = Relation.OTHER
        self.reverse = False # if the relation is (e2,e1)

    def insertData(self, lines):
        tmp = re.split('\"|\t|\n|\.| ',lines[0])
        self.id = int(tmp[0])
        cnt = 0
        for index, word in enumerate(tmp):
            w = ''
            if re.match('<e1>', word):
                w = word[4:-5]
                self.index1 = cnt
            elif re.match('<e2>', word):
                w = word[4:-5]
                self.index2 = cnt
            else: w = word
            if ( index == 0 or w == '' ): continue
            self.sentence.append(w)
            # do POS tagging
            tmppos = nltk.pos_tag([w])[0][1]
            self.pos.append ( tmppos )
            if( tmppos[0] == 'V' ): # is verb
                if( w not in verbs ):
                    verbs.append(w)
            cnt += 1

        tmp = re.split('\(|\)|\,',lines[1])
        if re.match('Cause-Effect', tmp[0]): self.relation = Relation.CE
        elif re.match('Instrument-Agency', tmp[0]): self.relation = Relation.IA
        elif re.match('Product-Producer', tmp[0]):  self.relation = Relation.PP
        elif re.match('Content-Container', tmp[0]): self.relation = Relation.CC
        elif re.match('Entity-Origin', tmp[0]): self.relation = Relation.EO
        elif re.match('Entity-Destination', tmp[0]):self.relation = Relation.ED
        elif re.match('Component-Whole', tmp[0]):   self.relation = Relation.CW
        elif re.match('Member-Collection', tmp[0]): self.relation = Relation.MC
        elif re.match('Message-Topic', tmp[0]): self.relation = Relation.MT
        else: self.relation = Relation.OTHER
        if ( self.relation != Relation.OTHER ): self.reverse = ( tmp[1] == "e2" )


class TestingDataManager(DataManager):

    def __init__(self):
        super().__init__()

    def insertData(self, line):
        tmp = re.split('\"|\t|\n|\.| ',line)
        self.id = int(tmp[0])
        cnt = 0
        for index, word in enumerate(tmp):
            w = ''
            if re.match('<e1>', word):
                w = word[4:-5]
                self.index1 = cnt
            elif re.match('<e2>', word):
                w = word[4:-5]
                self.index2 = cnt
            else: w = word
            if ( index == 0 or w == '' ): continue
            self.sentence.append(w)
            cnt += 1


# Read Training File
def readTrainingFile(path):
    dataList = []
    with open(path, 'r') as f:
        manager = TrainingDataManager()
        l = f.readline()
        lines = []
        while l:
            if l == '\n':
                manager.insertData(lines)
                if( manager.relation != Relation.OTHER ):
                    dataList.append(manager)
                manager = TrainingDataManager()
                lines = []
            else:
                lines.append(l)
            l = f.readline()
    return dataList


# Read Testing File
def readTestingFile(path):
    dataList = []
    with open(path, 'r') as f:
        l = f.readline()
        while l:
            manager = TestingDataManager()
            manager.insertData(l)
            dataList.append(manager)
            l = f.readline()
    return dataList


def main():
    TrainingData = readTrainingFile(TrainFile)
    print("Finish Preprocessing.")
    TrainingX = []
    TrainingY = []
    for dt in TrainingData:
        verb = dt.verbKernelVector()
        dist = dt.distanceKernelVector()
        TrainingX.append(verb + dist)
        TrainingY.append(dt.relation.value)

    # fit model by training set
    print("Start Training")
    clf = SVC(kernel='precomputed', probability=True, verbose=True)
    clf.fit(TrainingX, TrainingY)

    # testing data
    TestingData = readTestingFile(TestFile)
    print("Finish reading testing file.")

    TestingX = []
    for dt in TestingData:
        verb = dt.verbKernelVector()
        dist = dt.distanceKernelVector()
        TestingX.append(verb + dist)

    TestingY = clf.predict(TestingX).tolist()
    print(TestingY)
    print(type(TestingY))
    print("Finish predicting, start writing result to file.")

    # write out result
    with open(OutFile, 'w') as f:
        for index, y in enumerate(TestingY):
            line = str(index + 8001) + '\t' + relation2string(y) + "(e1,e2)\n"
            f.write(line)


if __name__ == "__main__":
    main()