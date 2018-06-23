# coding:utf-8
from enum import Enum
from sklearn
import re
import nltk

TrainFile = "TRAIN_FILE.txt"
TestFile = "TEST_FILE.txt"

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

class TrainingDataManager:
    def __init__(self):
        self.id = -1
        self.sentence = []
        self.pos = []
        self.index1 = -1 # index of e1
        self.index2 = -1 # index of e2
        self.relation = Relation.OTHER
        self.reverse = False # if the relation is (e2,21)

    def insertData(self, lines):
        tmp = re.split('\"|\t|\n|\.| ',lines[0])
        self.id = int(tmp[0])
        cnt = 0
        for index, word in enumerate(tmp):
            tmp = ''
            if re.match('<e1>', word):
                tmp = word[4:-5]
                self.index1 = cnt
            elif re.match('<e2>', word):
                tmp = word[4:-5]
                self.index2 = cnt
            else: tmp = word
            if ( index == 0 or tmp == '' ): continue
            self.sentence.append(tmp)
            # do POS tagging
            tmppos = nltk.pos_tag([tmp])[0][1]
            self.pos.append ( tmppos )
            if( tmppos[0] == 'V' ): # is verb
                if( tmp not in verbs ):
                    verbs.append(tmp)
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
                dataList.append(manager)
                manager = TrainingDataManager()
                lines = []
            else:
                lines.append(l)
            l = f.readline()
    return dataList

def main():
    TrainingData = readTrainingFile(TrainFile)
    for dt in TrainingData:
        verb = dt.verbKernelVector()
        dist = dt.verbKernelVector()

if __name__ == "__main__":
    main()