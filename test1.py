# coding:utf-8
import sys
import re
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from math import sqrt
from tgrocery import Grocery
from nltk.corpus import wordnet as wn

RT = ["Cause-Effect", "Component-Whole", "Content-Container", "Entity-Destination", "Entity-Origin", "Instrument-Agency", "Member-Collection", "Message-Topic", "Product-Producer", "Other"]

def GetRelation(string):
    if string == "Other":
        return string
    return string[:-7]

def ASC2(word):
    lonn = len(word)
    tem = 0
    t1 = 0
    t2 = lonn
    for i in range(0,lonn):
        if(tem==0):
            if((word[i] >= "0") and (word[i] <= "9")):
                tem = 1
                t1 = i
            elif((word[i] >= "A") and (word[i] <= "Z")):
                tem = 1
                t1 = i
            elif((word[i] >= "a") and (word[i] <= "z")):
                tem = 1
                t1 = i
        elif(tem==1):
            if((word[i] >= "0") and (word[i] <= "9")):
                tem = 1
            elif((word[i] >= "A") and (word[i] <= "Z")):
                tem = 1
            elif((word[i] >= "a") and (word[i] <= "z")):
                tem = 1
            else:
                tem = 2
                t2 = i
    if(t2==lonn):
        return word[0:t1],word[t1:t2],""
    else:
        return word[0:t1],word[t1:t2],word[t2:lonn]

def OtherSynset(line):
    temp = line.split(' ')
    word = []
    for k in temp:
        cut = ASC2(k)
        S = cut[0] + WN(cut[1]) + cut[2] 
        word.append(S)
    return ' '.join(word)

def WN(word, pos=False):
    if pos == False:
        n = -7
    else:
        n = -5
    syn = wn.synsets(word)
    if len(syn) > 0:
        return str(syn[0])[8:n]
    else:
        return word

def Synset(line):
    S1 = line.split('<e1>')
    S2 = S1[1].split('</e1>')
    S3 = S2[1].split('<e2>')
    S4 = S3[1].split('</e2>')
    temp = S2[0].split(' ')
    word1 = []
    for k in temp:
        word1.append(WN(k,True))
    temp = S4[0].split(' ')
    word2 = []
    for k in temp:
        word2.append(WN(k,True))
    #line = OtherSynset(S1[0]) + '<e1>' + ' '.join(word1) + '</e1>' + OtherSynset(S3[0]) + '<e2>' + ' '.join(word2) + '</e2>' + OtherSynset(S4[1])
    line = '<e1>' + ' '.join(word1) + '</e1>' + OtherSynset(S3[0]) + '<e2>' + ' '.join(word2) + '</e2>'
    return line

def Judge(array):
    MAX = -1
    INDEX = 0
    for i in range(10):
        if MAX < array[i]:
            MAX = array[i]
            INDEX = i
    return RT[INDEX]

class DataManager:
    def __init__(self):
        self.ID = 0
        self.Sentence = ""
        self.Relation = ""
        self.Comment = ""

    def insertData(self,DataList):
        string = DataList[0].split('\t')
        self.ID = int(string[0])
        self.Sentence = string[1][1:-2]
        self.Relation = DataList[1][:-1]
        string = DataList[2].split('Comment:')[1]
        if string != '\n':
            self.Comment = string[1:-1]


TrainingData = []
in_filename = 'TRAIN_FILE.txt'
infile = open(in_filename, 'r', encoding='utf-8')
num = 0
DataElement = []
for line in infile:
    if num == 3:
        temp = DataManager()
        temp.insertData(DataElement)
        temp.Sentence = Synset(temp.Sentence)
        TrainingData.append(temp)
        num = 0
        DataElement = []
        continue
    elif line == '\n':
        break
    DataElement.append(line)
    num += 1
infile.close()

TestingData = []
in_filename = 'TEST_FILE.txt'
infile = open(in_filename, 'r', encoding='utf-8')
for line in infile:
    string = line.split('\t')
    temp = DataManager()
    temp.ID = int(string [0])
    temp.Sentence = Synset(string[1][1:-2])
    TestingData.append(temp)
infile.close()


in_filename = 'answer_key.txt'
infile = open(in_filename, 'r', encoding='utf-8')
N = 0
for line in infile:
    string = line.split('\t')
    TestingData[N].Relation = string[1][:-1]
    N += 1
infile.close()


grocery = Grocery('test')
train_src = []
for i in range(8000):
    R = GetRelation(TrainingData[i].Relation)
    train_src.append((R,TrainingData[i].Sentence))
grocery.train(train_src)
'''
outfile = open('TG_train_1.txt', 'w', encoding='utf-8')
for i in range(8000):
    results = grocery.predict(TrainingData[i].Sentence).dec_values
    score = ''
    for k in RT:
        score += str(results[k]) + '\t'
    print(score[:-1], file=outfile)
outfile.close()
outfile = open('TG_test_1.txt', 'w', encoding='utf-8')
'''
test_src = []
for i in range(N):
    R = GetRelation(TestingData[i].Relation)
    test_src.append((R,TestingData[i].Sentence))
'''
    results = grocery.predict(TestingData[i].Sentence).dec_values
    score = ''
    for k in RT:
        score += str(results[k]) + '\t'
    print(score[:-1], file=outfile)
outfile.close()
'''
test_result = grocery.test(test_src)
print('Accuracy =',test_result)
test_result.show_result()

