# coding:utf-8
import sys
import re
import json
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from math import sqrt
from tgrocery import Grocery
import nltk
from nltk.corpus import wordnet as wn

RT = ["Cause-Effect", "Component-Whole", "Content-Container", "Entity-Destination", "Entity-Origin", "Instrument-Agency", "Member-Collection", "Message-Topic", "Product-Producer", "Other"]
ALL_RT = ["Cause-Effect(e1,e2)", "Component-Whole(e1,e2)", "Content-Container(e1,e2)", "Entity-Destination(e1,e2)", "Entity-Origin(e1,e2)", "Instrument-Agency(e1,e2)", "Member-Collection(e1,e2)", "Message-Topic(e1,e2)", "Product-Producer(e1,e2)", "Cause-Effect(e2,e1)", "Component-Whole(e2,e1)", "Content-Container(e2,e1)", "Entity-Destination(e2,e1)", "Entity-Origin(e2,e1)", "Instrument-Agency(e2,e1)", "Member-Collection(e2,e1)", "Message-Topic(e2,e1)", "Product-Producer(e2,e1)", "Other"]

def GetRelation(string):
    if string == "Other":
        return string
    return string[:-7]

def OtherSynset(line,tagged,ALL=0):
    word = []
    for k in tagged:
        word.append(WN(k[0], k[1], ALL))
    return ' '.join(word)

def hype(sys, ALL=0):
    if ALL == 0:
        return str(sys)
    H = wn.synset(sys.name()).hypernym_paths()
    LH = len(H)
    if ALL == 2 and LH > 1:
        h = H[1] 
    else:
        h = H[0]
    L = len(h)
    N = 7
    if L > N:
        return str(h[N])
    return str(sys) 

def WN(word, pos=None, ALL=0):
    if pos[:2] == 'NN':
        P = ['n']
    elif pos[:2] == 'VB':
        P = ['v']
    elif pos[:2] == 'JJ':
        P = ['a', 's']
    elif pos[:2] == 'RB':
        P = ['r']
    else:
        return word
    syn = wn.synsets(word)
    N = len(syn)
    if N == 0:
        return word
    for i in range(N):
        if str(syn[i])[-6] == P[0]:
            return hype(syn[i],ALL)[8:-7]
        elif len(P) > 1 and str(syn[i])[-6] == P[1]:
            return hype(syn[i],ALL)[8:-7]
    return hype(syn[0],ALL)[8:-7]

def Synset(line, ALL=0):
    S1 = line.split('<e1>')
    S2 = S1[1].split('</e1>')
    S3 = S2[1].split('<e2>')
    S4 = S3[1].split('</e2>')
    temp = line.replace("<e1>","").replace("</e1>","").replace("<e2>","").replace("</e2>","")
    tokens = nltk.word_tokenize(temp)
    L1 = len(nltk.word_tokenize(S1[0]))
    L2 = L1 + len(nltk.word_tokenize(S2[0]))
    L3 = L2 + len(nltk.word_tokenize(S3[0]))
    L4 = L3 + len(nltk.word_tokenize(S4[0]))
    tagged = nltk.pos_tag(tokens)
    #line = OtherSynset(S1[0],tagged[:L1],ALL) + ' <e1>' + OtherSynset(S2[0],tagged[L1:L2],ALL) + '</e1> ' + OtherSynset(S3[0],tagged[L2:L3],ALL) + ' <e2>' + OtherSynset(S4[0],tagged[L3:L4],ALL) + '</e2> ' + OtherSynset(S4[1],tagged[L4:],ALL)
    line = '<e1>' + OtherSynset(S2[0],tagged[L1:L2],ALL) + '</e1> ' + OtherSynset(S3[0],tagged[L2:L3],ALL) + ' <e2>' + OtherSynset(S4[0],tagged[L3:L4],ALL) + '</e2>'
    return line

def Judge(X):
    MAX = -100
    INDEX = ""
    for k in X:
        if MAX < X[k]:
            MAX = X[k]
            INDEX = k
    return INDEX

class DataManager:
    def __init__(self):
        self.ID = 0
        self.Sentence = ""
        self.Relation = ""
        self.Comment = ""
        self.Sentence_1 = ""
        self.Sentence_2 = ""

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
        temp.Sentence_1 = Synset(temp.Sentence, 1)
        temp.Sentence_2 = Synset(temp.Sentence, 2)
        #temp.Sentence = Synset(temp.Sentence)
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
    temp.Sentence_1 = Synset(string[1][1:-2], 1)
    temp.Sentence_2 = Synset(string[1][1:-2], 2)
    #temp.Sentence = Synset(string[1][1:-2])
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

E_grocery = Grocery('E_test')
E_train_src = []
grocery = Grocery('test')
train_src = []
for i in range(8000):
    R = GetRelation(TrainingData[i].Relation)
    E_train_src.append((R,TrainingData[i].Sentence_1))
    E_train_src.append((R,TrainingData[i].Sentence_2))
    #train_src.append((TrainingData[i].Relation,TrainingData[i].Sentence))
    train_src.append((TrainingData[i].Relation,TrainingData[i].Sentence_1))
    train_src.append((TrainingData[i].Relation,TrainingData[i].Sentence_2))
grocery.train(train_src)
E_grocery.train(E_train_src)


data = {}
results = dict()
for i in range(8000):
    rlt_1 = grocery.predict(TrainingData[i].Sentence_1).dec_values
    rlt_2 = grocery.predict(TrainingData[i].Sentence_2).dec_values
    E_rlt_1 = E_grocery.predict(TrainingData[i].Sentence_1).dec_values
    E_rlt_2 = E_grocery.predict(TrainingData[i].Sentence_2).dec_values
    for k in ALL_RT:
        R = GetRelation(k)
        results[k] = (rlt_1[k]**3 + rlt_2[k]**3)*abs(E_rlt_1[R]**3 + E_rlt_2[R]**3)
    data[str(1+i)] = results
with open('TG_train_2.json', 'w') as outfile:
    json.dump(data, outfile)
outfile.close()

outfile = open('result_2.txt', 'w', encoding='utf-8')
test_src = []
data = {}
results = dict()
for i in range(N):
    #R = GetRelation(TestingData[i].Relation)
    #test_src.append((TestingData[i].Relation,TestingData[i].Sentence))
    #A = grocery.predict(TestingData[i].Sentence).dec_values
    B = grocery.predict(TestingData[i].Sentence_1).dec_values
    C = grocery.predict(TestingData[i].Sentence_2).dec_values
    E_B = E_grocery.predict(TestingData[i].Sentence_1).dec_values
    E_C = E_grocery.predict(TestingData[i].Sentence_2).dec_values
    MAX = -100.0
    ans = ""
    for k in ALL_RT:
        R = GetRelation(k)
        score = (B[k]**3 + C[k]**3)*abs(E_B[R]**3+ E_C[R]**3)
        results[k] = score 
        if MAX < score:
            MAX = score
            ans = k
        data[str(8001+i)] = results
    buf = str(8001+i) + '\t' + ans
    print(buf, file=outfile)
outfile.close()
with open('TG_test_2.json', 'w') as outfile:
    json.dump(data, outfile)
outfile.close()