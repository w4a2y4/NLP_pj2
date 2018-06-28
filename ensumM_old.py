import sys, argparse, os
import keras
import _pickle as pk
import readline
import numpy as np
import json
from keras.models import load_model
from keras.models import Sequential
from keras.utils.data_utils import Sequence
from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional,Activation,CuDNNLSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
import keras.backend.tensorflow_backend as K
import tensorflow as tf
class DataSequence(Sequence):
    def __init__(self,x_set,y_set,batch_size):
        self.batch_size = batch_size
        self.x,self.y=x_set,y_set
    def __len__(self):
        return len(self.y) // self.batch_size
    def __getitem__(self,idx):
        return self.x[idx*self.batch_size:(idx+1)*self.batch_size],self.y[idx*self.batch_size:(idx+1)*self.batch_size]
    def on_epoch_end(self):
        pass
    def __iter__(self):
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item
dictt = ["Cause-Effect(e1,e2)","Cause-Effect(e2,e1)","Component-Whole(e1,e2)","Component-Whole(e2,e1)","Content-Container(e1,e2)","Content-Container(e2,e1)","Entity-Destination(e1,e2)","Entity-Destination(e2,e1)","Entity-Origin(e1,e2)","Entity-Origin(e2,e1)","Instrument-Agency(e1,e2)","Instrument-Agency(e2,e1)","Member-Collection(e1,e2)","Member-Collection(e2,e1)","Message-Topic(e1,e2)","Message-Topic(e2,e1)","Product-Producer(e1,e2)","Product-Producer(e2,e1)","Other"]
def datacut(textname):
	loaddata = open(textname,'r')
	rrr = {}
	rrr["index"] = []
	rrr["answer"] = []
	for line in loaddata:
		t1 = line.split('\t')
		#print t1
		t2 = t1[1].split('\n')
		rrr["index"].append(t1[0])
		rrr["answer"].append(t2[0])
	return rrr
def votee(t1,t2):
	lonnnn = len(dictt)
	w1 = [0.89,0.86,0.74,0.71,0.81,0.8,0.83,0,0.84,0.85,0.47,0.67,0.62,0.83,0.79,0.67,0.78,0.66,0.38]
	w2 = [0.85,0.83,0.6,0.52,0.79,0.75,0.8,0,0.73,0.86,0.43,0.57,0.46,0.75,0.72,0.51,0.6,0.48,0.35]
	w3 = [0.71,0.78,0.56,0.54,0.77,0.6,0.81,0,0.72,0.64,0.27,0.59,0.25,0.76,0.67,0.47,0.31,0.44,0.3]
	tv = []
	for i in range(0,lonnnn):
		ttt = 0
		if t1 == dictt[i]:
			ttt += w1[i]
		if t2 == dictt[i]:
			ttt += w2[i]
		#if t3 == dictt[i]:
		#	ttt += w3[i]
		tv.append(ttt)
	highnum = 0
	highindex = -1
	for i in range(0,lonnnn):
		if tv[i]>highnum:
			highnum = tv[i]
			highindex = i
	return dictt[highindex]
def cutdata(rawtxt):
	outf = []
	for line in rawtxt:
		tt = line.split('\t')
		lonn = len(tt)
		t1 = tt[lonn-1].split('\n')
		tt[lonn-1] = t1[0]
		ttt = []
		for word in tt:
			ttt.append(float(word))
		outf.append(ttt)
	return outf
def Trainanswerkey():
	ans = open('./TRAIN_FILE.txt','r')
	count = 0
	ansnum = 0
	anslist = []
	for line in ans:
		if count%4 == 1:
			#print line
			tt = line.split('\r')
			t1 = tt[0].split('\n')
			keepid = -1
			for i in range(0,19):
				if(t1[0]==dictt[i]):
					keepid = i
			#if(keepid==-1):
				#print(t1[0])
			anslist.append(keepid)
			ansnum += 1
		count += 1
	return anslist
def Testanswerkey():
	ans = open('./answer_key.txt','r')
	count = 0
	ansnum = 0
	anslist = []
	for line in ans:
		tt = line.split('\t')
		t1 = tt[1].split('\n')
		keepid = -1
		for i in range(0,19):
			if(t1[0]==dictt[i]):
				keepid = i
		anslist.append(keepid)
		#print(keepid)
	return anslist
def featureadd(f1,f2):
	lnn = len(f1)
	fa = []
	for i in range(0,lnn):
		ff = []
		ff.append(f1[i])
		ff.append(f2[i])
		fa.append(ff)
	return fa

def predata():
	f1 = open('./TG_data/TG_train_0.txt','r')
	data1 = cutdata(f1)
	f2 = open('./TG_data/TG_train_1.txt','r')
	data2 = cutdata(f2)

	tf1 = open('./TG_data/TG_test_0.txt','r')
	td1 = cutdata(tf1)
	tf2 = open('./TG_data/TG_test_1.txt','r')
	td2 = cutdata(tf2)
	trainingans = Trainanswerkey()
	testans = Testanswerkey()
	totf = featureadd(data1,data2)
	tott = featureadd(td1,td2)
	return totf,tott,trainingans,testans
def AngleRNN(X_data,Y_data,X_test,Y_test,Batchsize,Timesteps,Dimension):
	model = Sequential()
	model.add(CuDNNLSTM(units=32,batch_input_shape=(None,Timesteps,Dimension)))
	model.add(Dropout(0.9))
	#model.add(Dense(64))
	#model.add(Dropout(0.9))
	model.add(Dense(19))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])#adam or nadam
	model.summary()
	checkpointer = ModelCheckpoint(filepath="./savemodel/modelensumble"+"_{epoch:003d}.hdf5", verbose=1, save_best_only=True,mode='max',monitor='val_acc')
	his = model.fit_generator(DataSequence(X_data,Y_data,Batchsize),steps_per_epoch=8000,epochs=100,validation_data=DataSequence(X_test,Y_test,Batchsize),validation_steps=2717,callbacks=[checkpointer])
def AngleRNN1(X_data,Y_data,X_test,Y_test,Batchsize,Timesteps,Dimension):
	model = Sequential()
	model.add(CuDNNLSTM(units=32,batch_input_shape=(None,Timesteps,Dimension)))
	#model.add(Dropout(0.9))
	#model.add(Dense(64))
	#model.add(Dropout(0.9))
	model.add(Dropout(0.5))
	model.add(Dense(19))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])#adam or nadam
	model.summary()
	checkpointer = ModelCheckpoint(filepath="./savemodel/modelensumble1"+"_{epoch:003d}.hdf5", verbose=1, save_best_only=True,mode='max',monitor='val_acc')
	his = model.fit_generator(DataSequence(X_data,Y_data,Batchsize),steps_per_epoch=8000,epochs=100,validation_data=DataSequence(X_test,Y_test,Batchsize),validation_steps=2717,callbacks=[checkpointer])
def AngleRNN2(X_data,Y_data,X_test,Y_test,Batchsize,Timesteps,Dimension):
	model = Sequential()
	model.add(CuDNNLSTM(units=64,batch_input_shape=(None,Timesteps,Dimension)))
	model.add(Dropout(0.9))
	#model.add(Dense(64))
	#model.add(Dropout(0.9))
	model.add(Dense(19))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])#adam or nadam
	model.summary()
	checkpointer = ModelCheckpoint(filepath="./savemodel/modelensumble2"+"_{epoch:003d}.hdf5", verbose=1, save_best_only=True,mode='max',monitor='val_acc')
	his = model.fit_generator(DataSequence(X_data,Y_data,Batchsize),steps_per_epoch=8000,epochs=100,validation_data=DataSequence(X_test,Y_test,Batchsize),validation_steps=2717,callbacks=[checkpointer])
def testmodel(modelname):
	trainfeature,testfeature,trainans,testans = predata()
	X_data = np.array(trainfeature)
	print(X_data.shape)
	X_test = np.array(testfeature)
	print(X_test.shape)
	model = load_model(modelname)
	modeltxt = './savemodel/result_3.txt'
	savetxt = open(modeltxt,'w')
	savestr = ""
	Y_pred = model.predict_classes(X_test)
	#print(Y_pred)
	for i in range(0,2717):
		savestr += str((8001+i))+"\t"+dictt[Y_pred[i]]+"\n"

	savetxt.write(savestr)
	savetxt.close()

def traindata():
	trainfeature,testfeature,trainans,testans = predata()
	X_data = np.array(trainfeature)
	print(X_data.shape)
	X_test = np.array(testfeature)
	print(X_test.shape)
	Y_data = keras.utils.to_categorical(trainans,num_classes=19)
	Y_test = keras.utils.to_categorical(testans,num_classes=19)
	AngleRNN1(X_data,Y_data,X_test,Y_test,8,2,19)
	AngleRNN(X_data,Y_data,X_test,Y_test,8,2,19)
	AngleRNN2(X_data,Y_data,X_test,Y_test,8,2,19)
def main():
	#traindata()
	testmodel('./savemodel/modelensumble2_004.hdf5')
if __name__ == "__main__":
	main()
