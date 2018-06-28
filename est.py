
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
			keepid = -1
			for i in range(0,19):
				if(tt[0]==dictt[i]):
					keepid = i
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
		ff = f1[i]+f2[i]
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
#print haha 
predata()