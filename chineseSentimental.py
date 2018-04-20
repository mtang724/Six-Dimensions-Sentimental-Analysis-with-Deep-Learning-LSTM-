import os
import xml.etree.ElementTree as ET
from lxml import etree#导入lxml库
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras import optimizers
from sklearn.model_selection import train_test_split
import collections
import nltk
import numpy as np
from keras.utils import np_utils
# from gensim.models.word2vec import Word2Vec
# from gensim.corpora.dictionary import Dictionary
import pickle
import jieba
from keras.models import load_model

path = "./CEC_emotionCoprus" #文件夹目录  
files= os.listdir(path) #得到文件夹下的所有文件名称  
s = []  
stoplist = [' ','，','的','。','了','在','是','“','”','和','也','我','他们','我们','人']
## EDA 
maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
sentences = []
for file in files: #遍历文件夹  
	if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开  
		parser=etree.XMLParser()#首先根据dtd得到一个parser(注意dtd文件要放在和xml文件相同的目录)
		print(file)  
		tree = etree.parse(path+"/"+file,parser)#用上面得到的parser将xml解析为树结构
		root = tree.getroot()#获得该树的树根
		for fild in root:
			if(fild.tag == 'paragraph'):
				for paragraph in fild:
					fildName = paragraph.tag
					# if(fildName == 'Joy'):
			  # 			print(fildName,paragraph.text)
					# if(fildName == 'Sorrow'):
					# 	print(fildName,paragraph.text)
					# if(fildName == 'Surprise'):
					# 	print(fildName,paragraph.text)
					# if(fildName == 'Anger'):
					# 	print(fildName,paragraph.text)
					# if(fildName == 'Hate'):
					# 	print(fildName,paragraph.text)
					# if(fildName == 'Hate'):
					# 	print(fildName,paragraph.text)
					if(fildName == 'sentence'):
						sentences.append(paragraph.get('S'))
						# words = nltk.word_tokenize(paragraph.get('S').lower())
						words = jieba.cut(paragraph.get('S'))
						words = [word for word in list(words) if word not in stoplist]
						length = 0
						for word in words:
							word_freqs[word] += 1
							length = length + 1
						if length > maxlen:
							maxlen = length
						num_recs += 1
	  			# print(paragraph.text)
	        
print('max_len ',maxlen)
print('nb_words ', len(word_freqs))


# 准备数据
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 80
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
for i, x in enumerate(word_freqs.most_common(200)):
	print(x)
word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}
X = np.empty(num_recs,dtype=list)
# y = np.zeros(num_recs)
y = []
i = 0
for file in files: #遍历文件夹  
	if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开  
		parser=etree.XMLParser()#首先根据dtd得到一个parser(注意dtd文件要放在和xml文件相同的目录)  
		tree = etree.parse(path+"/"+file,parser)#用上面得到的parser将xml解析为树结构
		root = tree.getroot()#获得该树的树根
		for fild in root:
			if(fild.tag == 'paragraph'):
				for paragraph in fild:
					fildName = paragraph.tag
					if(fildName == 'sentence'):
						# words = nltk.word_tokenize(paragraph.get('S').lower())
						words = jieba.cut(paragraph.get('S'))
			            # print(label)
						words = jieba.cut(paragraph.get('S'))
						words = [word for word in list(words) if word not in stoplist]
						seqs = []
						for word in words:
							if word in word2index:
								seqs.append(word2index[word])
							else:
								seqs.append(word2index["UNK"])
						X[i] = seqs
						i += 1
						label = [0,0,0,0,0,0]
						for emotion in paragraph:
							fildName = emotion.tag
							if(fildName == 'Joy' and float(emotion.text)>0):
								label[0] = float(emotion.text)
							if(fildName == 'Sorrow' and float(emotion.text)>0):
								label[1] = float(emotion.text)
							if(fildName == 'Surprise' and float(emotion.text)>0):
								label[2] = float(emotion.text)
							if(fildName == 'Anger'and float(emotion.text)>0):
								label[3] = float(emotion.text)
							if(fildName == 'Hate' and float(emotion.text)>0):
								label[4] = float(emotion.text)
							if(fildName == 'Love' and float(emotion.text)>0):
								label[5] = float(emotion.text)
						y.append(label)
					
y = np.array(y)
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
## 数据划分
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
# 网络构建
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 500
NUM_EPOCHS = 5000
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(len(label)))
model.add(Activation("sigmoid"))
adam = optimizers.Adam(lr=2, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.5, amsgrad=False)
# model.compile(loss='mean_squared_error', optimizer=sgd,metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=["accuracy"])
## 网络训练
model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))
## 预测
score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
model.save('my_model.h5')
print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('{}   {}      {}'.format('Prediction','Real Value','Sentence'))
for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,MAX_SENTENCE_LENGTH)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0]
    sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
    print(' {}      {}     {}'.format(ypred, ylabel, sent))
##### 自己输入
INPUT_SENTENCES = ['我爱你','我讨厌毕设','我很开心']
XX = np.empty(len(INPUT_SENTENCES),dtype=list)
i=0
for sentence in  INPUT_SENTENCES:
    words = jieba.cut(sentence)
    seq = []
    for word in words:
        if word in word2index:
            seq.append(word2index[word])
        else:
            seq.append(word2index['UNK'])
    XX[i] = seq
    i+=1

XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
labels = [x for x in model.predict(XX) ]
label2word = {5:'Love',4:'Hate',3:'anger',2:'surprise',1:'sadness', 0:'joy'}
for i in range(len(INPUT_SENTENCES)):
    p = labels[i].tolist().index(max(labels[i].tolist()))
    print('{}   {}		{}'.format(label2word[p], INPUT_SENTENCES[i],labels[i][p]))
