##### 自己输入
from keras.models import load_model
import numpy as np
import nltk
from keras.preprocessing import sequence
import collections
import os
from lxml import etree#导入lxml库
import jieba

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

model = load_model('my_model.h5')

INPUT_SENTENCES = []
inputSentence = input("Please input a sentence:")
print(inputSentence)
INPUT_SENTENCES.append(inputSentence)

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
labels = [x for x in model.predict(XX)]
label2word = {6:'neutral',5:'Love',4:'Hate',3:'anger',2:'surprise',1:'sadness', 0:'joy'}
for i in range(len(INPUT_SENTENCES)):
	if max(labels[i].tolist()) > 0.5:
		p = labels[i].tolist().index(max(labels[i].tolist()))
	else:
		p = 6
	print('{}   {}		{}'.format(label2word[p], INPUT_SENTENCES[i],labels[i][p]))
