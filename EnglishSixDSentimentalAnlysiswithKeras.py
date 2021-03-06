# -*- coding: utf-8 -*-
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
import pickle
from keras.models import load_model


## EDA 
maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
sentences = []
with open('./Jan9-2012-tweets-clean.txt','r+',encoding='UTF-8') as f:
    for line in f:
        chunks = line.strip().split("\t")
        if len(chunks) == 3:
            userID = chunks[0]
            sentence = chunks[1]
            labelStr = chunks[2][3:]
            sentences.append(sentence)
            words = nltk.word_tokenize(sentence.lower())
            if len(words) > maxlen:
                maxlen = len(words)
            for word in words:
                word_freqs[word] += 1
            num_recs += 1
print('max_len ',maxlen)
print('nb_words ', len(word_freqs))


# 准备数据
MAX_FEATURES = 20000
MAX_SENTENCE_LENGTH = 60
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}
X = np.empty(num_recs,dtype=list)
# y = np.zeros(num_recs)
y = []
i=0
with open('./Jan9-2012-tweets-clean.txt','r+',encoding='UTF-8') as f:
    for line in f:
        chunks = line.strip().split("\t")
        if len(chunks) == 3:
            userID = chunks[0]
            sentence = chunks[1]
            labelStr = chunks[2][3:]
            label = -1
            if labelStr == "joy":
                label = 0
            elif labelStr == "sadness":
                label = 1
            elif labelStr == "surprise":
                label = 2
            elif labelStr == "disgust":
                label = 3
            elif labelStr == "fear":
                label = 4
            elif labelStr == "anger":
                label = 5
            label = np_utils.to_categorical(label,6)
            # print(label)
            words = nltk.word_tokenize(sentence.lower())
            seqs = []
            for word in words:
                if word in word2index:
                    seqs.append(word2index[word])
                else:
                    seqs.append(word2index["UNK"])
            X[i] = seqs
            # y[i] = label
            y.append(label)
            i += 1
y = np.array(y)
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
## 数据划分
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=42)
## 网络构建
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 200
NUM_EPOCHS = 20
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(6))
model.add(Activation("sigmoid"))
adam = optimizers.Adam(lr=1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.5, amsgrad=False)
# model.compile(loss='mean_squared_error', optimizer=sgd,metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=["accuracy"])
## 网络训练
model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))
## 预测
score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)

model.save('my_model.h5')

print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('{}   {}      {}'.format('Prediction','True Value','Sentence'))
for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,60)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0]
    sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
    print(' {}      {}     {}'.format(ypred, ylabel, sent))
##### 自己输入
INPUT_SENTENCES = []
inputSentence = '';

XX = np.empty(len(INPUT_SENTENCES),dtype=list)
i=0
for sentence in  INPUT_SENTENCES:
    words = nltk.word_tokenize(sentence.lower())
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
label2word = {5:'anger',4:'fear',3:'disgust',2:'surprise',1:'sadness', 0:'joy'}
for i in range(len(INPUT_SENTENCES)):
    p = labels[i].tolist().index(max(labels[i].tolist()))
    print('{}   {}'.format(label2word[p], INPUT_SENTENCES[i]))