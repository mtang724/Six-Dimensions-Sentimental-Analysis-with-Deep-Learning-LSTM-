##### 自己输入
from keras.models import load_model
import numpy as np
import nltk
from keras.preprocessing import sequence
import collections

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

model = load_model('my_model.h5')

INPUT_SENTENCES = []
inputSentence = input("Please input a sentence:")
print(inputSentence)
INPUT_SENTENCES.append(inputSentence)

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

XX = sequence.pad_sequences(XX, maxlen=60)
labels = [x for x in model.predict(XX) ]
label2word = {5:'anger',4:'fear',3:'disgust',2:'surprise',1:'sadness', 0:'joy'}
for i in range(len(INPUT_SENTENCES)):
    p = labels[i].tolist().index(max(labels[i].tolist()))
    print('{}   {}'.format(label2word[p], INPUT_SENTENCES[i]))