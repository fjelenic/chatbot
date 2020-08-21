import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

words=[]
labels=[]
docs_x=[]
docs_y=[]

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        words_temp=nltk.word_tokenize(pattern)
        words.extend(words_temp)
        docs_x.append(words_temp)
        docs_y.append(intent["tag"])
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w not in "?.!,"]
words = sorted(list(set(words)))

labels.sort()

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag=[]

    words_temp = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in words_temp:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])]=1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,9)
net = tflearn.fully_connected(net,9)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)
model.save("model.tflearn")

