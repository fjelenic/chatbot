import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import time
import json
import pickle

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

#preprocessing data if not already done
try:
    with open("data.pickle", "rb") as f:
        words,labels,training,output=pickle.load(f)
except:
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

    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training,output),f)

tf.reset_default_graph()

#making a NN model
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,9)
net = tflearn.fully_connected(net,9)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#training if not trained before
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

#making a bag of words from input
def bagOfWords(string,words):
    bag=[0 for _ in range(len(words))]

    s_words=nltk.word_tokenize(string)
    s_words=[stemmer.stem(word.lower()) for word in s_words]

    for s in s_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)

#function for chatting
def chat():
    print("Start talking with the bot")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bagOfWords(inp,words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]

            print("Bot: " + random.choice(responses))
        else:
            print("Bot: I'm not yet advanced enough to understand what you mean.")
            time.sleep(2)
            print("Bot: But soon I will...")

chat()

