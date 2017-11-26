from __future__ import print_function
import numpy as np
import os
import sys
from cntk import Trainer
import cntk as C
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk.ops import sequence
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.layers import LSTM, Stabilizer, Recurrence, Dense, For, Sequential
from cntk.logging import log_number_of_parameters, ProgressPrinter

def read(fn):
    f = open("d:\\work\\NeuroWorkshop\\Data\\Sentiment\\"+fn).readlines()
    return list(map(lambda s: s.split(",")[0],f)),list(map(lambda s: int(s.split(",")[1].strip()),f))

words, labels = read("sentiment-train.txt")

def char_to_num(c):
    if (c>='a' and c<='z'): return ord(c)-ord('a')
    else: return 0;

def num_to_char(n):
    return chr(ord('a')+n)

input_size = max(map(len,words))
vocab_size = char_to_num('z')+1

def to_onehot(n):
    return np.eye(vocab_size,dtype=np.float32)[n]

def fill(l):
    if (len(l)<input_size): return [0]*(input_size-len(l))+l
    else: return l

words_arr = [(to_onehot(fill(list(map(char_to_num,list(w)))))) for w in words]
labels_arr = [np.array([0,1],dtype=np.float32) if x==-1 else np.array([1,0],dtype=np.float32) for x in labels]

input_var = C.input_variable((input_size,vocab_size))
label_var = C.input_variable((2))

model = Sequential([Dense(500,activation=C.ops.relu),Dense(2,activation=None)])

z = model(input_var)

ce = cross_entropy_with_softmax(z, label_var)
errs = classification_error(z, label_var)

lr_per_sample = learning_rate_schedule(0.02, UnitType.minibatch)
learner = C.learners.sgd(z.parameters, lr_per_sample)
progress_printer = ProgressPrinter(freq=100, tag='Training')
trainer = Trainer(z, (ce, errs), learner, progress_printer)

    
log_number_of_parameters(z)

minibatch_size=10

for ep in range(20):
    print("Epoch={}".format(ep))
    for mb in range(0,len(words),minibatch_size):
        trainer.train_minibatch({input_var: words_arr[mb:mb+minibatch_size], label_var: labels_arr[mb:mb+minibatch_size]})

words_test, labels_test = read("sentiment-test.txt")

def check(net,dofill=True):
    total = 0
    correct = 0
    for w,l in zip(words_test,labels_test):
        if dofill: w_a = to_onehot(fill(list(map(char_to_num,list(w)))))
        else: w_a = to_onehot(list(map(char_to_num,list(w))))
        out = net(w_a)[0]
        if (out[1]>0.5 and l==-1): correct+=1
        if (out[0]>0.5 and l==1): correct+=1
        total+=1
    print("{} out of {} correct ({}%)".format(correct,total,correct/total*100))

z_sm = C.softmax(z)
check(z_sm)

# Now implement simple RNN
words_arr1 = [to_onehot(list(map(char_to_num,list(w)))) for w in words]

input_var = sequence.input_variable(vocab_size)
label_var = C.input_variable(2)

model = Sequential([Recurrence(C.layers.RNNStep(200,activation=C.relu)),sequence.last,Dense(100,activation=C.relu),Dense(2)])

z = model(input_var)
z_sm = C.softmax(z)

ce = cross_entropy_with_softmax(z, label_var)
errs = classification_error(z, label_var)

lr_per_sample = learning_rate_schedule(0.02, UnitType.minibatch)
learner = C.learners.sgd(z.parameters, lr_per_sample)
progress_printer = ProgressPrinter(freq=100, tag='Training')
trainer = Trainer(z, (ce, errs), learner, progress_printer)

log_number_of_parameters(z)

minibatch_size = 10

for ep in range(20):
    print("Epoch={}".format(ep))
    for mb in range(0, len(words), minibatch_size):
        trainer.train_minibatch(
            {input_var: words_arr1[mb:mb + minibatch_size], label_var: labels_arr[mb:mb + minibatch_size]})

check(z_sm,False)