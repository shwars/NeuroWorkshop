from __future__ import print_function
import numpy as np
import os
import sys
import cntk as C
from cntk import Trainer, Axis
from cntk.learners import momentum_sgd, momentum_as_time_constant_schedule, learning_rate_schedule, UnitType
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.layers import LSTM, Stabilizer, Recurrence, Dense, For, Sequential
from cntk.logging import log_number_of_parameters, ProgressPrinter

data = open("d:\\work\\NeuroWorkshop\\Data\\texts\\Alice.txt",encoding="utf-8").read()
data = data[0:len(data)//4].lower()
chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

nchars=100
def get_sample(p):
    xi = [char_to_ix[ch] for ch in data[p:p+nchars]]
    yi = [char_to_ix[data[p+1]]]
    
    X = np.eye(vocab_size, dtype=np.float32)[xi]
    Y = np.eye(vocab_size, dtype=np.float32)[yi]

    return X, Y
get_sample(0)


input_text = C.input_variable((nchars,vocab_size))
output_char = C.input_variable(shape=vocab_size)

model = Sequential([Dense(6000,activation=C.relu),Dense(vocab_size,activation=None)])

z = model(input_text)
z_sm = C.softmax(z)

ce = cross_entropy_with_softmax(z, output_char)
errs = classification_error(z, output_char)

lr_per_sample = learning_rate_schedule(0.01, UnitType.minibatch)
momentum_time_constant = momentum_as_time_constant_schedule(1100)
learner = C.learners.adam(z.parameters, lr_per_sample,momentum=momentum_time_constant)
progress_printer = ProgressPrinter(freq=100, tag='Training')
trainer = Trainer(z, (ce, errs), learner, progress_printer)
    
log_number_of_parameters(z)

def sample(net, prime_text='', use_hardmax=True, length=100, temperature=1.0):

    # Применяем температуру: T < 1 - сглаживание; T=1.0 - без изменений; T > 1 - выделение пиков
    def apply_temp(p):
        p = np.power(p, (temperature))
        # повторно нормализуем
        return (p / np.sum(p))

    def sample_word(p):
        if use_hardmax:
            w = np.argmax(p)
        else:
            # выбираем случайным образом исходя из вероятностей
            p = np.exp(p) / np.sum(np.exp(p))            
            p = apply_temp(p)
            w = np.random.choice(range(vocab_size), p=p.ravel())
        return w

    if prime_text=='': prime_text = data[0:nchars]

    if (len(prime_text)<nchars): prime_text = " "*(nchars-len(prime_text))+prime_text

    #out = prime_text+"|";
    out = "";

    inp = np.eye(vocab_size,dtype=np.float32)[np.array([char_to_ix[x] for x in prime_text])]

    for _ in range(length):
        # print([ix_to_char[np.argmax(x)] for x in inp])
        o = net.eval(inp)
        ochr = sample_word(o)
        out = out+ix_to_char[ochr]
        inp = np.roll(inp,-1,axis=0)
        inp[-1,:] = np.eye(vocab_size,dtype=np.float32)[ochr]
    return out



for ep in range(1):
    print("Epoch={}".format(ep))
    for mb in range(0,data_size-nchars-1,28):
        feat,lab = get_sample(mb)
        trainer.train_minibatch({input_text: feat, output_char: lab})
    print(sample(z_sm,'',True,length=300).replace("\n"," "))

print(sample(z_sm,'',False,length=300).replace("\n"," "))
sample(C.softmax(z), 'A quick brown fox jumped over the lazy sleeping dog. While I was reading this text, something happen',length=300,use_hardmax=True)
