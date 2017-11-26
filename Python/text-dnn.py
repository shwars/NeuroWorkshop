from __future__ import print_function
import numpy as np
import os
import sys
import cntk as C
from cntk import Trainer, Axis
from cntk.learners import momentum_sgd, momentum_as_time_constant_schedule, learning_rate_schedule, UnitType
from cntk.ops import sequence
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.ops.functions import load_model
from cntk.layers import LSTM, Stabilizer, Recurrence, Dense, For, Sequential
from cntk.logging import log_number_of_parameters, ProgressPrinter
from scipy.ndimage.interpolation import shift

data = open("d:\\work\\NeuroWorkshop\\Data\\texts\\Alice.txt", "r",encoding="utf-8").read()
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

model = Sequential([Dense(3000),Dense(vocab_size)])

z = model(input_text)

ce = cross_entropy_with_softmax(z, output_char)
errs = classification_error(z, output_char)

lr_per_sample = learning_rate_schedule(0.001, UnitType.sample)
momentum_time_constant = momentum_as_time_constant_schedule(1100)
clipping_threshold_per_sample = 5.0
gradient_clipping_with_truncation = True
learner = momentum_sgd(z.parameters, lr_per_sample, momentum_time_constant,
                    gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
                    gradient_clipping_with_truncation=gradient_clipping_with_truncation)
progress_printer = ProgressPrinter(freq=100, tag='Training')
trainer = Trainer(z, (ce, errs), learner, progress_printer)
    
log_number_of_parameters(z)

a,b = get_sample(0)
print("a={}, b={}".format(a.shape,b.shape))

for ep in range(1):
    print("Epoch={}".format(ep))
    m = [True]
    for mb in range(data_size-nchars-1):
        feat,lab = get_sample(mb)
        trainer.train_minibatch({input_text: feat, output_char: lab})
        m=[False]


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

    out = prime_text+"|";

    inp = np.eye(vocab_size,dtype=np.float32)[np.array([char_to_ix[x] for x in prime_text])]

    for _ in range(length):
        # print([ix_to_char[np.argmax(x)] for x in inp])
        o = net.eval(inp)
        ochr = sample_word(o)
        out = out+ix_to_char[ochr]
        inp = np.roll(inp,-1,axis=0)
        inp[-1,:] = np.eye(vocab_size,dtype=np.float32)[ochr]
    return out

sample(z, '', True)
