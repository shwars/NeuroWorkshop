from __future__ import print_function
import numpy as np
import cntk
from cntk import Trainer, Axis
from cntk.learners import momentum_sgd, momentum_as_time_constant_schedule, learning_rate_schedule, UnitType
from cntk.ops import sequence
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.layers import LSTM, Stabilizer, Recurrence, Dense, For, Sequential
from cntk.logging import log_number_of_parameters, ProgressPrinter

data = open("d:\\work\\NeuroWorkshop\\Data\\texts\\Alice.txt", "r",encoding="utf-8").read()
data = data[0:len(data)//3].lower()
chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }


minibatch_size=100
def get_sample(p):
    xi = [char_to_ix[ch] for ch in data[p:p+minibatch_size]]
    yi = [char_to_ix[ch] for ch in data[p+1:p+minibatch_size+1]]
    
    X = np.eye(vocab_size, dtype=np.float32)[xi]
    Y = np.eye(vocab_size, dtype=np.float32)[yi]

    return [X], [Y]

get_sample(0)

input_sequence = sequence.input_variable(shape=vocab_size)
label_sequence = sequence.input_variable(shape=vocab_size)

model = Sequential([
        For(range(2), lambda:
                   Sequential([Stabilizer(), Recurrence(LSTM(256), go_backwards=False)])),
        Dense(vocab_size)])

z = model(input_sequence)
z_sm = cntk.softmax(z)

ce = cross_entropy_with_softmax(z, label_sequence)
errs = classification_error(z, label_sequence)

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

def sample(net, prime_text='', use_hardmax=True, length=100, temperature=1.0):

    # Применяем температуру: T < 1 - сглаживание; T=1.0 - без изменений; T > 1 - выделение пиков
    def apply_temp(p):
        p = np.power(p, (temperature))
        # повторно нормализуем
        return (p / np.sum(p))

    def sample_word(p):
        if use_hardmax:
            w = np.argmax(p, axis=2)[0,0]
        else:
            # выбираем случайным образом исходя из вероятностей
            p = np.exp(p) / np.sum(np.exp(p))            
            p = apply_temp(p)
            w = np.random.choice(range(vocab_size), p=p.ravel())
        return w

    plen = 1
    prime = -1

    # инициализируем sequence начальной строкой или случайными значениями
    x = np.zeros((1, vocab_size), dtype=np.float32)    
    if prime_text != '':
        plen = len(prime_text)
        prime = char_to_ix[prime_text[0]]
    else:
        prime = np.random.choice(range(vocab_size))
    x[0, prime] = 1
    arguments = ([x], [True])

    # переменная для хранения результата
    output = []
    output.append(prime)
    
    # обрабатываем начальную строку
    for i in range(plen):            
        p = net.eval(arguments)        
        x = np.zeros((1, vocab_size), dtype=np.float32)
        if i < plen-1:
            idx = char_to_ix[prime_text[i+1]]
        else:
            idx = sample_word(p)

        output.append(idx)
        x[0, idx] = 1            
        arguments = ([x], [False])
    
    # обрабатываем дальнейший текст
    for i in range(length-plen):
        p = net.eval(arguments)
        idx = sample_word(p)
        output.append(idx)
        x = np.zeros((1, vocab_size), dtype=np.float32)
        x[0, idx] = 1
        arguments = ([x], [False])

    # преобразуем к строке и возвращаем
    return ''.join([ix_to_char[c] for c in output])


for ep in range(1):
    print("Epoch={}".format(ep))
    m = [True]
    for mb in range(0,data_size-minibatch_size-1,10):
        feat,lab = get_sample(mb)
        trainer.train_minibatch({input_sequence: feat, label_sequence: lab})
        m=[False]
    print(sample(z_sm,'',True,length=300).replace("\n"," "))


sample(z,'hello',True)
