import utils
import pandas as pd
import numpy as np

from keras.regularizers import l2
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers import Embedding
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import ModelCheckpoint

token_size = 3

data = pd.read_csv('../data/data.csv',skiprows=0)
filtered = data[['REGI','TYPO','VISUAL_SIMILARITY','SOUNDEX_DISTANCE']][(data['EDIT_DISTANCE'] == 1) & (data['IS_TYPO'] == 1) 
                 &( (data['VISUAL_SIMILARITY'] >= 0.8) | (data['SOUNDEX_DISTANCE'] <=1))                                                            ]
filtered = filtered[filtered.TYPO.map(lambda x: x.count('.')) == 2]
filtered = filtered[filtered.REGI.map(lambda x: x.count('.')) == 2]
filtered.reset_index(drop=True,inplace=True)

reg_list = list()
typo_list = list()
for i in range(t.shape[0]):
    reg_list.append(filtered['REGI'][i].split('.')[0])
    typo_list.append(filtered['TYPO'][i].split('.')[0])

in_list, out_list = utils.tokenize(reg_list, typo_list, token_size)

in_vocab = set()
out_vocab = set()
for name in in_list:
    for char in name:
        in_vocab.add(char)
for name in out_list:
    for char in name:
        out_vocab.add(char)
vocab = in_vocab.union(out_vocab)
num_encoder_tokens = len(in_vocab)
num_decoder_tokens = len(out_vocab)
max_encoder_seq_length = max([len(name) for name in in_list])
max_decoder_seq_length = max([len(name) for name in out_list])

table = utils.CharacterTable(vocab)
encoder_input_data = np.zeros(
    (len(in_list), max_encoder_seq_length, len(vocab)),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(out_list), max_decoder_seq_length, len(vocab)),
    dtype='float32')


for i in range(len(in_list)):
    encoder_input_data[i] = table.encode(in_list[i],token_size+2)
    decoder_input_data[i] = table.encode(out_list[i],token_size+2)

RNN = layers.LSTM
HIDDEN_SIZE = 40
BATCH_SIZE = 16
LAYERS = 2
EPOCH = 100

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='min',save_best_only=True)
callbacks_list = [checkpoint]

model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(token_size+2, len(out_vocab))))
model.add(layers.RepeatVector(token_size+2))
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

model.add(layers.TimeDistributed(layers.Dense(len(out_vocab))))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#model.summary()

history = model.fit(encoder_input_data,decoder_input_data,
              batch_size=BATCH_SIZE,
              epochs=EPOCH,
              validation_split = 0.2,
              callbacks = callbacks_list,
              verbose = True)

model.save('typo_model.h5')