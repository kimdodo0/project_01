import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import urllib.request
from tensorflow.keras.models import load_model



with open('models/text_token.pickle', 'rb') as f:
    src_tokenizer = pickle.load(f)

with open('models/summary_token.pickle', 'rb') as f:
    tar_tokenizer = pickle.load(f)

model = load_model('./models/model_1.6148828268051147.h5')

src_index_to_word = src_tokenizer.index_word # 원문 단어 집합에서 정수 -> 단어를 얻음
tar_word_to_index = tar_tokenizer.word_index # 요약 단어 집합에서 단어 -> 정수를 얻음
tar_index_to_word = tar_tokenizer.index_word # 요약 단어 집합에서 정수 -> 단어를 얻음

text_max_len = 300
summary_max_len = 50



encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])