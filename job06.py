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



encoder_input_train,decoder_input_train,decoder_target_train,encoder_input_test, decoder_input_test, decoder_target_test = np.load(
    './models/train_data.npy', allow_pickle=True)

print(encoder_input_train.shape, decoder_input_train.shape, decoder_target_train.shape)
print(encoder_input_test.shape, decoder_input_test.shape, decoder_target_test.shape)

embedding_dim = 128
hidden_size = 256
text_max_len = 50
summary_max_len = 20
src_vocab = 50000
tar_vocab = 16000

# 인코더
encoder_inputs = Input(shape=(text_max_len,))

# 인코더의 임베딩 층
enc_emb = Embedding(src_vocab, embedding_dim)(encoder_inputs)

# 인코더의 LSTM 1
encoder_lstm1 = LSTM(hidden_size, return_sequences=True, return_state=True ,dropout = 0.4, recurrent_dropout = 0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

# 인코더의 LSTM 2
encoder_lstm2 = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

# 인코더의 LSTM 3
encoder_lstm3 = LSTM(hidden_size, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

# 디코더
decoder_inputs = Input(shape=(None,))

# 디코더의 임베딩 층
dec_emb_layer = Embedding(tar_vocab, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)

# 디코더의 LSTM
decoder_lstm = LSTM(hidden_size, return_sequences = True, return_state = True, dropout = 0.4, recurrent_dropout=0.2)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state = [state_h, state_c])

# 디코더의 출력층
decoder_softmax_layer = Dense(tar_vocab, activation = 'softmax')
decoder_softmax_outputs = decoder_softmax_layer(decoder_outputs)

# 모델 정의
model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
model.summary()

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/20.%20Text%20Summarization%20with%20Attention/attention.py", filename="attention.py")

from attention import AttentionLayer

# 어텐션 층(어텐션 함수)
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# 어텐션의 결과와 디코더의 hidden state들을 연결
decoder_concat_input = Concatenate(axis = -1, name='concat_layer')([decoder_outputs, attn_out])

# 디코더의 출력층
decoder_softmax_layer = Dense(tar_vocab, activation='softmax')
decoder_softmax_outputs = decoder_softmax_layer(decoder_concat_input)

# 모델 정의
model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 2)
history = model.fit(x = [encoder_input_train, decoder_input_train], y = decoder_target_train, \
          validation_data = ([encoder_input_test, decoder_input_test], decoder_target_test),
          batch_size = 256, callbacks=[es], epochs = 50)

model.save('./models/model_{}.h5'.format(history.history['val_loss'][-1]))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
#
# with open('models/text_token.pickle', 'rb') as f:
#     src_tokenizer = pickle.load(f)
#
# with open('models/summary_token.pickle', 'rb') as f:
#     tar_tokenizer = pickle.load(f)
#
# model = load_model('./models/model_1.6148828268051147.h5', custom_objects={'AttentionLayer':attn_layer})
#
# src_index_to_word = src_tokenizer.index_word # 원문 단어 집합에서 정수 -> 단어를 얻음
# tar_word_to_index = tar_tokenizer.word_index # 요약 단어 집합에서 단어 -> 정수를 얻음
# tar_index_to_word = tar_tokenizer.index_word # 요약 단어 집합에서 정수 -> 단어를 얻음
#
# encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])
#
# # 이전 시점의 상태들을 저장하는 텐서
# decoder_state_input_h = Input(shape=(hidden_size,))
# decoder_state_input_c = Input(shape=(hidden_size,))
#
# dec_emb2 = dec_emb_layer(decoder_inputs)
# # 문장의 다음 단어를 예측하기 위해서 초기 상태(initial_state)를 이전 시점의 상태로 사용. 이는 뒤의 함수 decode_sequence()에 구현
# # 훈련 과정에서와 달리 LSTM의 리턴하는 은닉 상태와 셀 상태인 state_h와 state_c를 버리지 않음.
# decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])
#
# # 어텐션 함수
# decoder_hidden_state_input = Input(shape=(text_max_len, hidden_size))
# attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
# decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])
#
# # 디코더의 출력층
# decoder_outputs2 = decoder_softmax_layer(decoder_inf_concat)
#
# # 최종 디코더 모델
# decoder_model = Model(
#     [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
#     [decoder_outputs2] + [state_h2, state_c2])
#
# def decode_sequence(input_seq):
#     # 입력으로부터 인코더의 상태를 얻음
#     e_out, e_h, e_c = encoder_model.predict(input_seq)
#
#      # <SOS>에 해당하는 토큰 생성
#     target_seq = np.zeros((1,1))
#     target_seq[0, 0] = tar_word_to_index['sostoken']
#
#     stop_condition = False
#     decoded_sentence = ''
#     while not stop_condition: # stop_condition이 True가 될 때까지 루프 반복
#
#         output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
#         sampled_token_index = np.argmax(output_tokens[0, -1, :])
#         sampled_token = tar_index_to_word[sampled_token_index]
#
#         if(sampled_token!='eostoken'):
#             decoded_sentence += ' '+sampled_token
#
#         #  <eos>에 도달하거나 최대 길이를 넘으면 중단.
#         if (sampled_token == 'eostoken'  or len(decoded_sentence.split()) >= (summary_max_len-1)):
#             stop_condition = True
#
#         # 길이가 1인 타겟 시퀀스를 업데이트
#         target_seq = np.zeros((1,1))
#         target_seq[0, 0] = sampled_token_index
#
#         # 상태를 업데이트 합니다.
#         e_h, e_c = h, c
#
#     return decoded_sentence
#
# # 원문의 정수 시퀀스를 텍스트 시퀀스로 변환
# def seq2text(input_seq):
#     sentence=''
#     for i in input_seq:
#         if(i!=0):
#             sentence = sentence + src_index_to_word[i]+' '
#     return sentence
#
# # 요약문의 정수 시퀀스를 텍스트 시퀀스로 변환
# def seq2summary(input_seq):
#     sentence=''
#     for i in input_seq:
#         if((i!=0 and i!=tar_word_to_index['sostoken']) and i!=tar_word_to_index['eostoken']):
#             sentence = sentence + tar_index_to_word[i] + ' '
#     return sentence
#
# for i in range(500, 1000):
#     print("원문 : ",seq2text(encoder_input_test[i]))
#     print("실제 요약문 :",seq2summary(decoder_input_test[i]))
#     print("예측 요약문 :",decode_sequence(encoder_input_test[i].reshape(1, text_max_len)))
#     print("\n")