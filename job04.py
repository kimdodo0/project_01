import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import urllib.request
from konlpy.tag import Okt
from konlpy.tag import Okt



np.random.seed(seed=0)

data = pd.read_csv("datasets/cleaned_word.csv", nrows = 100000)
print('전체 리뷰 개수 :',(len(data)))

print(data.head())

data = data[['Text', 'summary']]
print(data.head())

print('Text 열에서 중복을 배제한 유일한 샘플의 수 :', data['Text'].nunique())
print('Summary 열에서 중복을 배제한 유일한 샘플의 수 :', data['summary'].nunique())

data.drop_duplicates(subset=['Text'], inplace=True)
print("전체 샘플수 :", len(data))

print(data.isnull().sum())

data.dropna(axis=0, inplace=True)
print('전체 샘플수 :',(len(data)))

data.replace('',np.nan, inplace=True)
print(data.isnull().sum())

text_len = [len(s.split()) for s in data['Text']]
summary_len = [len(s.split()) for s in data['summary']]

print('텍스트의 최소 길이 : {}'.format(np.min(text_len)))
print('텍스트의 최대 길이 : {}'.format(np.max(text_len)))
print('텍스트의 평균 길이 : {}'.format(np.mean(text_len)))
print('요약의 최소 길이 : {}'.format(np.min(summary_len)))
print('요약의 최대 길이 : {}'.format(np.max(summary_len)))
print('요약의 평균 길이 : {}'.format(np.mean(summary_len)))

plt.subplot(1,2,1)
plt.boxplot(summary_len)
plt.title('summary')
plt.subplot(1,2,2)
plt.boxplot(text_len)
plt.title('Text')
plt.tight_layout()
plt.show()

plt.title('summary')
plt.hist(summary_len, bins=40)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

plt.title('Text')
plt.hist(text_len, bins=40)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

text_max_len = 70
summary_max_len = 20

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s.split()) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))))

below_threshold_len(text_max_len, data['Text'])
below_threshold_len(summary_max_len, data['summary'])

data = data[data['Text'].apply(lambda x: len(x.split()) <= text_max_len)]
data = data[data['summary'].apply(lambda x: len(x.split()) <= summary_max_len)]
print('전체 샘플수 :',(len(data)))

def remove_list(cell_value):
    return str(cell_value).strip('[]')

data['Summary'] = data['summary'].apply(remove_list)
data = data.drop(columns=['summary'])
print(data.head())
print(data.info())
data['Text'] = data['Text'].replace('..', '.', inplace=True)
#
# okt = Okt()
# X = data['Text']
# Y = data['Summary']
#
# for i in range(len(X)):
#     if i in X.index:
#         X[i] = okt.morphs(X[i], stem=True)
#         print(X[i])
#
# stopwords = pd.read_csv('./stopwords.csv', index_col=0)
# for j in range(len(X)):
#     words = []
#     try:
#         for i in range(len(X[j])):
#             if len(X[j][i]) >1 :
#                 if X[j][i] not in list(stopwords['stopword']):
#                     words.append(X[j][i])
#     except:
#         words.append('')
#     X[j] = ' '.join(words)
#     print(X[j])
#
# for j in range(len(Y)):
#     if j in Y.index:
#         Y[j] = okt.morphs(Y[j], stem=True)
#         print(Y[j])
#
# stopwords = pd.read_csv('./stopwords.csv', index_col=0)
# for j in range(len(Y)):
#     words = []
#     try:
#         for i in range(len(Y[j])):
#             if len(Y[j][i]) >1 :
#                 if Y[j][i] not in list(stopwords['stopword']):
#                     words.append(Y[j][i])
#     except:
#         words.append('')
#     Y[j] = ' '.join(words)
#
# data['Text'] = X
# data['Summary'] = Y

data['decoder_input'] = data['Summary'].apply(lambda x : 'sostoken '+ x)
data['decoder_target'] = data['Summary'].apply(lambda x : x + ' eostoken')
print(data.head())

encoder_input = np.array(data['Text'])
decoder_input = np.array(data['decoder_input'])
decoder_target = np.array(data['decoder_target'])

data.to_csv('./datasets/data.csv', index=False)
