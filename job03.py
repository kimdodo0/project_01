import pandas as pd
from konlpy.tag import Okt
import re



df = pd.read_csv('./datasets/data_20231222.csv')

df.info()

okt =Okt()

df_stopwords = pd.read_csv('stopwords.csv')
stopwords = list(df_stopwords['stopword'])

cleaned_sentences = []

count = 0

for word in df.word:
    count +=1
    if count % 100 ==0:
        print('.', end='')
    if count%1000 ==0:
        print('')
    if count%10000 == 0:
        print(count/ 1000)
    word = re.sub('[^가-힣|a-z|A-Z|.|?|!]', '.', word)
    try:
        tokened_word = okt.pos(word, stem=True)
        df_token = pd.DataFrame(tokened_word, columns=['word', 'class'])

    except:
        df_token = ' '
    words = []
    if len(word) > 1:
        if word not in stopwords:
            words.append(word)
    cleaned_sentence = ' '.join(words)
    while '  ' in cleaned_sentence:
        cleaned_sentence = re.sub('  ', ' ', cleaned_sentence)
    cleaned_sentences.append(cleaned_sentence)
df['Text'] = cleaned_sentences
df = df[['Text', 'summary']]
print(df.head(10))
df.info()

df.to_csv('./datasets/cleaned_word.csv', index=False)