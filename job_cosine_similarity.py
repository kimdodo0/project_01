import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

sentences = ["안녕하세요?", "한국어 문장 임베딩을 위한 버트 모델입니다."]
embeddings = encoder.encode(sentences)

df = pd.read_csv('sample.csv')
print(df.columns)

df = df.dropna()

print(df.loc[0, 'Q'])
print(df.loc[0, 'A'])
encoder.encode(df.loc[0, 'Q'])

df['embedding'] = pd.Series([[]] * len(df)) # dummy

df['embedding'] = df['Q'].map(lambda x: list(encoder.encode(x)))

df.head()

text = '요즘 남편이 비트코인도 하고 속을 너무 썩이네'

embedding = encoder.encode(text)

df['similarity'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())

df.head()

answer = df.loc[df['similarity'].idxmax()]

# print('구분', answer['구분'])
print('유사한 질문', answer['Q'])
print('챗봇 답변', answer['A'])
print('유사도', answer['similarity'])