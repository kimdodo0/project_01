import json
import pandas as pd

# JSON 파일 경로
file_path = './datasets/KAKAO_898_15.json'

Question = []
Answer = []
ID = []
Index = []
index_number=0

# JSON 파일 읽기
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

for info_item in data['info']:
    for line in info_item['annotations']['lines']:
        text = line['norm_text']
        id = line['speaker']['id']
        id_num = line['id']
        Question.append(text)
        # Answer.append()
        ID.append(id)
        Index.append(id_num)
        index_number += 1

df=pd.DataFrame({'TEXT':Question,'ID':ID})
print(df)
# print(Question[index_number-1])
# print(index_number-1)
# print(df.iloc[0]) #데이터프레임의 특정 행 출력
# print(df['Q']) #데이터프레임의 특정 열 출력

df_real=pd.DataFrame({'Q':[],'A':[]})

current_ID = df['ID'].iloc[0]
current_TEXT = df['TEXT'].iloc[0]

for i in range(1, len(df)):
    if df[('ID')].iloc[i] == current_ID:
        current_TEXT += ' ' + df['TEXT'].iloc[i]
    else:
        if i < len(df) -1 and df[('ID')].iloc[i] == df[('ID')].iloc[i+1]:
            df_real = df_real.append({'Q': current_TEXT, 'A': df['TEXT'].iloc[i] + ' ' + df['TEXT'].iloc[i+1]}, ignore_index=True)
        else:
            df_real = df_real.append({'Q': current_TEXT, 'A': df['TEXT'].iloc[i]}, ignore_index=True)
        current_ID = df['ID'].iloc[i]
        current_TEXT = df['TEXT'].iloc[i]

print(df_real)

df_real.to_csv("./final_data/KAKAO_898_15.csv",index=False)