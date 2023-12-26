import pandas as pd
import json

# CSV 파일을 DataFrame으로 읽기
df = pd.read_csv('new_final_data.csv')

# JSONL 파일로 변환하는 함수 정의
def convert_to_jsonl(row):
    system_message = row['system']
    user_message = row['Q']
    assistant_message = row['A']

    # 각 행을 원하는 형식의 JSON 데이터로 변환
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]

    # JSON 데이터를 JSONL 형식으로 변환하여 반환
    return json.dumps({"messages": messages}, ensure_ascii=False)

# 각 행을 JSONL 형식으로 변환
jsonl_data = df.apply(convert_to_jsonl, axis=1)

# JSONL 파일로 저장
with open('data.jsonl', 'w', encoding='utf-8') as file:
    file.write('\n'.join(jsonl_data))

print("JSONL 파일로 변환 완료")
