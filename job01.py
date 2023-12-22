import pandas as pd
from konlpy.tag import Okt
import re
import datetime
import glob



data_path = glob.glob('./datasets/*.json')
print(data_path)

for path in data_path:
    print(path)
    count = 0
    df_temp = pd.read_json(path)
    df_temp.info()
    words = []
    summaries = []
    for data in df_temp['documents']:
        summary = data['abstractive']
        data = data['text']
        text_data = []
        for i in range(len(data)):
            for j in range(len(data[i])):
                text_data.append(data[i][j]['sentence'])
        print(text_data)
        print(summary)
        count += 1
        words.append(text_data)
        # word = '.'.join(talk_data)
        # word = re.sub('[^가-힣|a-z|A-Z|.]', ' ', word)
        # words.append(word)
        summaries.append(summary)
        if count % 1000 == 0:
            print('.', end='')
        # if count % 10000 ==0:
        #     df = pd.DataFrame({'word':words, 'summary':summaries})
        #     df.to_csv('./datasets/words_{}_{}.csv'.format(path.split('\\')[1][:-5],int(count/10000)))
    df = pd.DataFrame({'word': words, 'summary': summaries})
    df.to_csv('./datasets/words_{}.csv'.format(path.split('\\')[1][:-5]), index=False)
    print(df)
