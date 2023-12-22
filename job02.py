import datetime
import pandas as pd
import glob



data_path = glob.glob('datasets/word*.csv')

df = pd.DataFrame()

for path in data_path:
    df_temp = pd.read_csv(path)
    df_temp.dropna(inplace=True)
    df_temp.drop_duplicates(inplace=True)
    df = pd.concat([df, df_temp], ignore_index=True)
df.info()

df.to_csv('./datasets/data_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d')), index=False)
