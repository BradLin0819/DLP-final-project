import pandas as pd
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='input file')
args = parser.parse_args()

df = pd.read_csv(args.input)
filename_prefix = args.input.split('/')[-1].split('.')[0]
df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d %H:%M:%S")

train_start_date = list(map(int, '2017-01-01'.split('-')))
train_end_date = list(map(int, '2017-10-01'.split('-')))

test_start_date = list(map(int, '2017-10-01'.split('-')))
test_end_date = list(map(int, '2017-11-21'.split('-')))

trade_start_date = list(map(int, '2017-11-21'.split('-')))
trade_end_date = list(map(int, '2018-01-24'.split('-')))

# trade_start_date = list(map(int, '2018-01-01'.split('-')))
# trade_end_date = list(map(int, '2018-01-31'.split('-')))

df[(df["date"].dt.date >= datetime.date(*train_start_date)) & (df["date"].dt.date < datetime.date(*train_end_date))].to_csv(f'{filename_prefix}_train.csv', header=True, index=False)
df[(df["date"].dt.date >= datetime.date(*test_start_date)) & (df["date"].dt.date < datetime.date(*test_end_date))].to_csv(f'{filename_prefix}_val.csv', header=True, index=False)
df[(df["date"].dt.date >= datetime.date(*trade_start_date)) & (df["date"].dt.date < datetime.date(*trade_end_date))].to_csv(f'{filename_prefix}_test.csv', header=True, index=False)
# df[(df["date"].dt.date >= datetime.date(*trade_start_date)) & (df["date"].dt.date <= datetime.date(*trade_end_date))].to_csv(f'{filename_prefix}_trading.csv', header=True, index=False)
