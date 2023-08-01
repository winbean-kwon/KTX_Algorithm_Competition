import pandas as pd
import numpy as np
import csv

problem = []

def parse_dataset():
    dates = []
    codes = []
    names = []
    volumes = []
    starts = []
    high_price = []
    low_price = []
    closes = []

    with open('train_month.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for row in csv_reader:
            date = row[0]
            code = row[1]
            name = row[2]
            volume = int(row[3])
            start = int(row[4])
            high = int(row[5])
            low = int(row[6])
            close = int(row[7])

            if volume == 0:
                problem.append(code)
            dates.append(date)
            codes.append(code)
            names.append(name)
            volumes.append(volume)
            starts.append(start)
            high_price.append(high)
            low_price.append(low)
            closes.append(close)

    return dates, codes, names, volumes, starts, high_price, low_price, closes

dates, codes, names, volumes, starts, high_price, low_price, closes = parse_dataset()

volatility = {}

for code in set(codes):
    index_list = [i for i, x in enumerate(codes) if x == code]
    code_closes = [closes[i] for i in index_list]
    print(code_closes)

df = pd.DataFrame(code_closes)
print(df)
# 변동성 계산
returns = df[code_closes].pct_change()  # 일일 수익률 계산
volatility = np.sqrt(12) * returns.std()  # 252는 주식 거래일 수 (1년 기준)

print(volatility)


