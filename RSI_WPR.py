import csv
import numpy as np
import pandas as pd

problem = []

def parse_dataset():
    dates = []
    codes = []
    names = []
    volumes = []
    closes = []

    with open('train.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for row in csv_reader:
            date = row[0]
            code = row[1]
            name = row[2]
            volume = int(row[3])
            close = int(row[7])

            if volume == 0:
                problem.append(code)
            dates.append(date)
            codes.append(code)
            names.append(name)
            volumes.append(volume)
            closes.append(close)

    return dates, codes, names, volumes, closes

def calculate_rsi(prices, period):
    # 가격 데이터를 기반으로 RSI 계산
    # 가격 데이터를 기반으로 RSI 계산
    alpha = 0.2
    changes = []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        changes.append(change)
    data = np.array(changes)
    weighted_average = np.zeros_like(data)
    weighted_average[0] = data[0]
    for t in range(1, len(data)):
        weighted_average[t] = (1 - alpha) * data[t] + alpha * weighted_average[t - 1]
    
    gains = [change for change in weighted_average if change >= 0]
    losses = [-change for change in weighted_average if change < 0]
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    for i in range(period, len(prices)):
        change = weighted_average[i-1]
        if change >= 0:
            avg_gain = (avg_gain * (period - 1) + change) / period
            avg_loss = (avg_loss * (period - 1)) / period
        else:
            avg_gain = (avg_gain * (period - 1)) / period
            avg_loss = (avg_loss * (period - 1) - change) / period
    
    if avg_loss != 0:
        rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
    else:
        rsi = 100
    
    return rsi

def calculate_wpr(prices, period):
    # 종가를 기준으로 WPR 계산
    highest_high = np.max(prices[-period:])
    lowest_low = np.min(prices[-period:])
    wpr = (highest_high - prices[-1]) / (highest_high - lowest_low) * -100

    return wpr

dates, codes, names, volumes, closes = parse_dataset()

rsi_dict = {}
wpr_dict = {}
period = 6

for code in set(codes):
    index_list = [i for i, x in enumerate(codes) if x == code]
    code_closes = [closes[i] for i in index_list]
    rsi = calculate_rsi(code_closes, period)
    rsi_dict[code] = rsi
    wpr = calculate_wpr(code_closes, period)  # 기간은 14일로 설정합니다.
    wpr_dict[code] = wpr


sorted_rsi = sorted(rsi_dict.items(), key=lambda x: x[1], reverse=True)
sorted_wpr = sorted(wpr_dict.items(), key=lambda x: x[1], reverse=False)

# sorted_sharpe와 sorted_rsi를 활용하여 데이터프레임 생성
df_wpr = pd.DataFrame(sorted_wpr, columns=['종목코드', '순위_wpr'])
df_rsi = pd.DataFrame(sorted_rsi, columns=['종목코드', '순위_rsi'])
print(df_rsi.head(200))
print(df_rsi.tail(200))

# 두 데이터프레임을 '종목코드'를 기준으로 합치기
merged_data = pd.merge(df_wpr, df_rsi, on='종목코드')

# print(merged_data.head(200))
# print(merged_data.tail(200))


# 상관관계 계산하기
correlation = merged_data['순위_wpr'].corr(merged_data['순위_rsi'])

print("RSI와 WPR 전략의 상관관계:", correlation)

#Combine the rankings
combined_ranking = {}
for rank, (code, _) in enumerate(sorted_rsi):
    combined_ranking[code] = rank

for rank, (code, _) in enumerate(sorted_wpr):
    combined_ranking[code] += rank

# Sort the stocks based on the combined ranking
sorted_combined = sorted(combined_ranking.items(), key=lambda x: x[1])



with open('baseline_submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['종목코드', '순위'])

    for rank, item in enumerate(sorted_rsi, start=1):
        code = item[0]
        writer.writerow([code, rank])


#이상한 종목들 중 상위200 하위 200에 들어가는 종목들 분류

data = {}
with open('baseline_submission.csv', newline='') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        code = row[0]
        rank = int(row[1])
        data[code] = rank
    
i=1
s=1
print(set(problem))

for code in set(problem):
    if code in data:
        rank = data[code]
        test1 = 200+i
        test2 = 1800-s
        before = next(key for key, value in data.items() if value == rank)
        if rank <= 200:
            after = next(key for key, value in data.items() if value == test1)
            data[before], data[after] = data[after], data[before]
            i += 1
            
        elif rank >= 1800:
            after = next(key for key, value in data.items() if value == test2)
            data[before], data[after] = data[after], data[before]
            s += 1
            
sorted_data = dict(sorted(data.items(), key=lambda x: x[1]))

for i, (code, rank) in enumerate(sorted_data.items(), start=1):
    sorted_data[code] = i

with open("adjusted_submission.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['종목코드', '순위'])

    for code, rank in sorted_data.items():
        writer.writerow([code, rank])