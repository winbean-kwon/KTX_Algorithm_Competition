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
    changes = []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        changes.append(change)
    
    gains = [change for change in changes if change >= 0]
    losses = [-change for change in changes if change < 0]
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    for i in range(period, len(prices)):
        change = changes[i-1]
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

def calculate_sharpe_ratio(returns, risk_free_rate=0.035):
    # 샤프 지수를 계산합니다.
    excess_returns = returns - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    sharpe_ratio = mean_excess_return / std_excess_return
    
    return sharpe_ratio

def calculate_obv(volumes, closes):
    obv = [0]  # 초기 OBV 값을 0으로 설정

    for i in range(1, len(closes)):
        if volumes[i-1] == 0:
            obv.append(obv[i-1])
        else:
            volume_change = (volumes[i] / volumes[i-1] - 1) * 100  # 이전 거래량 대비 현재 거래량의 차이를 백분율로 계산

            if closes[i] > closes[i-1]:
                obv.append(obv[i-1] + volume_change)
            elif closes[i] < closes[i-1]:
                obv.append(obv[i-1] - volume_change)
            else:
                obv.append(obv[i-1])
    return obv

dates, codes, names, volumes, closes = parse_dataset()

sharpe_dict = {}
rsi_dict = {}
obv_dict = {}
period=6

for code in set(codes):
    index_list = [i for i, x in enumerate(codes) if x == code]
    code_closes = [closes[i] for i in index_list]
    # 종가를 기반으로 수익률 계산
    returns = np.diff(code_closes) / code_closes[:-1]
    returns = np.concatenate([[0], returns])  # returns 배열의 길이를 원래 데이터와 동일하게 만듭니다.
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.035)
    sharpe_dict[code] = sharpe

    index_list = [i for i, x in enumerate(codes) if x == code]
    code_closes = [closes[i] for i in index_list]
    rsi = calculate_rsi(code_closes, period)
    rsi_dict[code] = rsi

    code_volumes = [volumes[i] for i in index_list]
    code_closes = [closes[i] for i in index_list]
    obv = calculate_obv(code_volumes, code_closes)
    obv_dict[code] = obv[-1]

sorted_rsi = sorted(rsi_dict.items(), key=lambda x: x[1], reverse=True)
sorted_sharpe = sorted(sharpe_dict.items(), key=lambda x: x[1], reverse=True)
sorted_obv = sorted(obv_dict.items(), key=lambda x: x[1], reverse=True)

# sorted_sharpe와 sorted_rsi를 활용하여 데이터프레임 생성
df_sharpe = pd.DataFrame(sorted_sharpe, columns=['종목코드', '순위_sharpe'])
df_rsi = pd.DataFrame(sorted_rsi, columns=['종목코드', '순위_rsi'])

# 두 데이터프레임을 '종목코드'를 기준으로 합치기
merged_data = pd.merge(df_sharpe, df_rsi, on='종목코드')

# 상관관계 계산하기
correlation = merged_data['순위_sharpe'].corr(merged_data['순위_rsi'])

print("RSI와 샤프지수 전략의 상관관계:", correlation)

#Combine the rankings
combined_ranking = {}
for rank, (code, _) in enumerate(sorted_rsi):
    combined_ranking[code] = rank

for rank, (code, _) in enumerate(sorted_sharpe):
    combined_ranking[code] += rank

for rank, (code, _) in enumerate(sorted_obv):
    combined_ranking[code] += rank

# Sort the stocks based on the combined ranking
sorted_combined = sorted(combined_ranking.items(), key=lambda x: x[1])

with open('baseline_submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['종목코드', '순위'])

    for rank, (code, _) in enumerate(sorted_combined, start=1):
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
        