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
sorted_codes = []
[sorted_codes.append(code) for code in codes if code not in sorted_codes]

for code in sorted_codes:
    index_list = [i for i, x in enumerate(codes) if x == code][:-15]
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

# 분류된 주식에 따라 마지막 15일 데이터로 샤프지수 계산
returns = {}
final_buy_returns = 0
final_sell_returns = 0

for code in sorted_codes:
    rank = sorted_data.get(code)

    if 1 <= rank <= 200 or 1801 <= rank <= 2000:
    
        # 마지막 15일 동안의 주식 가격 데이터 가져오기
        index_list = [i for i, x in enumerate(codes) if x == code][-16:]
        code_closes = [closes[i] for i in index_list]

        # 최종 매수, 공매도 수익률
        if rank <= 200:
            final_buy_returns += ((code_closes[-1]-code_closes[1]) / code_closes[1])
   
        if rank >= 1801:
            final_sell_returns += ((code_closes[-1]-code_closes[1]) / code_closes[1]) * -1

        # 일간 매수 수익률
        if rank >= 1801:
            daily_returns = [((code_closes[i] - code_closes[i - 1]) / code_closes[i - 1]) * -1 for i in range(1, 16)]
        else:
            daily_returns = [(code_closes[i] - code_closes[i - 1]) / code_closes[i - 1] for i in range(1, 16)]

        returns[code] = daily_returns

n_values = range(2, 16)  # n=2부터 n=15까지

# 각 n에 해당하는 일간 수익률을 저장할 리스트 초기화
avg_n_day_returns = [0] * len(n_values)

# 종목별로 2일부터 15일까지의 일간 수익률을 더해줍니다.
for code, daily_returns in returns.items():
    for i, n in enumerate(n_values):
        if len(daily_returns) >= n:
            avg_n_day_returns[i] += daily_returns[n-1]

# 종목별로 2일부터 15일까지의 일간 수익률 평균을 구합니다.
total_num_codes = len(returns)
avg_n_day_returns = [total_return / 400 for total_return in avg_n_day_returns]

# 전체 종목에 대한 일간 수익률을 리스트로 변환
all_daily_returns = [daily_returns for daily_returns_list in returns.values() for daily_returns in daily_returns_list]

# 연율화된 일간 수익률의 평균
avg_daily_return = np.mean(all_daily_returns) * 250

# n=2에서 n=15까지의 연율화된 n 번째 매매일의 일간 수익률의 평균을 구하고, 이들의 차이를 제곱하여 합산
sum_diff_squared = 0
        
for i in range(0,14):

    avg_n_day_return = avg_n_day_returns[i] * 250
    diff_squared = (avg_n_day_return - avg_daily_return) ** 2
    sum_diff_squared += diff_squared

# 변동성 계산
volatility = np.sqrt(sum_diff_squared / 13)

# 마지막 15일 동안의 연율화된 총 수익률 계산
avg_total_returns = ((final_buy_returns + final_sell_returns) / 400) * 250 / 15

# 샤프지수 계산
risk_free_rate = 0.035  # 연율화된 무위험 수익률 (3.5%)
sharpe_ratio = (avg_total_returns - risk_free_rate) / volatility

# print(cumulative_returns)
# print(long_returns_sum)
# print(short_returns_sum)
# print(avg_total_returns)
# print(volatility)
print("마지막 15일 동안의 샤프지수:", sharpe_ratio)