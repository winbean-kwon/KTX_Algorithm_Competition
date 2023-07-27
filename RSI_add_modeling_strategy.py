import csv
import numpy as np
import pandas as pd

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

    with open('train.csv', 'r') as file:
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

def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    # MACD 값을 계산합니다.
    short_ema = prices.ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, min_periods=1, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_histogram

dates, codes, names, volumes, starts, high_price, low_price, closes = parse_dataset()

# MACD 생성
macd = calculate_macd(pd.Series(closes))

# RSI와 MACD의 상관관계 계산
rsi_dict = {}
period = 15

for code in set(codes):
    index_list = [i for i, x in enumerate(codes) if x == code][:-15]
    code_closes = [closes[i] for i in index_list]
    rsi = calculate_rsi(code_closes, period)
    rsi_dict[code] = rsi

rsi_values = np.array(list(rsi_dict.values()))
macd_values = np.array(macd.dropna()[-len(rsi_values):])
correlation_macd = np.corrcoef(rsi_values, macd_values)[0, 1]

print("RSI와 MACD의 상관관계:", correlation_macd)

# 새로운 특징을 기준으로 종목들을 순위별로 정렬
combined_dict = {}
for code, rsi_value in rsi_dict.items():
    combined_dict[code] = rsi_value

sorted_combined = sorted(combined_dict.items(), key=lambda x: x[1], reverse=True)

# 결과를 baseline_submission.csv 파일에 작성
with open('baseline_submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['종목코드', '순위'])

    for rank, item in enumerate(sorted_combined, start=1):
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
for code in set(codes):
    rank = sorted_data.get(code)
    
    if rank >= 1800 or rank <= 200:
        # 마지막 15일 동안의 주식 가격 데이터 가져오기
        index_list = [i for i, x in enumerate(codes) if x == code][-15:]
        code_closes = [closes[i] for i in index_list]

        # 마지막 15일 동안의 일간 수익률 계산
        daily_returns = [(code_closes[i]-code_closes[i-1]) / code_closes[i-1] for i in range(1, len(code_closes))]

        returns[code] = daily_returns
        
        # 기간 동안의 평균 일간 수익률 계산
        avg_daily_return = np.mean(daily_returns)*250
        if rank >= 1800:
            avg_daily_return = avg_daily_return * -1

        # n=2에서 n=15까지의 연율화된 n 번째 매매일의 일간 수익률의 평균을 구하고, 이들의 차이를 제곱하여 합산
        n_values = range(2, 16)  # n=2부터 n=15까지
        sum_diff_squared = 0
        
        for n in n_values:
            avg_n_day_return = np.mean(daily_returns[:n])*250
            diff_squared = (avg_n_day_return - avg_daily_return) ** 2
            sum_diff_squared += diff_squared

        # 변동성 계산
        volatility = np.sqrt(sum_diff_squared / 13)
    
# 마지막 15일 동안의 누적 수익률 계산
cumulative_returns = {code: np.prod(np.array(daily_returns) + 1) - 1 for code, daily_returns in returns.items()}

# 마지막 15일 동안의 평균 Long 및 Short 수익률 계산
long_returns_sum = sum(cumulative_returns[code] for code, rank in sorted_data.items() if rank <= 200)
short_returns_sum = sum(cumulative_returns[code] for code, rank in sorted_data.items() if rank >= 1800) * -1

# 마지막 15일 동안의 연율화된 총 수익률 계산
avg_total_returns = ((long_returns_sum + short_returns_sum) / 400) * 250 / 15

# 샤프지수 계산
risk_free_rate = 0.035  # 연율화된 무위험 수익률 (3.5%)
sharpe_ratio = (avg_total_returns - risk_free_rate) / volatility

print("마지막 15일 동안의 샤프지수:", sharpe_ratio)