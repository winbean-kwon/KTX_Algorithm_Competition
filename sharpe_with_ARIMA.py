import csv
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

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


#기간 동안의 투자 수익률 계산
def calculate_returns(closes):
    returns = np.diff(closes) / closes[:-1]
    returns = np.concatenate([[0], returns])
    return returns

# ARIMA 모델을 사용하여 미래 수익 예측
def predict_returns(closes, forecast_steps=15):
    model = ARIMA(closes, order=(1,0,0))  # ARIMA(1,0,0) 모델로 가정
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_steps)  # forecast_steps만큼 미래로 예측
    return forecast[-1]  # 가장 마지막 날의 예측 수익을 반환


period = 6

#기술적 지표 계산
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

# 샤프 지수 계산
def calculate_sharpe_ratio(returns, risk_free_rate=0.035):
    excess_returns = returns - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    sharpe_ratio = mean_excess_return / std_excess_return
    return sharpe_ratio

# ARIMA 모델로 미래 수익 예측
def predict_returns(closes, forecast_steps=15):
    model = ARIMA(closes, order=(1,0,0))  # ARIMA(1,0,0) 모델로 가정
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_steps)  # forecast_steps만큼 미래로 예측
    return forecast

dates, codes, names, volumes, opens, highs, lows, closes = parse_dataset()

sharpe_dict = {}

count=1
for code in set(codes):
    index_list = [i for i, x in enumerate(codes) if x == code]
    code_closes = [closes[i] for i in index_list]

    # RSI 계산
    rsi = calculate_rsi(code_closes,period)

    # 기간 동안의 투자 수익률 계산
    returns = calculate_returns(code_closes)

    # 샤프 지수 계산
    sharpe_ratio = calculate_sharpe_ratio(returns)

    # ARIMA 모델을 사용하여 15일 동안의 미래 수익 예측
    forecast_returns = predict_returns(code_closes, forecast_steps=15)

    # 15일 동안의 미래 예측 수익을 기반으로 종목의 스코어 계산
    forecast_mean = np.mean(forecast_returns)
    score = rsi + sharpe_ratio + forecast_mean
    print(score)
    print(code)
    print(count)
    count+=1
    sharpe_dict[code] = score

sorted_sharpe = sorted(sharpe_dict.items(), key=lambda x: x[1], reverse=True)

# 정렬된 샤프지수를 기준으로 종목 랭킹 작성
with open('sharpe_ratio_ranking.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['종목코드', '순위'])

    for rank, (code, _) in enumerate(sorted_sharpe, start=1):
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

#변동성 큰 종목들도 분류해야하는데...