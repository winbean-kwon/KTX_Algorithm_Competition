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

def calculate_z_score(values):
    mean_value = np.mean(values)
    std_value = np.std(values)
    z_scores = (values - mean_value) / std_value
    return z_scores

#기간 동안의 투자 수익률 계산
def calculate_returns(closes):
    returns = np.diff(closes) / closes[:-1]
    returns = np.concatenate([[0], returns])
    return returns

period = 15

#기술적 지표 계산
def calculate_rsi(prices, period):
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

# 샤프 지수 계산
def calculate_sharpe_ratio(returns, risk_free_rate=0.035):
    alpha = 0.2
    # 샤프 지수를 계산합니다.
    excess_returns = returns - risk_free_rate
    data = np.array(excess_returns)
    weighted_average = np.zeros_like(data)
    weighted_average[0] = data[0]
    for t in range(1, 16):
        weighted_average[t] = (1 - alpha) * data[t] + alpha * weighted_average[t - 1]
    mean_excess_return = np.mean(weighted_average)
    std_excess_return = np.std(weighted_average)
    sharpe_ratio = mean_excess_return / std_excess_return
    return sharpe_ratio

# ARIMA 모델로 미래 수익 예측
def predict_returns(closes, forecast_steps=15):
    model = ARIMA(closes, order=(1,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_steps)  # forecast_steps만큼 미래로 예측
    return forecast

dates, codes, names, volumes, opens, highs, lows, closes = parse_dataset()

sharpe_dict = {}

count=1
z_scores_dict = {}
sorted_codes = []
[sorted_codes.append(code) for code in codes if code not in sorted_codes]

for code in sorted_codes:
    index_list = [i for i, x in enumerate(codes) if x == code][:-15]
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
    score = -rsi + sharpe_ratio + forecast_mean

    z_scores = calculate_z_score([-rsi, forecast_mean, sharpe_ratio])
    z_scores_sum = np.sum(z_scores)
    z_scores_dict[code] = z_scores_sum

    count+=1
    sharpe_dict[code] = score

sorted_sharpe = sorted(z_scores_dict.items(), key=lambda x: x[1])

# 정렬된 샤프지수를 기준으로 종목 랭킹 작성
with open('baseline_submission.csv', 'w', newline='') as file:
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

for code in sorted(problem):
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
sorted_codes = []
[sorted_codes.append(code) for code in codes if code not in sorted_codes]
 

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