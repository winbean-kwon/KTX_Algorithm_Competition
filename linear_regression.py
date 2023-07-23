import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

# Feature Engineering: Add additional features if needed
def add_additional_features(df):
    # Example: Calculate moving averages
    df['5-day MA'] = df['closes'].rolling(window=5).mean()
    df['10-day MA'] = df['closes'].rolling(window=10).mean()

    return df

# Calculate daily returns
def calculate_returns(prices):
    return prices.pct_change().dropna()

# Calculate Sharpe ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0.035):
    daily_risk_free_rate = (1 + risk_free_rate) ** (1 / 250) - 1
    excess_returns = returns - daily_risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    
    if std_excess_return == 0:
        sharpe_ratio = 0  # 0으로 나누기 오류를 방지하기 위해 예외 처리
    else:
        sharpe_ratio = mean_excess_return / std_excess_return
    
    return sharpe_ratio

dates, codes, names, volumes, starts, high_price, low_price, closes = parse_dataset()

sharpe_dict = {}
for code in set(codes):
    index_list = [i for i, x in enumerate(codes) if x == code]
    code_closes = [closes[i] for i in index_list]
    # 종가를 기반으로 수익률 계산
    returns = np.diff(code_closes) / code_closes[:-1]
    returns = np.concatenate([[0], returns])  # returns 배열의 길이를 원래 데이터와 동일하게 만듭니다.
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.035)
    sharpe_dict[code] = sharpe

sorted_sharpe = sorted(sharpe_dict.items(), key=lambda x: x[1], reverse=True)

for rank, item in enumerate(sorted_sharpe, start=1):
    code = item[0]

with open('baseline_submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['종목코드', '순위'])

    for rank, item in enumerate(sorted_sharpe, start=1):
        code = item[0]
        writer.writerow([code, rank])

#이상한 종목들 중 상위200 하위 200에 들어가는 종목들 분류

# ... (이전 코드와 sorted_sharpe, adjusted_submission.csv 등은 그대로 사용)
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

# 선형 회귀 모델 구현
X = np.array(list(sorted_data.values())).reshape(-1, 1)  # 랭크 값을 피처로 사용
y = np.array(list(sharpe_dict.values()))  # 샤프 지수를 타겟 변수로 사용
model = LinearRegression()
model.fit(X, y)
print(model.fit(X,y))


# 예측 결과를 저장할 딕셔너리
prediction_dict = {}
for code in set(codes):
    rank = sorted_data[code]
    prediction = model.predict([[rank]])[0]  # 선형 회귀 모델로 샤프 지수를 예측
    prediction_dict[code] = prediction

# 예측 결과를 정렬하여 최종 결과 생성
sorted_prediction = sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True)

with open("final_submission.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['종목코드', '순위'])

    for i, (code, _) in enumerate(sorted_prediction, start=1):
        writer.writerow([code, i])







