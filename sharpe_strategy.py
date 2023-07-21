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

def calculate_sharpe_ratio(returns, risk_free_rate=0.035):
    # 샤프 지수를 계산합니다.
    excess_returns = returns - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    sharpe_ratio = mean_excess_return / std_excess_return
    
    return sharpe_ratio

dates, codes, names, volumes, closes = parse_dataset()

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