import csv
import numpy as np

problem = []

def parse_dataset(filename):
    dates = []
    codes = []
    names = []
    volumes = []
    closes = []


    with open('train.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 헤더 라인 건너뛰기

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

# 새로운 특징 생성
new_features = generate_feature(closes)

# RSI와 20일 이동평균의 상관관계 계산
rsi_values = np.array(new_features)[:, 0]
volume_values = np.array(new_features)[:, 1]
correlation = np.corrcoef(rsi_values, volume_values)[0, 1]

print("RSI와 거래량의 상관관계:", correlation)

# 새로운 특징을 기준으로 종목들을 순위별로 정렬
combined_dict = {}
for code, feature in zip(set(codes), new_features):
    combined_dict[code] = feature

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


# 6개월치 데이터로 돌려보는 것도 방법?
# 다른 지표 찾아보기 MACD?