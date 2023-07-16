import csv


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

dates, codes, names, volumes, closes = parse_dataset('dataset.csv')

obv_dict = {}  # 종목별 OBV 값을 저장할 딕셔너리

for code in set(codes):  # 종목 코드를 기준으로 반복
    index_list = [i for i, x in enumerate(codes) if x == code]  # 종목 코드에 해당하는 인덱스 리스트 추출
    code_volumes = [volumes[i] for i in index_list]  # 종목 코드에 해당하는 거래량 리스트 추출
    code_closes = [closes[i] for i in index_list]  # 종목 코드에 해당하는 종가 리스트 추출
    obv = calculate_obv(code_volumes, code_closes)
    obv_dict[code] = obv[-1]  # 최신 OBV 값을 딕셔너리에 저장
    

# OBV 값을 기준으로 종목을 상승 가능성이 높은 순서로 정렬
sorted_obv = sorted(obv_dict.items(), key=lambda x: x[1], reverse=True)

for rank, item in enumerate(sorted_obv, start=1):
    code = item[0]
    

# 결과를 baseline_submission.csv 파일에 작성
with open('baseline_submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['종목코드', '순위'])

    for rank, item in enumerate(sorted_obv, start=1):
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

# print(data)
# for rank, item in enumerate(problem, start=1):
#     for i in range(len(data)):
#         if item in data[i][0]:
#             data[i][1] = str(rank)

#A056730 처럼 주식 가격이 없는 애는 빼야됨
#거래량도 큰 종목이 상위로 포함되는거니까 어떻게 해야할지 결정