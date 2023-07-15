import csv

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

            dates.append(date)
            codes.append(code)
            names.append(name)
            volumes.append(volume)
            closes.append(close)

    return dates, codes, names, volumes, closes

def calculate_obv(volumes, closes):
    obv = [0]  # 초기 OBV 값을 0으로 설정

    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv.append(obv[i-1] + volumes[i])
        elif closes[i] < closes[i-1]:
            obv.append(obv[i-1] - volumes[i])
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

# 결과를 baseline_submission.csv 파일에 작성
with open('baseline_submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['종목코드', '순위'])

    for rank, item in enumerate(sorted_obv, start=1):
        code = item[0]
        writer.writerow([code, rank])
