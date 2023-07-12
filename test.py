import pandas as pd
import numpy as np
import random
import os

from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import warnings
warnings.filterwarnings("ignore")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

#무조건 안전한 전략. 특정 이하의 거래량 종목들은 매매 안하는 순위에 랜덤으로 넣고 순위를 채울것


train = pd.read_csv('./train.csv')

# 추론 결과를 저장하기 위한 dataframe 생성
results_df = pd.DataFrame(columns=['종목코드', 'final_return'])

# train 데이터에 존재하는 독립적인 종목코드 추출
unique_codes = train['종목코드'].unique()
print(unique_codes)

# A060310 종목 같은 경우엔 1차 차분 하자마자 시계열이 정상상태를 보이는 것으로 판단되어 d 값에 1대입
train_close = train[train['종목코드'] == 'A060310'][['일자','종가']]
train_close['일자'] = pd.to_datetime(train_close['일자'], format='%Y%m%d')
train_close.set_index('일자', inplace=True)
tc = train_close['종가']

model = ARIMA(tc, order=(0, 1, 1))
model_fit = model.fit()
print(model_fit.summary())
model = ARIMA(tc, order=(0, 2, 1))
model_fit = model.fit()
print(model_fit.summary())
model = ARIMA(tc, order=(0, 3, 1))
model_fit = model.fit()
print(model_fit.summary())
predictions = model_fit.forecast(steps=15) # 향후 15개의 거래일에 대해서 예측
print(predictions)

# 최종 수익률 계산
final_return = (predictions.iloc[-1] - predictions.iloc[0]) / predictions.iloc[0]

# 각 종목코드에 대해서 모델 학습 및 추론 반복
for code in tqdm(unique_codes):
    
    # 학습 데이터 생성
    train_close = train[train['종목코드'] == code][['일자', '종가']]
    train_close['일자'] = pd.to_datetime(train_close['일자'], format='%Y%m%d')
    train_close.set_index('일자', inplace=True)
    tc = train_close['종가']
    
    # 모델 선언, 학습 및 추론
    model = ARIMA(tc, order=(1, 1, 1))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=15) # 향후 15개의 거래일에 대해서 예측
    
    # 최종 수익률 계산
    final_return = (predictions.iloc[-1] - predictions.iloc[0]) / predictions.iloc[0]
    
    # 결과 저장
    results_df = pd.concat([results_df, pd.DataFrame({'종목코드': [code], 'final_return': [final_return]})], ignore_index=True)



results_df['순위'] = results_df['final_return'].rank(method='first', ascending=False).astype('int') # 각 순위를 중복없이 생성
results_df

sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission

baseline_submission = sample_submission[['종목코드']].merge(results_df[['종목코드', '순위']], on='종목코드', how='left')
baseline_submission

baseline_submission.to_csv('baseline_submission.csv', index=False)