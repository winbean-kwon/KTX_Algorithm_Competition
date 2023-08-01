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

seed_everything(42)  # Seed 고정

train = pd.read_csv('./train_month.csv')


def detect_outliers_zscore(data, threshold=2):
    z_scores = (data - data.mean()) / data.std()
    outliers = data[abs(z_scores) > threshold]
    return outliers


# 추론 결과를 저장하기 위한 dataframe 생성
results_df = pd.DataFrame(columns=['종목코드', 'final_return'])

# train 데이터에 존재하는 독립적인 종목코드 추출
unique_codes = train['종목코드'].unique()

# A060310 종목 같은 경우엔 1차 차분 하자마자 시계열이 정상상태를 보이는 것으로 판단되어 d 값에 1대입
train_close = train[train['종목코드'] == 'A060310'][['일자', '종가']]
print(train_close)
train_close['일자'] = pd.to_datetime(train_close['일자'], format='%Y%m%d')
train_close.set_index('일자', inplace=True)
tc = train_close['종가']
daily_changes = tc.diff()
daily_changes_without_first = daily_changes[1:]
# plot_acf(daily_changes_without_first)
# plot_pacf(daily_changes_without_first)

outliers = detect_outliers_zscore(daily_changes_without_first, threshold=2)

cleaned_data = daily_changes_without_first[~daily_changes_without_first.isin(outliers)]
plt.plot(cleaned_data)
print(type(cleaned_data))
window = 15
moving_average = cleaned_data.rolling(window).mean()
print(moving_average)

cleaned_data[window-1:] = cleaned_data[window-1:].fillna(moving_average[window-1:])
plt.plot(cleaned_data)
plt.show()

diff_1 = cleaned_data.diff().dropna()

diff_2 = diff_1.diff().dropna()

model = ARIMA(cleaned_data, order=(0, 1, 0))
model_fit = model.fit()
print(model_fit.summary())

predictions = model_fit.forecast(steps=15)  # 향후 15개의 거래일에 대해서 예측
print(predictions)

# 최종 수익률 계산
final_return = (predictions.iloc[-1] - predictions.iloc[0]) / predictions.iloc[0]
