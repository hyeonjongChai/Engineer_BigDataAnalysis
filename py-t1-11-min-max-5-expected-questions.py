# min-max스케일링 기준 상하위 5% 구하기
# 주어진 데이터에서 'f5'컬럼을 min-max 스케일 변환한 후, 상위 5%와 하위 5% 값의 합을 구하시오

# - 데이터셋 : basic1.csv
# - 오른쪽 상단 copy&edit 클릭 -> 예상문제 풀이 시작
# - File -> Editor Type -> Script


# 라이브러리 및 데이터 불러오기
import pandas as pd
import numpy as np

df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['f5'] = scaler.fit_transform(df[['f5']])
print(df['f5'])

lower = df['f5'].quantile(0.05)
upper = df['f5'].quantile(0.95)

print(lower, upper)
print(lower + upper)