# 주어진 데이터 셋에서 age컬럼 상위 20개의 데이터를 구한 다음 
# f1의 결측치를 중앙값으로 채운다.
# 그리고 f4가 ISFJ와 f5가 20 이상인 
# f1의 평균값을 출력하시오!

# - 데이터셋 : basic1.csv 
# - 오른쪽 상단 copy&edit 클릭 -> 예상문제 풀이 시작
# - File -> Editor Type -> Script

import pandas as pd

# 데이터 불러오기
df = pd.read_csv("../input/bigdatacertificationkr/basic1.csv")
# df.head()

df1 = df.sort_values(by=['age'], ascending= False)
df2 = df1.head(20)
df2['f1'] = df2['f1'].fillna(df2['f1'].median())
condition = (df2['f4']=='ISFJ')&(df2['f5']>=20)
print(df2[condition]['f1'].mean())