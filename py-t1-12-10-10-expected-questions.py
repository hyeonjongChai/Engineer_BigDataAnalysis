# 주어진 데이터에서 상위 10개 국가의 접종률 평균과 하위 10개 국가의 접종률 평균을 구하고, 그 차이를 구해보세요 
# (단, 100%가 넘는 접종률 제거, 소수 첫째자리까지 출력)

# - 데이터셋 : ../input/covid-vaccination-vs-death/covid-vaccination-vs-death_ratio.csv
# - 오른쪽 상단 copy&edit 클릭 -> 예상문제 풀이 시작
# - File -> Editor Type -> Script


import pandas as pd

df = pd.read_csv("../input/covid-vaccination-vs-death/covid-vaccination-vs-death_ratio.csv")
# print(df.head())

df1 = df.groupby(['country']).max()
# print(df1)

df1 = df1.sort_values(by = 'ratio')
df1 = df1.drop(df1[df1['ratio']>100].index)
# print(df1)

df2 = df1[['ratio']]

bottom_10 = df2.head(10)
top_10 = df2.tail(10)

# print(bottom_10)
# print(top_10)


print(round(top_10.mean() - bottom_10.mean(),1))
