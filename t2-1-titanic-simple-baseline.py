# %% [markdown]
# ## 생존여부 예측모델 만들기
# ### 학습용 데이터 (X_train, y_train)을 이용하여 생존 예측 모형을 만든 후, 이를 평가용 데이터(X_test)에 적용하여 얻은 예측값을 다음과 같은 형식의 CSV파일로 생성하시오(제출한 모델의 성능은 accuracy 평가지표에 따라 채점)
# 
# (가) 제공 데이터 목록
# - y_train: 생존여부(학습용)
# - X_trian, X_test : 승객 정보 (학습용 및 평가용)
# 
# (나) 데이터 형식 및 내용
# - y_trian (712명 데이터)
# 
# **시험환경 세팅은 예시문제와 동일한 형태의 X_train, y_train, X_test 데이터를 만들기 위함임**
# 
# ### 유의사항
# - 성능이 우수한 예측모형을 구축하기 위해서는 적절한 데이터 전처리, 피처엔지니어링, 분류알고리즘, 하이퍼파라미터 튜닝, 모형 앙상블 등이 수반되어야 한다.
# - 수험번호.csv파일이 만들어지도록 코드를 제출한다.
# - 제출한 모델의 성능은 accuracy로 평가함
# 
# csv 출력형태
# 
# ![image.png](attachment:de1920de-121e-47c3-a61f-e905386713bf.png)

# %% [markdown]
# ## [참고]작업형2 문구
# - 출력을 원하실 경우 print() 함수 활용
# - 예시) print(df.head())
# - getcwd(), chdir() 등 작업 폴더 설정 불필요
# - 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가
# 
# ### 데이터 파일 읽기 예제
# - import pandas as pd
# - X_test = pd.read_csv("data/X_test.csv")
# - X_train = pd.read_csv("data/X_train.csv")
# - y_train = pd.read_csv("data/y_train.csv")
# 
# ### 사용자 코딩
# 
# ### 답안 제출 참고
# - 아래 코드 예측변수와 수험번호를 개인별로 변경하여 활용
# - pd.DataFrame({'cust_id': X_test.cust_id, 'gender': pred}).to_csv('003000000.csv', index=False)

# %% [code] {"execution":{"iopub.status.busy":"2021-12-01T23:14:43.533181Z","iopub.execute_input":"2021-12-01T23:14:43.533625Z","iopub.status.idle":"2021-12-01T23:14:43.557905Z","shell.execute_reply.started":"2021-12-01T23:14:43.533576Z","shell.execute_reply":"2021-12-01T23:14:43.557206Z"}}
# 시험환경 세팅 (코드 변경 X)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def exam_data_load(df, target, id_name="", null_name=""):
    if id_name == "":
        df = df.reset_index().rename(columns={"index": "id"})
        id_name = 'id'
    else:
        id_name = id_name
    
    if null_name != "":
        df[df == null_name] = np.nan
    
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=2021)
    
    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[target])

    
    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[target])
    return X_train, X_test, y_train, y_test 


# My code

import pandas as pd

df = pd.read_csv("../input/titanic/train.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='Survived', id_name='PassengerId')

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# EDA
print(X_train.describe())
print()
print(X_train.describe(include = 'object'))
print()
print(X_train.isnull().sum())
print()
print(X_test.isnull().sum())

# 결측치
X_train.drop(['Cabin'], axis=1, inplace = True)
X_test.drop(['Cabin'], axis=1, inplace = True)

# X_train.dropna(subset = ['Embarked'], inplace = True)
# X_test.dropna(subset = ['Embarked'], inplace = True)

age_median = X_train.Age.median()
X_train.Age.fillna(age_median, inplace = True)
X_test.Age.fillna(age_median, inplace = True)
X_train.Embarked.fillna('S', inplace = True)
X_test.Embarked.fillna('S', inplace = True)


# Feature Engeenering
print(X_train.describe(include = 'object'))

X_train.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace = True)
PassengerId = X_test.PassengerId
X_test.drop(['PassengerId','Name', 'Ticket'], axis=1, inplace = True)

print(X_train.describe(include = 'object'))

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X_train['Sex'] = le.fit_transform(X_train['Sex'])
X_test['Sex'] = le.transform(X_test['Sex'])
X_train['Embarked'] = le.fit_transform(X_train['Embarked'])
X_test['Embarked'] = le.transform(X_test['Embarked'])

print(X_train.head())

from sklearn.ensemble import RandomForestClassifier

model_1 = RandomForestClassifier()

model_1.fit(X_train, y_train['Survived'])
model_1.score(X_train, y_train['Survived'])
model_1.predict(X_test)
model_1.score(X_test, y_test['Survived'])
