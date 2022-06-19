# %% [markdown]
# ## 당뇨병 여부 판단
# - 이상치 처리 (Glucose, BloodPressure, SkinThickness, Insulin, BMI가 0인 값)

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

# %% [code] {"execution":{"iopub.status.busy":"2021-12-01T23:19:33.777209Z","iopub.execute_input":"2021-12-01T23:19:33.777607Z","iopub.status.idle":"2021-12-01T23:19:33.804526Z","shell.execute_reply.started":"2021-12-01T23:19:33.777571Z","shell.execute_reply":"2021-12-01T23:19:33.803391Z"}}
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
    
df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='Outcome')

# EDA
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print()
print(X_train.head())
print()
print(y_train.head())
print()
print(X_train.info())
print()
print(X_train.describe())
print()
print(y_train.describe())
print()
print(X_train.isnull().sum())
print()
print(len(X_train[X_train['Glucose']==0]))
print(len(X_train[X_train['BloodPressure']==0]))
print(len(X_train[X_train['SkinThickness']==0]))
print(len(X_train[X_train['Insulin']==0]))
print(len(X_train[X_train['BMI']==0]))
print()
print(len(X_test[X_test['Glucose']==0]))
print(len(X_test[X_test['BloodPressure']==0]))
print(len(X_test[X_test['SkinThickness']==0]))
print(len(X_test[X_test['Insulin']==0]))
print(len(X_test[X_test['BMI']==0]))

# preprocessing


# 이상치
# 이상치는 모두 median으로 대체

outlier_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
cols_mean = X_train[outlier_cols].mean()
X_train[outlier_cols].replace(0,cols_mean)
X_test[outlier_cols].replace(0,cols_mean)


# Scale

from sklearn.preprocessing import StandardScaler,QuantileTransformer

scaler = QuantileTransformer()
scale_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

print(X_train.head())


# ID 처리

X = X_train.iloc[:,1:]
X_ = X_test.iloc[:,1:]

# Model

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth = 8, min_samples_split = 3, random_state=2022)
model.fit(X, y_train['Outcome'])
pred = model.predict(X_)

print(model.score(X, y_train['Outcome']))
print(model.score(X_,y_test['Outcome']))
