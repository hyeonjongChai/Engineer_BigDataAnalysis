# %% [markdown]
# # 작업형2 기출 유형(심화)
# - 본 문제는 변형한 심화 문제 입니다.
# - 오리지널 3회 기출 유형을 보고 싶은 분은 아래 클래스-커리큘럼 탭에 무료공개(3회 작업형2)로 영상과 데이터셋을 올려놨어요!
# - https://class101.net/products/467P0ZPH0lVX9FwFBDz7
# 
# ### 여행 보험 패키지 상품을 구매할 확률 값을 구하시오
# - 예측할 값(y): TravelInsurance (여행보험 패지지를 구매 했는지 여부 0:구매안함, 1:구매)
# - 평가: roc-auc 평가지표
# - data: t2-1-train.csv, t2-1-test.csv
# - 제출 형식
# 
# ~~~
# id,TravelInsurance
# 0,0.3
# 1,0.48
# 2,0.3
# 3,0.83
# ~~~
# 
# # Baseline
# ### 3회 기출문제에서 데이터 셋을 편집해 조금 더 어렵게 만들었어요
# - 결측치 추가
# - Employment Type 컬럼에 카테고리 추가 
# - sample_submission 파일은 제공된 적 없음(3회 때 제출 형식에 대한 이슈가 있어 제공하거나 제출 형식을 명확하게 설명할 가능성 있어 보임)

# %% [code] {"execution":{"iopub.status.busy":"2022-06-14T14:55:19.742731Z","iopub.execute_input":"2022-06-14T14:55:19.743242Z","iopub.status.idle":"2022-06-14T14:55:19.771533Z","shell.execute_reply.started":"2022-06-14T14:55:19.743149Z","shell.execute_reply":"2022-06-14T14:55:19.770761Z"},"jupyter":{"outputs_hidden":false}}
# 라이브러리 불러오기
import pandas as pd

# %% [code] {"execution":{"iopub.status.busy":"2022-06-14T14:55:19.772981Z","iopub.execute_input":"2022-06-14T14:55:19.773329Z","iopub.status.idle":"2022-06-14T14:55:19.805999Z","shell.execute_reply.started":"2022-06-14T14:55:19.773291Z","shell.execute_reply":"2022-06-14T14:55:19.805117Z"},"jupyter":{"outputs_hidden":false}}
# 데이터 불러오기
train = pd.read_csv("../input/big-data-analytics-certification/t2-1-train.csv")
test = pd.read_csv("../input/big-data-analytics-certification/t2-1-test.csv")


train.shape, test.shape

X_train = train.iloc[:,1:-1]
y_train = train['TravelInsurance']

X_test = test.iloc[:,1:]
X_test_id = test.iloc[:,0]
# y_test = test['TravelInsurance']

# dir(X_train)

num_cols = list(X_train.select_dtypes(exclude = 'object').columns)
cat_cols = list(X_train.select_dtypes(include = 'object').columns)

X_train[num_cols].describe()
X_train[cat_cols].describe()


# Missing value handling

X_train[num_cols].isnull().sum()
X_train['AnnualIncome'].fillna(X_train['AnnualIncome'].median(), inplace = True)
X_test[num_cols].isnull().sum()
X_test['AnnualIncome'].fillna(X_train['AnnualIncome'].median(), inplace = True)


# X_train[cat_cols].isnull().sum()
X_test[cat_cols].isnull().sum()


# Feature engineering

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

scaler = MinMaxScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

for c in cat_cols:
    le = LabelEncoder()
    X_train[c] = le.fit_transform(X_train[c])
    X_test[c] = le.fit_transform(X_test[c])
    

        
# Modelling

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
model.score(X_train, y_train)

pred = model.predict_proba(X_test)


# submission

submit = pd.DataFrame()
submit['id'] = test['id']
submit['TravelInsurance'] = pred[:,1]

submit