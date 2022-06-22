# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T13:20:07.787108Z","iopub.execute_input":"2021-11-24T13:20:07.787697Z","iopub.status.idle":"2021-11-24T13:20:08.751546Z","shell.execute_reply.started":"2021-11-24T13:20:07.78766Z","shell.execute_reply":"2021-11-24T13:20:08.750509Z"}}
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
    
    X_train, X_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=2021)
    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[id_name, target])
    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[id_name, target])
    return X_train, X_test, y_train, y_test 
    
df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='SalePrice', id_name='Id')

X_train.shape, X_test.shape, y_train.shape, y_test.shape

X_train.head()
X_train.info()

X_train = X_train.select_dtypes(exclude=['object'])
X_test = X_test.select_dtypes(exclude=['object'])
y_train_target = y_train['SalePrice']
y_test_target = y_test['SalePrice']


len(X_train)


# Missing Value

X_train.isnull().sum()

LF_med = X_train['LotFrontage'].median()
GYB_med = X_train['GarageYrBlt'].median()
MVA_med = X_train['MasVnrArea'].median()

def missing_processing(df):
    df['LotFrontage'] = df['LotFrontage'].fillna(LF_med)
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(GYB_med)
    df['MasVnrArea'] = df['MasVnrArea'].fillna(MVA_med)
    # or
    # df.fillna(df.median())
    
    return df

missing_processing(X_train)
missing_processing(X_test)

X_train.isnull().sum()



# Feature processing

from sklearn.preprocessing import StandardScaler

X_train.describe()

# help(sklearn.preprocessing)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeling

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

model_1 = RandomForestRegressor()
model_1.fit(X_train_scaled, y_train_target)
y_pred_1 = model_1.predict(X_test_scaled)

model_2 = XGBRegressor()
model_2.fit(X_train_scaled, y_train_target)
y_pred_2 = model_2.predict(X_test_scaled)


#Scoring Model 
from sklearn.metrics import mean_squared_error

def rmse(y, y_pred):
    return mean_squared_error(y, y_pred, squared = False) # False -> RMSE

print(f"랜덤포레스트's RMSE: {rmse(y_pred_1, y_test_target)}")
print(f"XGBoost's RMSE: {rmse(y_pred_2, y_test_target)}")