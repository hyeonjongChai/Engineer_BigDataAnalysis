# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train = pd.read_csv('../input/big-data-analytics-certification/t2-1-train.csv')
test = pd.read_csv('../input/big-data-analytics-certification/t2-1-test.csv')
sample_submission = pd.read_csv('../input/big-data-analytics-certification/t2-1-sample_submission.csv')


train.head()
train_id = train.iloc[:,0]
X_train = train.iloc[:,1:-1]
y_train = train.iloc[:,-1]

test_id = test.iloc[:,0]
X_test = test.iloc[:,1:]


X_train.shape, X_test.shape


# EDA

num_cols = list(X_train.select_dtypes(exclude = 'object').columns)
cat_cols = list(X_train.select_dtypes(include = 'object').columns)

# X_train.describe(exclude = 'object')
# X_train.describe(include = 'object')
X_train.info()
X_test.info()

X_train.isnull().sum()
X_test.isnull().sum()

# # Feature Engeering

X_train['AnnualIncome'].fillna(X_train['AnnualIncome'].median(), inplace=True)
X_test['AnnualIncome'].fillna(X_test['AnnualIncome'].median(), inplace=True)


X_train['ChronicDiseases'] = X_train['ChronicDiseases'].astype(int)
X_test['ChronicDiseases'] = X_test['ChronicDiseases'].astype(int)



X_train_test = pd.concat([X_train, X_test])
X_train_test.head()

from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer

scaler = RobustScaler()

X_train_test[num_cols] = scaler.fit_transform(X_train_test[num_cols])

from sklearn.preprocessing import LabelEncoder

for c in cat_cols:
    encoder = LabelEncoder()
    X_train_test[c] = encoder.fit_transform(X_train_test[[c]])


X_train = X_train_test[:len(X_train)]
X_test = X_train_test[len(X_train):]

X_train.head()
X_test.head()

# Modelling

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier, XGBClassifier

# help(RandomForestClassifier)
RF = RandomForestClassifier()
# XGB = XGBClassifier()
# XGBRF = XGBRFClassifier()

RF.fit(X_train, y_train)
print(RF.score(X_train, y_train))

# XGB.fit(X_train, y_train)
# print(XGB.score(X_train, y_train))

# XGBRF.fit(X_train, y_train)
# print(XGBRF.score(X_train, y_train))

pred = RF.predict_proba(X_test)[:,1]
pred

submission = pd.DataFrame({'id': test_id, 'TravelInsurance': pred})
submission
submission.to_csv('TravelInsurance_Predict_Proba', index = False)

pd.read_csv('TravelInsurance_Predict_Proba')
