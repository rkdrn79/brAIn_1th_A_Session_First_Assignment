import numpy as np
import pandas as pd

# to debug
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from src.naivebayes import NaiveBayesClassifier

# 데이터 생성
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print(train.head())

# 데이터 전처리
train = train.fillna('null')
test = test.fillna('null')

le = LabelEncoder()

obj_cols = train.select_dtypes(include='object').columns
for col in obj_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# 학습 데이터와 테스트 데이터 분리
X_train, y_train = train.drop('international', axis=1).values, train['international'].values
X_test, y_test = test.drop('international', axis=1).values, test['international'].values

# NaiveBayesClassifier 객체 생성 및 학습
nb_classifier = NaiveBayesClassifier(smoothing=1)
nb_classifier.fit(X_train, y_train)

# 테스트 데이터에 대해 예측 수행
predictions = nb_classifier.predict(X_test)

f1 = f1_score(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
confusion = confusion_matrix(y_test, predictions)

print(f'F1 Score: {f1}')
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{confusion}')
