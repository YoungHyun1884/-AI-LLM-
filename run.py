import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# 경로 설정
train_path = "/home/park/Desktop/sw중심대학/Data/train.csv"
test_path = "/home/park/Desktop/sw중심대학/Data/test.csv"
submit_path = "/home/park/Desktop/sw중심대학/Data/sample_submission.csv"
save_path = "/home/park/Desktop/sw중심대학/submission.csv"

# 1. 데이터 불러오기
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission = pd.read_csv(submit_path)

# 2. TF-IDF + XGBoost 파이프라인 구성
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('xgb', XGBClassifier(n_estimators=200, max_depth=4, use_label_encoder=False, eval_metric='logloss'))
])

# 3. 학습
X_train = train_df['full_text']
y_train = train_df['generated']
pipeline.fit(X_train, y_train)

# 4. 예측
X_test = test_df['paragraph_text']

y_pred = pipeline.predict(X_test)

# 5. 결과 저장
submission['generated'] = y_pred
submission.to_csv(save_path, index=False)
print("✅ 제출 파일 저장 완료:", save_path)
