# customer_churn.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 示例數據
data = {
    'age': np.random.randint(18, 70, 100),
    'income': np.random.randint(20000, 100000, 100),
    'churn': np.random.randint(0, 2, 100)
}

# 創建 DataFrame
df = pd.DataFrame(data)

# 分割數據集
X = df[['age', 'income']]
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 創建並訓練模型
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 評估
print(classification_report(y_test, y_pred))