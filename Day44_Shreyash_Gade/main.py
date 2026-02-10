import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

X = pd.DataFrame(data.data,columns= data.feature_names)
Y = data.target

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model = LogisticRegression()

model.fit(X_scaled,Y)

pred_lr = model.predict(X_scaled)

accuracy_lr = accuracy_score(Y,pred_lr)
print('Accuracy of Breast Cancer: ',accuracy_lr)

import joblib
joblib.dump(model,'model.pkl')
joblib.dump(scaler,'scaler.pkl')