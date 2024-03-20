# -*- coding: utf-8 -*-
"""
submission.py

Генерация ответа
"""

# %%
# Импортируем библиотеки

import pandas as pd
import os
from lightgbm import LGBMRegressor
import joblib


# %%
# Получаем данные

df_test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'results/df_test.csv'))
submission = pd.read_csv(os.path.join(os.path.dirname(__file__), 'submission_sample.csv'))
X_test = df_test.drop('id', axis=1)


# %%
# Загружаем модель и получаем предсказания

model = joblib.load(os.path.join(os.path.dirname(__file__), 'results/model.pkl'))
preds = model.predict(X_test)


# %%
# Генерируем файл submission.csv

submission['score'] = preds
submission.to_csv(os.path.join(os.path.dirname(__file__), 'submission.csv'), index=False)