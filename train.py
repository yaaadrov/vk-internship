# -*- coding: utf-8 -*-
"""
train.py

Обучение модели
"""

# %%
# Импортируем библиотеки

import pandas as pd
import os
from lightgbm import LGBMRegressor
import joblib


# %%
# Получаем данные

df_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'results/df_train.csv'))
X_train = df_train.drop(['id', 'score'], axis=1)
y_train = df_train['score']


# %%
# Модель - LGBMRegressor
# Оптимальные гиперпараметры определены с помощью Optuna

params = {
        'objective': 'regression',
        'metric': 'mae',
        'n_estimators': 1000,
        'verbosity': -1,
        'bagging_freq': 1,
        'learning_rate': 0.013986692443014913,
        'num_leaves': 741,
        'subsample': 0.896949691829106,
        'colsample_bytree': 0.26637160625306044,
        'min_data_in_leaf': 13,
    }


# %%
# Создаем и обучаем модель

model = LGBMRegressor(**params)
model.fit(X_train, y_train)


# %%
# Сохраняем модель

joblib.dump(model, os.path.join(os.path.dirname(__file__), 'results/model.pkl'))