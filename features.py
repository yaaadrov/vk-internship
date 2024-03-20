# -*- coding: utf-8 -*-
"""
features.py

Генерация признаков
"""

# %%
# Импортируем библиотеки

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os


# %%
# Получаем данные

train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'train.csv'))
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'test.csv'))
features = pd.read_csv(os.path.join(os.path.dirname(__file__), 'features.csv'))


# %%
# Функция для генерации признаков

def add_features(df_original, df_features, n_neighbors):
    """
    Создает датасет с подготоленными переменными
    Как:
        по координатам (lat, lon) оригинального датасета подбирает n_neighbors ближайших соседей из датасета с признаками
        и для всех наблюдений в качестве признаков устанавливает вектор усредненных знаечний
    Input:
        df_original (DataFrame): датасет с обязательными столбцами 'id', 'lat' и 'lon' для добавления данных
        df_features (DataFrame): датасет с обязательными столбцами 'lat', 'lon' и столбцами-признаками
        n_neighbors (int):       число соседей для усреднения признаков
    Output:
        df (DataFrame):          подготовленный датасет
    """

    # Все значения широты и долготы df_features
    lat_lon = df_features[['lat', 'lon']]

    # Копия df_original со столбцами df_features
    df = df_original.merge(df_features, how='left')

    # Столбцы со значениями признаков
    feature_cols = df_features.drop(['lat', 'lon'], axis=1).columns

    for id in range(df.shape[0]):
        x = lat_lon - df.loc[id, ['lat', 'lon']]
        ids = ((x.iloc[:, 0] ** 2 + x.iloc[:, 1] ** 2) ** 0.5).nsmallest(n_neighbors).index
        df.loc[id, feature_cols] = df_features.iloc[ids].mean()

    return df


# %%
# Генерируем признаки

df_train = add_features(train, features, 5)
df_test = add_features(test, features, 5)


# %%
# Масштабируем

scaler = MinMaxScaler()
df_train[['lat', 'lon']] = scaler.fit_transform(df_train[['lat', 'lon']])
df_test[['lat', 'lon']] = scaler.transform(df_test[['lat', 'lon']])


# %%
# Список лишних признаков получен с помощью
# (1) корреляционного анализа
# (2) анализа важности признаков через feature_importances_

features_to_drop = [
    '20', '14', '13', '129', '128', '135', '244', '243', '49', '250',
    '188', '303', '73', '176', '164', '85', '334', '57', '191', '319',
    '257', '315', '84', '172', '298', '88', '66', '349', '302', '64',
    '279', '195', '274', '187', '335', '75', '216', '306', '168', '340',
    '68', '113', '107', '61', '140', '142', '215', '17', '350', '353',
    '3', '4', '5', '77', '78', '79', '81', '86', '87', '94', '95', '97',
    '98', '99', '102', '103', '115', '116', '117', '118', '119', '120',
    '121', '122', '126', '131', '132', '133', '134', '137', '138', '141',
    '145', '146', '148', '149', '150', '151', '156', '157', '158', '161',
    '162', '165', '166', '167', '169', '170', '173', '174', '177', '178',
    '181', '182', '185', '186', '189', '190', '192', '193', '194', '196',
    '197', '198', '199', '201', '202', '205', '206', '207', '209', '210',
    '212', '213', '214', '217', '218', '219', '223', '227', '228', '229',
    '230', '231', '232', '233', '234', '235', '236', '241', '246', '247',
    '248', '249', '252', '253', '256', '260', '261', '263', '264', '265',
    '266', '269', '271', '272', '273', '275', '276', '277', '280', '281',
    '282', '284', '285', '288', '289', '292', '293', '296', '297', '300',
    '301', '304', '305', '307', '308', '309', '311', '312', '313', '314',
    '316', '317', '320', '321', '322', '324', '325', '327', '328', '329',
    '332', '333', '337', '341', '342', '343', '344', '355', '356', '360',
    '361', '362'
]


# %%
# Удаляем лишние признаки

df_train = df_train.drop(features_to_drop, axis=1)
df_test = df_test.drop(features_to_drop, axis=1)


# %%
# Сохраняем подготовленные данные в папку results

df_train.to_csv(os.path.join(os.path.dirname(__file__), 'results/df_train.csv'), index=False)
df_test.to_csv(os.path.join(os.path.dirname(__file__), 'results/df_test.csv'), index=False)


