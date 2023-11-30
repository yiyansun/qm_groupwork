import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm import trange
import time

import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn import datasets

# path choose from(grass, shrub and forest)
x_path = 'D:\\Spain_clip\\random_forest_x\\grass'
y_path = 'D:\\Spain_clip\\random_forest_y\\grass'

excel_data = pd.read_excel(r'D:\Spain_clip\LFMC_Spain.xlsx', sheet_name='Sheet2')

all_data = pd.DataFrame()
single_data = pd.DataFrame()

output = 'D:\\Spain_clip\\train_material\\grass/' + 'x_and_y_all.txt'

# classify the vegetation of sites;
for site_number in range(60, 170):

    file_x_path = 'D:\\Spain_clip\\random_forest_x'
    file_y_path = 'D:\\Spain_clip\\random_forest_y'

    file_judge = file_x_path + '/' + 'x_from_site_' + str(site_number) + '.txt'

    if not os.path.exists(file_judge):
        continue

    x_shrub_path = 'D:\\Spain_clip\\random_forest_x\\shrub/' + 'x_from_site_' + str(site_number) + '.txt'
    x_grass_path = 'D:\\Spain_clip\\random_forest_x\\grass/' + 'x_from_site_' + str(site_number) + '.txt'
    x_forest_path = 'D:\\Spain_clip\\random_forest_x\\forest/' + 'x_from_site_' + str(site_number) + '.txt'
    y_shrub_path = 'D:\\Spain_clip\\random_forest_y\\shrub/' + 'y_from_site_' + str(site_number) + '.txt'
    y_grass_path = 'D:\\Spain_clip\\random_forest_y\\grass/' + 'y_from_site_' + str(site_number) + '.txt'
    y_forest_path = 'D:\\Spain_clip\\random_forest_y\\forest/' + 'y_from_site_' + str(site_number) + '.txt'

    x_file = pd.read_csv(file_x_path + '/' + 'x_from_site_' + str(site_number) + '.txt', sep=',', header=0)
    y_file = pd.read_csv(file_y_path + '/' + 'y_from_site_' + str(site_number) + '.txt', sep=',', header=0)

    for index, row in excel_data.iterrows():
        vegetation = row['Vegetation (Grass, Shrub, Forest)']
        site = row['Site number']

        if site == float(site_number):
            if vegetation == 'Shrub':
                x_file.to_csv(x_shrub_path, index=False)
                y_file.to_csv(y_shrub_path, index=False)
            if vegetation == 'Grass':
                x_file.to_csv(x_grass_path, index=False)
                y_file.to_csv(y_grass_path, index=False)
            else:
                x_file.to_csv(x_forest_path, index=False)
                y_file.to_csv(y_forest_path, index=False)

for site_number in range(60, 170):

    file_judge = x_path + '/' + 'x_from_site_' + str(site_number) + '.txt'

    if not os.path.exists(file_judge):
        continue

    x_file = pd.read_csv(x_path + '/' + 'x_from_site_' + str(site_number) + '.txt', sep=',', header=0)
    y_file = pd.read_csv(y_path + '/' + 'y_from_site_' + str(site_number) + '.txt', sep=',', header=0)

    single_data = x_file
    single_data["FMC"] = y_file["FMC"]

    all_data = pd.concat([all_data, single_data], axis=0)
all_data.sort_values(by='date', inplace=True, ascending=True)
all_data.to_csv(output, index=False)

train_group = all_data.sample(frac=0.7, replace=False)
test_group = all_data.sample(frac=0.3, replace=False)

train_y = train_group['FMC']
train_x = train_group.loc[:, 'humidity':'FFMC_quantile']
test_y = test_group['FMC']
test_x = test_group.loc[:, 'humidity':'FFMC_quantile']

regressor = RandomForestRegressor(n_estimators=500, random_state=0)
regressor.fit(train_x, train_y)
score = regressor.score(test_x, test_y)

print('R2 = ', score)

result = regressor.predict(test_x)
plt.figure()
# plt.plot(np.arange(len(result)), test_y, "go-", label="True value")
# plt.plot(np.arange(len(result)), result, "ro-", label="Predict value")
plt.scatter(test_y, result)
plt.title(f"RandomForest---score:{score}")
# plt.legend(loc="best")
plt.show()

joblib.dump(regressor, 'D:\Spain_clip\save_model/grass_model.pkl')
