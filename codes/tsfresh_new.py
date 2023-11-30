import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import datetime

import tsfresh
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute, make_forecasting_frame
from tsfresh.feature_extraction import ComprehensiveFCParameters

if __name__ == '__main__':
    all_path = 'D:\\Spain_clip\\to_tsfresh_false'
    path = all_path + '/ISI'
    files = [file for file in os.listdir(path) if file.endswith(".txt")]

    excel_data = pd.read_excel(r'D:\Spain_clip\LFMC_Spain.xlsx', sheet_name='Sheet1')
    start_time = pd.to_datetime('2003-1-1')

    y = []
    features = pd.DataFrame()
    for index, row in excel_data.iterrows():
        site = row['Site number']
        time = row['Date']
        FMC = row['FMC']

        if time >= start_time:

            txt_tsfresh = pd.DataFrame()

            for file in tqdm(files):
                if file.split('Site_')[1].split('.txt')[0] == str(site):
                    txt = pd.read_csv(path + '/' + file, sep=',', header=0)

                    count = -1
                    for index_1, row_1 in txt.iterrows():
                        count = count + 1
                        time_txt = row_1['date']

                        time_int = int(time.strftime("%Y%m%d"))

                        if time_txt == time_int and count >= 9:
                            y.append(FMC)
                            txt_tsfresh.append(txt.loc[0])
                            i = 0

                            txt_tsfresh = txt.loc[count-9:count]
                            # for j in range(count - 9, count):
                            #     txt_tsfresh.append(txt.loc[j])
                            #     i = i + 1
                            feature = extract_features(txt_tsfresh,
                                                       column_id='Site number',
                                                       column_sort='date',
                                                       column_value='data')
                            features = features.append(feature)
    impute(features)
    y_array = np.array(y)

    features.to_csv('D:\\Spain_clip\\results/features_ISI.txt', index=False)

    features_filtered = select_features(features, y_array)
    features_filtered.to_csv('D:\\Spain_clip\\results/features_ISI_filtered.txt', index=False)
