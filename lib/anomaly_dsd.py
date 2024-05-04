import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns

import matplotlib.pyplot as plt
from read_data import loadData
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

def generate_segment_means(series_segments, num_series, num_segments):
    '''
    Generating mean row wise for as above
    '''
    means = []
    for j in range(num_segments):
        C = [series_segments[i][j] for i in range(num_series)] # Taking all row values in each segment
        means.append([sum(x)/num_series for x in zip(*C)]) # Taking the mean of each row.
    return means

def diffstd(s1v, s2v):
    '''
    Both will be 1D vector of length one segment. 
    s1v: Segment for which we are doing the difference.
    s2v: Here the segment mean is row wise mean so each element in the segment has a mean value here the mean is mean of all the column 
    values for a specific row.
    '''
    dt = [x1 - x2 for (x1, x2) in zip(s1v, s2v)]
    n = len(s1v)
    mu = np.mean(dt)
    diff2 = [(d-mu)**2 for d in dt]
    return (np.sum(diff2)/n)**0.5

def check_diffstd(series_segments, segment_means, num_series, num_segments):
    '''
    diffstd_distance: It has the standard deviation of the difference of each element in the segment 
    '''
    for i in range(num_series): #22
        for j in range(num_segments): #22908
            series_segments[i][j]['segment_number'] = j
            series_segments[i][j]['diffstd_distance'] = diffstd(series_segments[i][j], segment_means[j])
    return series_segments

def add_label(df_all, max_perc):
    df_all['actual_label'] = 0
    for engine in list(set(df_all['engine'])):
        max_cycle = df_all[df_all['engine'] == engine]['cycle'].max() 
        condition = (df_all['engine'] == engine) & (df_all['cycle'] > max_cycle*max_perc)
        df_all.loc[condition, 'actual_label'] = 1
    df_all = df_all.fillna(0)
    return df_all

def anomaly_diffstd(df_all, df_full):
    series = [df_all[col] for col in df_all.columns.drop('engine', errors='ignore')]
    num_series = len(series)
    l = len(series[0])
    series_segments = [np.array_split(series[x], (l // 7)) for x in range(num_series)]
    num_segments = len(series_segments[0])
    segment_means = generate_segment_means(series_segments, num_series, num_segments)
    #len(test1), len(test1[0]), len(test1[0][0]),  len(segment_means), len(segment_means[0]), num_series, num_segments
    segments_diffstd = check_diffstd(series_segments, segment_means, num_series, num_segments)
    
    df_diff = pd.DataFrame()
    for i in range(len(segments_diffstd)): #num of columns
        col = segments_diffstd[i][0].name
        df_diff[col] = [segments_diffstd[i][j]['diffstd_distance'] for j in range(len(segments_diffstd[i]))]
    
    sensitivity_score = 100
    for i, col in enumerate(df_diff.columns):
        diffstd_mean = df_diff[col].mean() # Taking mean of diffstd values 
        diffstd_sensitivity_threshold = diffstd_mean + ((1.5 - (sensitivity_score / 100.0)) * diffstd_mean)
        df_diff[f'{col}_anomaly_score'] = (df_diff[col] - diffstd_sensitivity_threshold) / diffstd_sensitivity_threshold

    max_fraction_anomalies = 0.2
    max_fraction_anomaly_scores = [np.quantile(df_diff[col], 1.0 - max_fraction_anomalies) for col in df_diff.columns if col.endswith("anomaly_score")]
    sensitivity_thresholds = [max(0.01, mfa) for mfa in max_fraction_anomaly_scores]
    df_anomaly = df_diff[[col for col in df_diff.columns if col.endswith("anomaly_score")]]
    df_final = df_all.drop(columns = ['engine'], errors='ignore')
    df_repeated = pd.DataFrame(np.repeat(df_anomaly.values, 7, axis=0), columns=df_anomaly.columns)
    df_result = pd.concat([df_final[:len(df_repeated)].reset_index(drop = True) ,df_repeated.reset_index(drop = True)], axis=1)
    sens_dict = {}
    for i, col in enumerate(df_result.filter(like = 'anomaly').columns):
        sens_dict[col] = sensitivity_thresholds[i]
    df_result['pred_label'] = 0
    for col in df_result.filter(like = 'anomaly').columns:
        for i in range(len(df_result)):
            if df_result.loc[i, col] > sens_dict[col]:
                df_result.loc[i, 'pred_label'] = 1
    
    df_full = add_label(df_full, 0.9)
    print(classification_report(df_full['actual_label'][:len(df_result)], df_result['pred_label']))
    return df_result['pred_label']


if __name__ =="__main__":
        
    data_train, testDatasets, expectedRulDatasets = loadData()
    drop_col = ['setting_1', 'setting_2', 'setting_3'] 
    data_clean = [data.drop(columns = drop_col) for data in data_train]

    df_full = pd.DataFrame()
    for i in range(len(data_clean)):
        df_full =pd.concat([df_full, data_clean[i]])

    temp_cols = ['Fan_inlet_temperature_R', 'LPC_outlet_temperature_R', 'HPC_outlet_temperature_R', 'LPT_outlet_temperature_R']
    rpm_cols = ['Physical_fan_speed_rpm', 'Physical_core_speed_rpm', 'Corrected_fan_speed_rpm', 'Corrected_core_speed_rpm']
    press_cols = ['bypass_duct_pressure_psia', 'HPC_outlet_pressure_psia','Engine_pressure_ratioP50_P2', 'HPC_outlet_Static_pressure_psia']
    air_flow_cols = ['High_pressure_turbines_Cool_air_flow', 'Low_pressure_turbines_Cool_air_flow']
    # diff_col_type = [temp_cols,rpm_cols,press_cols,air_flow_cols]
    diff_col_type = [air_flow_cols]
    for col_typ in diff_col_type:
        df_all = df_full[col_typ]
        print(f"Report for {str(col_typ)}")
        result = anomaly_diffstd(df_all, df_full)
        result.to_csv(f"{str(col_typ)}_result.csv", index = False)