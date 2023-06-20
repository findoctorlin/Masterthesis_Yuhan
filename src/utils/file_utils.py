import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import warnings
warnings.filterwarnings('ignore')

index_names = ['unit_number', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
col_names = index_names + setting_names + sensor_names
drop_labels_1 = ['s_1', 's_5','s_6','s_10',  's_16', 's_18', 's_19']

def read_original_file(file_path):
    df = pd.read_csv(file_path, sep='\s+', header=None,index_col=False,names=col_names)
    return df


def read_original_file_RUL(file_path):
    df = pd.read_csv(file_path, sep='\s+', header=None,index_col=False,names=['RUL'])
    return df


def add_RUL_column(df):
    train_grouped_by_unit = df.groupby(by='unit_number') 
    max_time_cycles = train_grouped_by_unit['time_cycles'].max() 
    merged = df.merge(max_time_cycles.to_frame(name='max_time_cycle'), left_on='unit_number',right_index=True)
    merged["RUL"] = merged["max_time_cycle"] - merged['time_cycles']
    merged = merged.drop("max_time_cycle", axis=1) 
    return merged


# dropping unnecessary features
def drop_col(df):
    drop_labels = index_names+setting_names
    df = df.drop(columns=drop_labels)
    df = df.drop(columns=drop_labels_1, axis=1)
    return df


# clip the rows where RUL > RUL_clip_value
def clip_row(df, RUL_clip_value):
    df_clip = df[df['RUL'] <= RUL_clip_value]
    df_clip = df_clip.reset_index(drop=True)
    return df_clip

# Adding new past data as new features
def add_history_data(df, window_size):
    df_new = pd.DataFrame()
    df_new = df.copy()
    for feature in df.columns[:-1]:
        df_new[feature] = df[feature]

    for i in range(1, window_size+1):
        for feature in df.columns[:-1]:
            df_new[f'{feature}_lag{i}'] = df[feature].shift(i)

    rul_column = df_new.pop('RUL') # 使用 pop 函数将第 15 列 'RUL' 从 X_train_new 中移除并存储到变量 rul_column 中
    df_new.insert(df_new.shape[1], 'RUL', rul_column) # 使用 insert 函数将 rul_column 插入到最后一列

    # process the NaN value in first window_size columns
    for j in range(1, window_size+1):
        for feature in df.columns[:-1]:
            lag_feature = f'{feature}_lag{j}'
            df_new[lag_feature].fillna(df[feature], inplace=True)

    return df_new

# dropping RUL row and split dataset into train/validation
def drop_n_split(df, split_ratio):
    label = df['RUL']
    features = df.drop('RUL', axis=1)
    X_train, X_val, y_train, y_val = train_test_split(features, label, test_size=split_ratio, shuffle=False)
    
    y_train = y_train.to_frame().values.reshape(-1,1) # transfer y from (xxx, ) to (xxx, 1)
    y_val = y_val.to_frame().values.reshape(-1,1)

    return X_train, X_val, y_train, y_val

# dropping RUL row but not split dataset into train/validation
def drop_not_split(df):
    y_train = df['RUL']
    X_train = df.drop('RUL', axis=1)
    
    y_train = y_train.to_frame().values.reshape(-1,1) # transfer y from (xxx, ) to (xxx, 1)

    return X_train, y_train

# scalling the data
def data_scale(df):
    scaler = StandardScaler()
    df_new = scaler.fit_transform(df)
    if len(df.shape) == 1: # if we scale the label y
        df_new = df_new.values.reshape(-1,1)

    return df_new

# Data reshape
def reshape_data(ndarray):
    if(ndarray.shape[1] != 1): # features X
        df = pd.DataFrame(ndarray)
        df_values = df.values
        df_tensor = torch.tensor(df_values, dtype=torch.float32)
        df_train= df_tensor.contiguous()
    else:                      # label y
        df = pd.DataFrame(ndarray)
        df_values = df.values
        df_tensor = torch.tensor(df_values, dtype=torch.float32)
        df_tensor = df_tensor.squeeze() # transfer y from (xxx, ) to (xxx, 1)
        df_train= df_tensor.contiguous()
    
    return df_train