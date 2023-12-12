import pandas as pd
import numpy as np
import math
import joblib
import gc
gc.enable()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', 100)
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import lightgbm as lgb

def read_file_csv(path_file):
    data = pd.read_csv(path_file)
    return data

def convert_type(df):
    df.ref_xa_code = df.ref_xa_code.astype(np.int64)
    df.ref_huyen_code = df.ref_huyen_code.astype(np.int64)
    df.ref_tinh_code = df.ref_tinh_code.astype(np.int64)
    df.ref_xa_code = df.ref_xa_code.astype('str')
    df.ref_huyen_code = df.ref_huyen_code.astype('str')
    df.ref_tinh_code = df.ref_tinh_code.astype('str')
    df.prj_name = df.prj_name.astype('str')
    df.duong = df.duong.astype('str')
    return df

def caculator_dataframe(df):
    df['ref_xa_huyen_tinh_code'] = df['ref_xa_code'] + df['ref_huyen_code'] + df['ref_tinh_code']
    df['ref_huyen_tinh_code'] = df['ref_huyen_code'] + df['ref_tinh_code']
    df['ref_xa_huyen_code'] = df['ref_xa_code'] + df['ref_huyen_code']
    df['ref_xa_tinh_code'] = df['ref_xa_code'] + df['ref_tinh_code']
    df.ref_xa_code = df.ref_xa_code.astype('category')
    df.ref_huyen_code = df.ref_huyen_code.astype('category')
    df.ref_tinh_code = df.ref_tinh_code.astype('category')
    df.prj_name = df.prj_name.astype('category')
    df.ref_xa_huyen_tinh_code = df.ref_xa_huyen_tinh_code.astype('category')
    df.ref_huyen_tinh_code = df.ref_huyen_tinh_code.astype('category')
    df.ref_xa_huyen_code = df.ref_xa_huyen_code.astype('category')
    df.ref_xa_tinh_code = df.ref_xa_tinh_code.astype('category')
    df.prj_name = df.prj_name.astype('category')
    df.duong = df.duong.astype('category')
    return df

def quantiles(df):
    quanti = [i/10 for i in range(1,10)] + [.95,.975, .99, .995]
    df.unit_price.describe(quanti)
    # sns.distplot(df.unit_price)
    return df

def processing_data(df):
    df = convert_type(df)
    df = caculator_dataframe(df)
    df = quantiles(df)
    return df


def split_data(df):
    global X, y
    X = df[['prj_name', 'duong', 'ref_xa_code', 'ref_huyen_code', 'ref_tinh_code', 
            'ref_xa_huyen_code', 'ref_huyen_tinh_code', 'ref_xa_tinh_code', 'ref_xa_huyen_tinh_code', 'pn', 'area']]
    y = df.unit_price
    X, y = shuffle(df[['prj_name', 'duong', 'ref_xa_code', 'ref_huyen_code', 'ref_tinh_code', 
            'ref_xa_huyen_code', 'ref_huyen_tinh_code', 'ref_xa_tinh_code', 'ref_xa_huyen_tinh_code', 'pn', 'area']], df.unit_price, random_state=22)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=23)
    return X_train, X_test, y_train, y_test

def save_model(model, path_save):
    joblib.dump(model, path_save)

def modelling(X_train, X_test, y_train, y_test):
    hyper_params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'mape',
                'learning_rate': 0.01,
                'feature_fraction': 0.80,
                'bagging_fraction': 0.80,
                'bagging_freq': 8,
                'max_depth': 16,
                'num_leaves': 32,
                'max_bin': 256,
                'num_iterations': 1000,              
                'min_child_weight': 0.0001, 
                #'max_cat_threshold': 40,
                #'min_data_per_group':80, 
                'min_split_gain': 0.0,                  
                'min_data_in_leaf':10,
                'cat_smooth': 8,
                'cat_l2': 4,
                'path_smooth':2,
                'verbose':-1
                }
    gbm = lgb.LGBMRegressor(**hyper_params)
    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='mape')
    return gbm

def evaluate(gbm_model, X_test, y_test):
    y_pred_test = gbm_model.predict(X_test, num_iteration=gbm_model.best_iteration_) 
    MSE_gbm_test = mean_squared_error(y_test, y_pred_test)
    RMSE_gbm_test = math.sqrt(MSE_gbm_test)
    MAE_gbm_test = mean_absolute_error(y_test, y_pred_test) 
    MAPE_gbm_test = mean_absolute_percentage_error(y_test, y_pred_test)
    R2_gbm_test = r2_score(y_test, y_pred_test)
    R2a_gbm_test = 1 - (1-metrics.r2_score(y_test, y_pred_test))*(len(y_test)-1)/(len(y_test) - X_test.shape[1])
    return {
        "MSE_gbm_test": MSE_gbm_test,
        "RMSE_gbm_test": RMSE_gbm_test,
        "MAE_gbm_test": MAE_gbm_test,
        "MAPE_gbm_test": MAPE_gbm_test,
        "R2_gbm_test": R2_gbm_test,
        "R2a_gbm_test": R2a_gbm_test
    }

def train_model():
    path_data = r'C:\Users\biennn1\Downloads\housing_master_v2-20231211T011230Z-001\housing_master_v2\data_output\bds_processed_0512.csv'
    dataframe = read_file_csv(path_data)
    dataframe = processing_data(dataframe)
    X_train, X_test, y_train, y_test = split_data(dataframe)
    gbm_model = modelling(X_train, X_test, y_train, y_test)
    save_model(gbm_model, r'C:\Users\biennn1\Downloads\housing_master_v2-20231211T011230Z-001\housing_master_v2\model\model.joblib')
    result = evaluate(gbm_model, X_test, y_test)
    print(result)

if __name__ == '__main__':
    train_model()