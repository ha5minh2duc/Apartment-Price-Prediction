from modelling import read_file_csv, predict
from joblib import  load, dump
from utils import convert_dataframe
import pandas as pd
from processing import preprocessing_data, select_feature_data, filtering, fillna_missing

def convert_data(df, path_loc):
    df = preprocessing_data(df, path_loc)
    df = select_feature_data(df)
    df = filtering(df)
    df = fillna_missing(df)
    return df

def main():
    data_input = [{
        "price": 4500000000.0,
        "area": 106.0,
        "pn": 2,
        "toilet": 2,
        "date": "sunrise-city",
        "prj_name": "Đầy đủ",
        "noi_that": "Đông",
        "huong_nha": "Đông",
        "huong_ban_cong": "missing",
        "phap_ly": "Sổ đỏ/ Sổ hồng",
        "long": 10.73862361907959,
        "lat": 106.70059967041016,
        "duong": 'duong-nguyen-huu-tho',
        "xa": 'phuong-tan-hung-14',
        "huyen": 'Quận 7',
        "tinh": 'Hồ Chí Minh',
        "url": '',
        "source": 'bds'
    }]
    path_model = r'/home/phuonghuu/Phuong_WorkSpace/Apartment-Price-Prediction/model/model.joblib'
    path_loc = r'/home/phuonghuu/Phuong_WorkSpace/Apartment-Price-Prediction/resources/loc.csv'
    X = pd.DataFrame.from_dict(data_input, orient='columns')
    X = convert_data(X, path_loc)
    X = convert_dataframe(X)
    gbm_model = load(path_model)
    results = predict(gbm_model, X)
    for i in range(len(X)):
        print("Input: ", data_input)
        print("price predict:", results[i])

if __name__ == '__main__':
    main()