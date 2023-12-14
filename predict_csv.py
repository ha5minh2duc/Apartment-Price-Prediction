from modelling import read_file_csv, predict
from utils import load_data
from joblib import  load, dump
from processing import preprocessing_data, select_feature_data, filtering, fillna_missing
import pandas as pd

def convert_data(df, path_loc):
    df = preprocessing_data(df, path_loc)
    df = select_feature_data(df)
    df = filtering(df)
    df = fillna_missing(df)
    return df

def main():
    path_data = r'/home/phuonghuu/Phuong_WorkSpace/Apartment-Price-Prediction/resources/bds_1112.csv'
    path_model = r'/home/phuonghuu/Phuong_WorkSpace/Apartment-Price-Prediction/model/model.joblib'
    path_loc = r'/home/phuonghuu/Phuong_WorkSpace/Apartment-Price-Prediction/resources/loc.csv'
    X = pd.read_csv(path_data)
    X = X.iloc[1:5]
    X = convert_data(X, path_loc)
    X, y = load_data(X)
    gbm_model = load(path_model)
    results = predict(gbm_model, X)
    for i in range(len(X)):
        print("Input: ", X.iloc[i])
        print("price predict:", results[i])

if __name__ == '__main__':
    main()