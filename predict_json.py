from modelling import read_file_csv, predict
from utils import load_data
from joblib import  load, dump
from utils import load_json
import pandas as pd


def main():
    data_input = [{
        "prj_name": "golden-park-tower",
        "duong": "duong-pham-van-bach",
        "ref_xa_code": 195297,
        "ref_huyen_code": 149252,
        "ref_tinh_code": 148873,
        "ref_xa_huyen_code": 195297149252,
        "ref_huyen_tinh_code": 149252148873,
        "ref_xa_tinh_code": 195297148873,
        "ref_xa_huyen_tinh_code": 195297149252148873,
        "pn": 3,
        "area": 106.0,
        "Name": 1448
    }]
    path_model = r'C:\Users\huuph\OneDrive\Documents\chungcu\Apartment-Price-Prediction\model\model.joblib'
    X = load_json(data_input)
    gbm_model = load(path_model)
    results = predict(gbm_model, X)
    for i in range(len(X)):
        print("Input: ", X.iloc[i])
        print("price predict:", results[i])

if __name__ == '__main__':
    main()