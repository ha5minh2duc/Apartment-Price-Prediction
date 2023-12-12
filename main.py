from modelling import read_file_csv, evaluate
from utils import load_data
from joblib import  load, dump


def main():
    path_data = r'C:\Users\biennn1\Downloads\housing_master_v2-20231211T011230Z-001\housing_master_v2\data_output\bds_processed_0512.csv'
    path_model = r'C:\Users\biennn1\Downloads\housing_master_v2-20231211T011230Z-001\housing_master_v2\model\model.joblib'
    X, y = load_data(path_data)
    gbm_model = load(path_model)
    results = evaluate(gbm_model, X, y)
    print(results)

if __name__ == '__main__':
    main()