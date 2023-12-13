from modelling import read_file_csv, evaluate
from utils import load_data
from joblib import  load, dump


def main():
    path_data = r'C:\Users\huuph\OneDrive\Documents\chungcu\Apartment-Price-Prediction\data output\bds_processed_0512.csv'
    path_model = r'C:\Users\huuph\OneDrive\Documents\chungcu\Apartment-Price-Prediction\model\model.joblib'
    X, y = load_data(path_data)
    gbm_model = load(path_model)
    results = evaluate(gbm_model, X, y)
    print(results)

if __name__ == '__main__':
    main()