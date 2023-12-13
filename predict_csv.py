from modelling import read_file_csv, predict
from utils import load_data
from joblib import  load, dump


def main():
    path_data = r'C:\Users\huuph\OneDrive\Documents\chungcu\Apartment-Price-Prediction\data output\bds_processed_0512.csv'
    path_model = r'C:\Users\huuph\OneDrive\Documents\chungcu\Apartment-Price-Prediction\model\model.joblib'
    X, y = load_data(path_data)
    X = X.iloc[1:5]
    gbm_model = load(path_model)
    results = predict(gbm_model, X)
    for i in range(len(X)):
        print("Input: ", X.iloc[i])
        print("price predict:", results[i])
    print(results)

if __name__ == '__main__':
    main()