import pandas as pd

def read_file(path):
    path = r'/housing_master_v2/resources/bds_1112/bds_1112.csv'
    df = pd.read_csv(path)
    print(df.shape)

if __name__ == '__main__':
    path = r'/housing_master_v2/resources/bds_1112/bds_1112.csv'
    read_file(path)

