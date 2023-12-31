from processing import read_file_csv
from modelling import processing_data
from sklearn.utils import shuffle



def load_data(path_data):
    data_frame = read_file_csv(path_data)
    df = processing_data(data_frame)
    global X, y
    X = df[['prj_name', 'duong', 'ref_xa_code', 'ref_huyen_code', 'ref_tinh_code', 
            'ref_xa_huyen_code', 'ref_huyen_tinh_code', 'ref_xa_tinh_code', 'ref_xa_huyen_tinh_code', 'pn', 'area']]
    y = df.unit_price
    X, y = shuffle(df[['prj_name', 'duong', 'ref_xa_code', 'ref_huyen_code', 'ref_tinh_code', 
            'ref_xa_huyen_code', 'ref_huyen_tinh_code', 'ref_xa_tinh_code', 'ref_xa_huyen_tinh_code', 'pn', 'area']], df.unit_price, random_state=22)
    return X, y