import pandas as pd
import unidecode
import re
import gc
gc.enable()
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_predict as cvp



LIST_FILE_DATA = ['2023-12-03_11_01_04_bds.csv', '2023-12-03_11_40_50_bds.csv', '2023-12-03_12_21_32_bds.csv',
                  '2023-12-03_12_50_00_bds.csv', '2023-12-03_13_37_53_bds.csv', '2023-12-03_14_35_01_bds.csv',
                  '2023-12-03_15_04_07_bds.csv', '2023-12-03_15_34_14_bds.csv', '2023-12-03_16_05_22_bds.csv',
                  '2023-12-03_16_36_04_bds.csv', '2023-12-03_17_02_43_bds.csv', '2023-12-03_17_31_40_bds.csv',
                  '2023-12-03_18_14_26_bds.csv', '2023-12-03_18_44_10_bds.csv', '2023-12-03_19_12_21_bds.csv',
                  '2023-12-03_19_32_12_bds.csv', '2023-12-03_20_59_39_bds.csv', '2023-12-03_21_36_56_bds.csv',
                  '2023-12-03_22_19_14_bds.csv', '2023-12-03_23_57_06_bds.csv', '2023-12-04_00_21_07_bds.csv',
                  '2023-12-04_01_18_30_bds.csv', '2023-12-04_01_57_30_bds.csv', '2023-12-04_02_39_23_bds.csv',
                  '2023-12-04_04_33_45_bds.csv', '2023-12-04_05_59_13_bds.csv', '2023-12-04_06_24_27_bds.csv',
                  '2023-12-04_06_50_35_bds.csv', '2023-12-04_07_16_54_bds.csv', '2023-12-04_07_47_04_bds.csv',
                  '2023-12-04_08_15_52_bds.csv', '2023-12-04_08_42_12_bds.csv', '2023-12-04_09_10_28_bds.csv',
                  '2023-12-04_09_37_43_bds.csv', '2023-12-04_10_06_31_bds.csv', '2023-12-04_10_36_01_bds.csv',
                  '2023-12-04_11_07_26_bds.csv', '2023-12-04_11_32_06_bds.csv', '2023-12-04_12_13_15_bds.csv',
                  '2023-12-04_12_43_00_bds.csv', '2023-12-04_13_14_04_bds.csv', '2023-12-04_21_00_30_bds.csv',
                  '2023-12-04_21_34_54_bds.csv', '2023-12-04_22_11_12_bds.csv', '2023-12-04_22_50_32_bds.csv',
                  '2023-12-04_23_22_11_bds.csv', '2023-12-04_23_57_06_bds.csv', '2023-12-05_00_30_14_bds.csv',
                  '2023-12-05_01_01_10_bds.csv', '2023-12-05_01_33_11_bds.csv', '2023-12-05_02_01_56_bds.csv',
                  '2023-12-05_02_36_24_bds.csv', '2023-12-05_03_03_23_bds.csv', '2023-12-05_03_30_25_bds.csv',
                  '2023-12-05_03_58_17_bds.csv', '2023-12-05_04_30_01_bds.csv', '2023-12-05_04_55_17_bds.csv',
                  '2023-12-05_05_22_10_bds.csv', '2023-12-05_05_51_12_bds.csv', '2023-12-05_06_20_26_bds.csv',
                  '2023-12-05_06_50_27_bds.csv', '2023-12-05_07_20_05_bds.csv', '2023-12-05_07_43_01_bds.csv',
                  '2023-12-05_08_12_07_bds.csv']

def read_file_csv(path_file):
    data = pd.read_csv(path_file)
    return data

def concat_data(list_data_csv):
    mul_data = pd.concat(list_data_csv, axis=0)
    return mul_data

def save_to_csv(data, save_data):
    pd.save_to_csv(save_data, data)

def lowercase_transform(feature):
    from unidecode import unidecode
    feature = feature.map(lambda x: str(x)).map(lambda x: unidecode(x))
    feature = feature.fillna("missing").str.lower().str.strip().str.replace('\s+', '-').str.replace('-+', '-').str.replace("\(.+", "").str.strip('-')
    return feature

def get_lowercase_df(df):
    df = df.rename(columns={c:c.lower() for c in df.columns})
    for col in ['prj_name', 'duong', 'xa', 'huyen', 'tinh']:
        df[col] = df[col].map(lambda x: str(x).lower())
        df[col] = df[col].map(lambda x: unidecode.unidecode(str(x)).replace(' ', '-'))
        df[col] = df[col].str.replace("-+", '-')
    return df

def get_loc(path_loc):
    DICT_TYPE = {'TINH': 'category', 
                 'REF_TINH_CODE': 'category',
                 'REF_HUYEN_CODE': 'category',
                 'HUYEN': 'category',
                 'REF_XA_CODE': 'category',
                 'XA': 'category'}
    loc = pd.read_csv(path_loc, delimiter='|', dtype=DICT_TYPE)
    loc = loc.rename(columns={c:c.lower() for c in loc.columns})
    for col in ['xa', 'huyen', 'tinh']:
        loc[col] = loc[col].map(lambda x: str(x).lower())
        loc[col] = loc[col].map(lambda x: unidecode.unidecode(str(x)).replace(' ', '-'))
        loc[col] = loc[col].str.replace('-+', '-')
    return loc

def map_unique_xa(df, loc):
    def from_xa_map_huyen(x):
        current_huyen = x[0]
        loc_huyen = x[1]
        if loc_huyen == loc_huyen:
            return loc_huyen
        if current_huyen == "huyen-phu-quoc":
            return "thanh-pho-phu-quoc"
        if current_huyen == "thanh-pho-phan-rang":
            return "thanh-pho-phan-rang-thap-cham"
        if current_huyen == "quan-thanh-tri":
            return "huyen-thanh-tri"
        return current_huyen

    tt = loc.groupby('xa')[['huyen', 'tinh']].size()
    tt.name = 'count_xa'
    tt.index.name = 'xa'
    tt = tt.reset_index()
    loc2 = pd.merge(loc, tt, how='left', on ='xa')
    loc2 = loc2[loc2.count_xa==1]
    df = pd.merge(df, loc2, how='left', on ='xa' ,suffixes=["", "_y"])
    df['huyen'] = df[['huyen', 'huyen_y']].apply(lambda x: from_xa_map_huyen(x), axis=1)
    drop_cols = [c for c in df.columns if c.endswith('_y')]
    df = df.drop(columns=drop_cols)
    return df

def map_xa(xa):
    if xa == "thi-tran-an-thoi":
        return "phuong-an-thoi"
    if xa == "phuong-tan-phu-19":
        return "phuong-tan-phu"
    if xa == "phuong-tan-hung-14":
        return "phuong-tan-hung"
    if xa == "phuong-phu-thuong-1":
        return "phuong-phu-thuong"
    if xa == "phuong-phu-thuan-3":
        return "phuong-phu-thuan"
    if xa == "phuong-phu-huu-2":
        return "phuong-phu-huu"
    if xa == "phuong-trung-hoa-4":
        return "phuong-trung-hoa"
    if xa == "phuong-tan-phong-9":
        return "phuong-tan-phong"
    if xa == "phuong-phu-my-9":
        return "phuong-phu-my"
    if xa == "phuong-yen-hoa-2":
        return "phuong-yen-hoa"
    if xa == "thi-tran-cau-dien":
        return "phuong-cau-dien"
    if xa == "phuong-an-lac-6":
        return "phuong-an-lac"
    if xa == "phuong-hoang-van-thu-4":
        return "phuong-hoang-van-thu"
    if xa == "phuong-2-22":
        return "phuong-2"
    if xa == "phuong-14-5":
        return "phuong-14"
    if xa == "phuong-an-lac-6":
        return "phuong-an-lac"
    if xa == "phuong-9-12":
        return "phuong-9"
    if xa == "phuong-4-15":
        return "phuong-4"
    if xa == "bo-de":
        return "phuong-bo-de"
    if xa == "vinh-tuy":
        return "phuong-vinh-tuy"
    if xa == "phu-thuong":
        return "phuong-phu-thuong"
    return xa

def map_xa_version2(xa):
    xa = str(xa)
    if xa == "missing":
        return xa
    if not re.search("^(xa|phuong|thi-tran)", xa):
        xa = "phuong-" + xa
    if re.search(".+-\d+-\d+", xa):
        xa = re.sub("-\d+$", "", xa)
    return xa

def __map_xa_huyen_tinh__(x):
    xa = str(x[0])
    huyen = str(x[1])
    tinh = str(x[2])
    HN = "thanh-pho-ha-noi"
    HCM = "thanh-pho-ho-chi-minh"
    if xa == "phuong-long-binh":
        tinh = HCM
    if xa == "phuong-dong-hoa":
        tinh = "tinh-binh-duong"
        huyen = "thanh-pho-di-an"
    if xa == "phuong-my-binh":
        tinh = "tinh-ninh-thuan"
    if xa == "phuong-an-thoi":
        tinh = "tinh-kien-giang"
    if xa == "phuong-an-phu":
        huyen = "thanh-pho-thu-duc"
        tinh = HCM
    if huyen == "vinh-tuy":
        xa = "phuong-" + huyen
        huyen = "quan-hai-ba-trung"
        tinh = HN
    elif huyen in ("my-dinh-2", "me-tri" ,"tay-mo", "my-dinh-1", "dai-mo"):
        xa = "phuong-" + huyen
        huyen = "quan-nam-tu-liem"
        tinh = HN
    elif huyen in ("thanh-xuan-trung" ,"nhan-chinh", "thuong-dinh") :
        xa ="phuong-" + huyen
        huyen = "quan-thanh-xuan"
        tinh = HN
    elif huyen in ("phu-thuong-1", "xuan-la", "thuy-khue"):
        xa = "phuong-" + huyen
        if xa[-1].isdigit():
            xa = xa[:-2]
        huyen = "quan-tay-ho"
        tinh = HN
    elif huyen in ("hoang-liet", "dai-kim", "dinh-cong", "yen-so"):
        xa = "phuong-"+huyen
        huyen = "quan-hoang-mai"
        tinh = HN
    elif huyen in ("trung-hoa-4"):
        xa = "phuong-trung-hoa"
        huyen = "quan-cau-giay"
        tinh = HN
    elif huyen in ("phu-dien", "dong-ngac", "xuan-dinh", "xuan-tao"):
        xa = "phuong-"+huyen
        huyen == "quan-bac-tu-liem"
        tinh = HN
    elif huyen == "yen-hoa-2":
        xa = "phuong-yen-hoa"
        huyen = "quan-cau-giay"
        tinh = HN
    elif huyen in ("la-khe", "phuc-la", "phu-la", "ha-cau", "yen-nghia-1", "yet-kieu-2"):
        xa = "phuong-"+huyen
        if xa[-1].isdigit():
            xa = xa[:-2]
        huyen = "quan-ha-dong"
        tinh = HN
    elif huyen in ("bo-de", "sai-dong", "duc-giang-2"):
        xa = "phuong-"+huyen
        if xa[-1].isdigit():
            xa  = xa[:-2]
        huyen = "quan-long-bien"
        tinh = HN
    elif huyen in ("dich-vong-hau"):
        xa = "phuong-"+huyen
        huyen = "quan-cau-giay"
        tinh = HN
    elif huyen == "hoang-van-thu-4":
        xa = "phuong-hoang-van-thu"
        huyen = "quan-hoang-mai"
        tinh = HN
    elif huyen in ("tan-trieu", "tu-hiep", "ta-thanh-oai"):
        xa = "phuong-"+huyen
        huyen = "huyen-thanh-tri"
        tinh = HN
    elif huyen in ("ngoc-khanh", "giang-vo"):
        xa = "phuong-"+huyen
        huyen = "quan-ba-dinh"
        tinh = HN
    elif huyen in ("trung-tu", "lang-ha"):
        xa = "phuong-"+huyen
        huyen = "quan-dong-da"
        tinh = HN
    elif huyen in ("an-khanh-4"):
        xa = "xa-"+huyen
        if xa[-1].isdigit():
            xa = xa[:-2]
        huyen = "huyen-hoai-duc"
        tinh = HN
    return xa, huyen, tinh

def map_xa_huyen_tinh(df):
    this_map = df[['xa', 'huyen', 'tinh']].apply(lambda x: __map_xa_huyen_tinh__(x), axis=1)
    df['xa'] = this_map.map(lambda x: x[0])
    df['huyen'] = this_map.map(lambda x: x[1])
    df['tinh'] = this_map.map(lambda x: x[2])
    return df

def drop_duplicated_records(df):
        selected_columns = ['prj_name', 'duong', 'xa', 'huyen', 'tinh', 'ref_xa_code', 'ref_huyen_code', 'ref_tinh_code', 'area', 'pn', 'source', 'price']
        df = df[selected_columns].drop_duplicates()
        return df

def select_feature_data(df):
    selected_features = ['prj_name', 'duong', 'xa', 'huyen', 'tinh', 'ref_xa_code', 'ref_huyen_code', 'ref_tinh_code', 
                         'area', 'pn', 'price', 'source']
    df = df[selected_features]
    return df

def filtering(df):
    df['unit_price'] = df.apply(lambda row: row.price/row.area, axis=1)
    LOW_THRESHOLD_PRICE = 500_000_000
    HIGH_THRESHOLD_PRICE = 20_000_000_000
    LOW_THRESHOLD_AREA = 30.0
    HIGH_THRESHOLD_AREA = 300.0 
    LOW_THRESHOLD_UNITPRICE = 10_000_000
    HIGH_THRESHOLD_UNITPRICE = 250_000_000 
    
    df = df[((df.price >= LOW_THRESHOLD_PRICE) & (df.price <= HIGH_THRESHOLD_PRICE))
            & ((df.area >= LOW_THRESHOLD_AREA) & (df.area <= HIGH_THRESHOLD_AREA))
            & ((df.unit_price >= LOW_THRESHOLD_UNITPRICE) & (df.unit_price <= HIGH_THRESHOLD_UNITPRICE))]
    df = df.query(
    "(xa==xa and ref_xa_code==ref_xa_code) or (huyen==huyen and ref_huyen_code==ref_huyen_code) or (tinh==tinh and ref_tinh_code==ref_tinh_code)")
    df = df.dropna(subset=['ref_xa_code', 'ref_huyen_code', 'ref_tinh_code'], axis=0)
    return df

def fillna_missing(df):
    df.prj_name = df.prj_name.fillna('missing')
    df.duong = df.duong.fillna('missing')
    df.xa = df.xa.fillna('missing')
    df.huyen = df.huyen.fillna('missing')
    df.tinh = df.tinh.fillna('missing')
    return df

def preprocessing_data(df, path_loc):
    df = get_lowercase_df(df)
    loc = get_loc(path_loc)
    data = df.copy()
    data['xa'] = data['xa'].map(lambda x: map_xa(x))
    list_incorrect_xa = data[data.xa.isin(loc.xa)].xa.unique()
    data['xa'] = data['xa'].map(lambda x: map_xa_version2(x))
    data = map_xa_huyen_tinh(data)
    data = map_unique_xa(data, loc)
    assert len(data.columns)==len(set(data.columns)), f"{data.columns}"
    data = data[df.columns]
    keys = ['xa', 'huyen']
    loc = loc.drop_duplicates(subset=keys)
    assert loc[keys].duplicated().sum()==0   
    data = pd.merge(data, loc, how="left", on = keys)
    data = data.rename(columns={'tinh_y': 'tinh'})
    del data['tinh_x']
    data = drop_duplicated_records(data)
    return data

def main():
    path_loc = r'C:\Users\biennn1\Downloads\housing_master_v2-20231211T011230Z-001\housing_master_v2\resources\loc.csv'
    # path_data = ''
    # data_csv = []
    # for file_csv in LIST_FILE_DATA:
    #     path_file_csv = os.path.join(path_data, file_csv)
    #     df = read_file_csv(path_file_csv)
    #     data_csv.append(df)
    # dataframe = concat_data(data_csv)
    dataframe = pd.read_csv(r'C:\Users\biennn1\Downloads\housing_master_v2-20231211T011230Z-001\housing_master_v2\resources\bds_1112\bds_1112.csv')
    dataframe_new = preprocessing_data(dataframe, path_loc)
    df = select_feature_data(dataframe_new)
    df = filtering(df)
    df = fillna_missing(df)
    df.to_csv(r'C:\Users\biennn1\Downloads\housing_master_v2-20231211T011230Z-001\housing_master_v2\data_output')
    return df

if __name__ == '__main__':
    main()

