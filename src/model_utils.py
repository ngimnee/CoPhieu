import pandas as pd
import numpy as np
import random
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model(df, input_cols, target_col):
    if df.empty or target_col not in df.columns or not all(col in df.columns for col in input_cols):
        raise ValueError("Dữ liệu đầu vào không hợp lệ hoặc thiếu cột.")

    df_train = df.copy()
    # Tạo biến mục tiêu: giá đóng cửa ngày hôm sau
    df_train['Target'] = df_train[target_col].shift(-1)
    # Loại bỏ các giá trị NaN
    df_train = df_train.dropna()

    X = df_train[input_cols]  # Lấy các đặc trưng
    y = df_train['Target']    # Lấy mục tiêu
    # Chia tập train/test
    train_size = int(len(X) * 0.8)  # Kích thước tập train (80%)
    X_train = X[:train_size]        # Tập train đặc trưng
    X_test = X[train_size:]         # Tập test đặc trưng
    y_train = y[:train_size]        # Tập train mục tiêu
    y_test = y[train_size:]         # Tập test mục tiêu

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=input_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=input_cols)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, scaler, mae, rmse, r2, y_test, y_pred

def train_mo_model(df):
    return train_model(df, ['Dong_cua', 'Phan_tram', 'KL'], 'Mo')

def train_cao_model(df):
    df_temp = df.copy()
    df_temp['Chenh_lech'] = df_temp['Cao'] - df_temp['Thap']
    return train_model(df_temp, ['Mo', 'Dong_cua', 'Cao', 'Chenh_lech'], 'Cao')

def train_thap_model(df):
    df_temp = df.copy()
    df_temp['Chenh_lech'] = df_temp['Cao'] - df_temp['Thap']
    return train_model(df_temp, ['Mo', 'Dong_cua', 'Thap', 'Chenh_lech'], 'Thap')

def train_kl_model(df):
    df_temp = df.copy()
    df_temp['KL_TB7'] = df_temp['KL'].rolling(window=7).mean()
    df_temp.dropna(inplace=True)
    return train_model(df_temp, ['Mo', 'Cao', 'Thap', 'KL_TB7'], 'KL')

def train_phan_tram_model(df):
    return train_model(df, ['Mo', 'Cao', 'Thap', 'KL'], 'Phan_tram')

def train_dong_cua_model(df):
    return train_model(df, ['Mo', 'Cao', 'Thap', 'KL', 'Phan_tram'], 'Dong_cua')

def predict(model, scaler, input_data, columns):
    input_df = pd.DataFrame([input_data], columns=columns)
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=columns)
    return model.predict(input_scaled_df)[0]


def du_bao_xu_huong(df_data, n=5):
    if df_data.empty:
        raise ValueError("Dữ liệu đầu vào rỗng.")

    df = df_data.copy()
    df['Ngay'] = pd.to_datetime(df['Ngay'], format='%d/%m/%Y')
    df = df[df['Ngay'] < datetime.today()].copy()

    ket_qua = []

    # Huấn luyện mô hình
    model_mo, scaler_mo, *_ = train_mo_model(df)
    model_cao, scaler_cao, *_ = train_cao_model(df)
    model_thap, scaler_thap, *_ = train_thap_model(df)
    model_kl, scaler_kl, *_ = train_kl_model(df)
    model_pt, scaler_pt, *_ = train_phan_tram_model(df)
    model_dc, scaler_dc, *_ = train_dong_cua_model(df)

    # Sử dụng ngày gần nhất để bắt đầu dự báo
    last_row = df.iloc[0]

    mean_price = df['Dong_cua'].mean()
    std_price = df['Dong_cua'].std()
    mean_range = (df['Cao'] - df['Thap']).mean()

    for i in range(n):
        chenh_lech = last_row['Cao'] - last_row['Thap']
        mo = predict(model_mo, scaler_mo, [last_row['Dong_cua'], last_row['Phan_tram'], last_row['KL']], ['Dong_cua', 'Phan_tram', 'KL'])
        cao = predict(model_cao, scaler_cao, [mo, last_row['Dong_cua'], last_row['Cao'], chenh_lech], ['Mo', 'Dong_cua', 'Cao', 'Chenh_lech'])
        thap = predict(model_thap, scaler_thap, [mo, last_row['Dong_cua'], last_row['Thap'], chenh_lech], ['Mo', 'Dong_cua', 'Thap', 'Chenh_lech'])
        kl = predict(model_kl, scaler_kl, [mo, cao, thap, last_row['KL']], ['Mo', 'Cao', 'Thap', 'KL_TB7'])
        pt = predict(model_pt, scaler_pt, [mo, cao, thap, kl], ['Mo', 'Cao', 'Thap', 'KL'])
        dong_cua = predict(model_dc, scaler_dc, [mo, cao, thap, kl, pt], ['Mo', 'Cao', 'Thap', 'KL', 'Phan_tram'])

        # Điều chỉnh
        dong_cua = dieu_chinh(ket_qua, dong_cua, i, mean_price, std_price)

        cao = min(cao, dong_cua + mean_range)
        thap = max(thap, dong_cua - mean_range)

        dong_cua = round(dong_cua, 2)
        ket_qua.append(dong_cua)

        last_row = pd.Series({
            'Dong_cua': dong_cua,
            'Mo': mo,
            'Cao': cao,
            'Thap': thap,
            'KL': kl,
            'Phan_tram': pt
        })
    return ket_qua

def dieu_chinh(ket_qua, gia_moi, i, mean_price, std_price):
    bien_do_ngay = 0.012  # giảm biên độ mỗi ngày
    bien_do_giam_nhe = 0.003  # yếu tố giảm chậm hơn
    bien_do_pha_dao = 0.006   # đảo chiều nhẹ hơn nữa
    min_pha = 3  # số phiên để đảo chiều
    giam_toi_da = 0.008  # nếu đang giảm, không cho giảm quá mức

    # Làm trơn bằng EMA 3 ngày
    if len(ket_qua) >= 2:
        gia_moi = 0.65 * gia_moi + 0.25 * ket_qua[-1] + 0.1 * ket_qua[-2]
    elif len(ket_qua) == 1:
        gia_moi = 0.8 * gia_moi + 0.2 * ket_qua[-1]

    # Giảm nhẹ theo thời gian để không tăng mạnh về sau
    gia_moi *= 1 - bien_do_giam_nhe * (i - 1)

    # Giới hạn dao động trong ngày ±1.2%
    if ket_qua:
        prev = ket_qua[-1]
        gia_moi = max(min(gia_moi, prev * (1 + bien_do_ngay)), prev * (1 - giam_toi_da))

    # Nếu tăng hoặc giảm liên tiếp → điều chỉnh hướng đi
    if len(ket_qua) >= min_pha + 1:
        deltas = [ket_qua[-k] - ket_qua[-k-1] for k in range(1, min_pha+1)]
        if all(d < 0 for d in deltas):
            gia_moi = max(gia_moi, ket_qua[-1] * (1 + bien_do_pha_dao))
        elif all(d > 0 for d in deltas):
            gia_moi = min(gia_moi, ket_qua[-1] * (1 - bien_do_pha_dao))

    # Giới hạn tuyệt đối trong khoảng thống kê
    gia_moi = min(max(gia_moi, mean_price - 2 * std_price),
                  mean_price + 2 * std_price)

    return round(gia_moi, 2)
