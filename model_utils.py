import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Hàm huấn luyện mô hình hồi quy tuyến tính
def train_model(df):
    # Định nghĩa các đặc trưng (features) để huấn luyện
    features = ['Mo', 'Cao', 'Thap', 'KL', 'Phan_tram']
    # Tạo bản sao dữ liệu và thêm cột mục tiêu (giá đóng cửa ngày tiếp theo)
    df_scaled = df.copy()
    df_scaled['Target'] = df_scaled['Dong_cua'].shift(-1)
    df_scaled.dropna(inplace=True)

    # Phân tách đặc trưng (X) và mục tiêu (y)
    X = df_scaled[features]
    y = df_scaled['Target']

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80-20)
    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]

    # Khởi tạo và huấn luyện mô hình
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)

    # Tính các chỉ số đánh giá
    mae = mean_absolute_error(y_test, y_pred)   # Sai số tuyệt đối trung bình
    mse = mean_squared_error(y_test, y_pred)    # Sai số bình phương trung bình
    rmse = np.sqrt(mse)                         # Căn bậc hai của MSE
    r2 = r2_score(y_test, y_pred)               # Hệ số xác định (R²)

    return model, scaler, mae, rmse, r2, y_test.reset_index(drop=True), y_pred


# Hàm dự báo xu hướng
def du_doan_xu_huong(model, scaler, df_data, n=10):
    features = ['Mo', 'Cao', 'Thap', 'KL', 'Phan_tram']
    du_bao = []

    # Lấy dòng dữ liệu cuối cùng làm khởi đầu
    last_row = df_data.iloc[-1][features].values.reshape(1, -1)

    for _ in range(n):
        # Đảm bảo input có tên cột (tránh cảnh báo sklearn)
        df_last = pd.DataFrame(last_row, columns=features)
        scaled_input = scaler.transform(df_last)

        # Dự đoán giá đóng cửa
        predicted_close = model.predict(scaled_input)[0]

        # Nếu 3 ngày gần nhất đều tăng → làm chậm xu hướng
        if len(du_bao) >= 3 and du_bao[-1] > du_bao[-2] > du_bao[-3]:
            delta = predicted_close - du_bao[-1]
            predicted_close = du_bao[-1] + 0.5 * delta  # chỉ tăng 50%

        du_bao.append(predicted_close)

        # Tạo KL và % ngẫu nhiên pha trung bình
        kl_ngau_nhien = random.choice(df_data['KL'].values[-10:])
        pt_ngau_nhien = random.choice(df_data['Phan_tram'].values[-10:])
        kl_trung_binh = np.mean(df_data['KL'].values[-5:])
        pt_trung_binh = np.mean(df_data['Phan_tram'].values[-5:])

        kl = 0.5 * kl_ngau_nhien + 0.5 * kl_trung_binh
        pt = 0.5 * pt_ngau_nhien + 0.5 * pt_trung_binh

        # Tạo hàng mới dùng giá dự báo cho Mo, Cao, Thap
        new_row = [[predicted_close, predicted_close, predicted_close, kl, pt]]
        last_row = np.array(new_row)

    return du_bao
