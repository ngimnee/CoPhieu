import pandas as pd
import numpy as np
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


def ve_bieu_do(ax, y_test, y_pred, n=15):
    x_range = range(n)
    ax.plot(x_range, y_test[-n:], label='Giá thực tế', color='#1f77b4', linewidth=2, marker='o', markersize=4)
    ax.plot(x_range, y_pred[-n:], label='Giá dự đoán', color='#ff7f0e', linestyle='--', linewidth=2, marker='x', markersize=4)
    ax.set_title(f"Giá thực tế và giá dự đoán trong {n} ngày", fontsize=12)
    ax.set_ylabel("Giá đóng cửa")
    ax.set_xlabel("Ngày")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.figure.tight_layout()
