"""
    Notes pip install:
    pip install pandas numpy scikit-learn matplotlib
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import os

DATA_FILE = "data/du_lieu_fpt.csv"
USER_DATA_FILE = "data/du_lieu_du_doan.csv"


# Hàm chuyển đổi chuỗi thành số, xử lý các định dạng như 'M', '%', '.', ','
def chuyen_doi_so(chuoi):
    if isinstance(chuoi, str):
        chuoi = chuoi.replace('M', '').replace('%', '').replace('.', '').replace(',', '.')
    try:
        return float(chuoi)
    except:
        return None


# Hàm đọc và xử lý dữ liệu từ tệp CSV
def load_data():
    df = pd.read_csv(DATA_FILE)
    # Nếu có tệp dữ liệu dự đoán, hợp nhất với dữ liệu gốc
    if os.path.exists(USER_DATA_FILE):
        df_user = pd.read_csv(USER_DATA_FILE)
        df = pd.concat([df, df_user], ignore_index=True)
    # Lấy các cột cần thiết từ dữ liệu
    df = df.iloc[:, [0, 1, 2, 3, 4, 5, 6]]
    # Đặt tên cột cho dữ liệu
    df.columns = ['Ngay', 'Dong_cua', 'Mo', 'Cao', 'Thap', 'KL', 'Phan_tram']

    # Áp dụng hàm chuyển đổi cho các cột số
    for cot in ['Dong_cua', 'Mo', 'Cao', 'Thap', 'KL', 'Phan_tram']:
        df[cot] = df[cot].apply(chuyen_doi_so)
    # Loại bỏ các hàng có giá trị thiếu
    df.dropna(inplace=True)
    # Lọc dữ liệu, chỉ giữ giá đóng cửa dưới 1000
    df = df[df['Dong_cua'] < 1000]
    return df


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
    mae = mean_absolute_error(y_test, y_pred)  # Sai số tuyệt đối trung bình
    mse = mean_squared_error(y_test, y_pred)  # Sai số bình phương trung bình
    rmse = np.sqrt(mse)  # Căn bậc hai của MSE
    r2 = r2_score(y_test, y_pred)  # Hệ số xác định (R²)

    return model, scaler, mae, rmse, r2, y_test.reset_index(drop=True), y_pred


# Hàm lưu dữ liệu dự đoán mới vào tệp
def append_new_data(mo, cao, thap, kl, pt, ket_qua):
    try:
        # Lấy ngày hiện tại
        today = datetime.today().strftime('%Y-%m-%d')
        # Tạo DataFrame cho dữ liệu mới
        df_new = pd.DataFrame([[today, ket_qua, mo, cao, thap, kl, pt]],
                              columns=['Ngay', 'Dong_cua', 'Mo', 'Cao', 'Thap', 'KL', 'Phan_tram'])
        # Nếu tệp tồn tại, hợp nhất với dữ liệu cũ
        if os.path.exists(USER_DATA_FILE):
            df_existing = pd.read_csv(USER_DATA_FILE)
            df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_updated = df_new
        # Lưu dữ liệu vào tệp
        df_updated.to_csv(USER_DATA_FILE, index=False)
    except Exception as e:
        print("Không thể lưu dữ liệu dự đoán mới:", e)


# Tải dữ liệu và huấn luyện mô hình ban đầu
df_data = load_data()
model, scaler, mae, rmse, r2, y_test, y_pred = train_model(df_data)
y_test_short = y_test[-30:].reset_index(drop=True)  # Lấy 30 ngày dữ liệu thực tế gần nhất
y_pred_short = pd.Series(y_pred[-30:]).reset_index(drop=True)  # Lấy 30 ngày dự đoán gần nhất

# Tạo giao diện tkinter
window = tk.Tk()
window.title("Dự báo giá đóng cửa cổ phiếu FPT")
window.configure(bg="white")

main_frame = ttk.Frame(window)
main_frame.pack(padx=10, pady=10, fill='both', expand=True)

form_frame = ttk.Frame(main_frame)
form_frame.grid(row=0, column=0, sticky="nw")  # Frame cho form nhập liệu
plot_frame = ttk.Frame(main_frame)
plot_frame.grid(row=0, column=1, sticky="ne", padx=20)  # Frame cho biểu đồ

style = ttk.Style()
style.configure("TLabel", font=("Arial", 11), background="white")
style.configure("TButton", font=("Arial", 11, "bold"))
style.configure("TEntry", font=("Arial", 11))

entries = {}  # Từ điển lưu các ô nhập liệu
labels = ["Giá mở cửa", "Giá cao nhất", "Giá thấp nhất", "KL giao dịch (M)", "% Thay đổi"]
for i, label in enumerate(labels):
    ttk.Label(form_frame, text=label + ":").grid(row=i, column=0, sticky="e", padx=10, pady=5)
    entry = ttk.Entry(form_frame)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries[label] = entry

result_label = ttk.Label(form_frame, text="")
result_label.grid(row=6, column=0, columnspan=2, pady=10)

metrics_label = ttk.Label(form_frame, text=f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
metrics_label.grid(row=7, column=0, columnspan=2, pady=5)  # Hiển thị các chỉ số đánh giá


# Hàm xử lý dự đoán khi nhấn nút
def predict():
    global model, scaler, mae, rmse, r2, y_test, y_pred
    try:
        # Lấy giá trị từ các ô nhập liệu và chuyển đổi
        mo = float(entries["Giá mở cửa"].get())
        cao = float(entries["Giá cao nhất"].get())
        thap = float(entries["Giá thấp nhất"].get())
        kl = float(entries["KL giao dịch (M)"].get()) * 1_000_000  # Đổi triệu sang đơn vị thực
        pt = float(entries["% Thay đổi"].get())

        # Chuẩn bị dữ liệu và dự đoán
        data = pd.DataFrame([[mo, cao, thap, kl, pt]], columns=['Mo', 'Cao', 'Thap', 'KL', 'Phan_tram'])
        data_scaled = scaler.transform(data)
        ket_qua = round(model.predict(data_scaled)[0], 2)

        # Hiển thị kết quả và lưu dữ liệu
        result_label.config(text=f"Giá đóng cửa dự đoán: {ket_qua}", foreground="blue")
        append_new_data(mo, cao, thap, kl, pt, ket_qua)

        # Huấn luyện lại mô hình với dữ liệu cập nhật
        df_data = load_data()
        model, scaler, mae, rmse, r2, y_test, y_pred = train_model(df_data)

        # Cập nhật chỉ số đánh giá mô hình
        metrics_label.config(text=f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    except Exception as e:
        result_label.config(text=f"Lỗi: {e}", foreground="red")



ttk.Button(form_frame, text="Dự đoán", command=predict).grid(row=5, column=0, columnspan=2, pady=10)

# Vẽ biểu đồ tĩnh cho dự đoán 30 ngày tiếp theo
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(30), y_test_short[-30:], label='Giá thực tế', color='#1f77b4', linewidth=2, marker='o', markersize=4)
# Vẽ 30 ngày thực tế từ 0 đến 30
ax.plot(range(30, 60), y_pred[-30:], label='Giá dự đoán', color='#ff7f0e', linestyle='--', linewidth=2, marker='x',
        markersize=4)
# Vẽ 30 ngày dự đoán từ 30 đến 60
ax.set_title("Dự báo xu hướng giá đóng cửa (30 ngày tiếp theo)", fontsize=12)
ax.set_ylabel("Giá đóng cửa")
ax.set_xlabel("Ngày")
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
fig.tight_layout()  # Tối ưu hóa bố cục

canvas = FigureCanvasTkAgg(fig, master=plot_frame)  # Tích hợp biểu đồ với tkinter
canvas.draw()
canvas.get_tk_widget().pack()

window.mainloop()  # Chạy giao diện