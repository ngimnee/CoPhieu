import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data_utils import load_data, append_new_data
from model_utils import train_model
from chart import bieu_do_so_sanh
from ui import create_gui

# Tải dữ liệu từ file và huấn luyện mô hình ban đầu
df_data = load_data()
model, scaler, mae, rmse, r2, y_test, y_pred = train_model(df_data)


# Hàm xử lý khi người dùng nhấn nút "Dự đoán"
def predict():
    global model, scaler, mae, rmse, r2, y_test, y_pred, df_data
    # Đọc dữ liệu người dùng nhập từ các ô Entry
    try:
        mo = float(entries["Giá mở cửa"].get())
        cao = float(entries["Giá cao nhất"].get())
        thap = float(entries["Giá thấp nhất"].get())
        kl = float(entries["KL giao dịch (M)"].get()) * 1_000_000
        pt = float(entries["% Thay đổi"].get())

        # Tạo DataFrame từ input để đưa vào mô hình
        data = pd.DataFrame([[mo, cao, thap, kl, pt]], columns=['Mo', 'Cao', 'Thap', 'KL', 'Phan_tram'])
        # Chuẩn hóa dữ liệu đầu
        data_scaled = scaler.transform(data)
        # Dự đoán giá đóng cửa
        ket_qua = round(model.predict(data_scaled)[0], 2)

        # Hiển thị và lưu kết quả
        result_label.config(text=f"Giá đóng cửa dự đoán: {ket_qua}", foreground="blue")
        append_new_data(mo, cao, thap, kl, pt, ket_qua)

        # Cập nhật lại mô hình với dữ liệu mới
        df_data = load_data()
        model, scaler, mae, rmse, r2, y_test, y_pred = train_model(df_data)
        metrics_label.config(text=f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    except Exception as e:
        result_label.config(text=f"Lỗi: {e}", foreground="red")

# Tạo GUI từ form
window, entries, result_label, metrics_label, plot_frame = create_gui(predict, mae, rmse, r2)

# Vẽ biểu đồ
fig, ax = plt.subplots(figsize=(8, 4))
bieu_do_so_sanh(ax, y_test, y_pred, n=15)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.draw()
canvas.get_tk_widget().pack()

# Chạy chương trình
window.mainloop()