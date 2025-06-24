'''
    pip install pandas numpy scikit-learn matplotlib
'''

import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data_utils import load_data, append_new_data
from model_utils import train_model, du_doan_xu_huong
from chart import bieu_do_so_sanh, bieu_do_du_bao, show_bieu_do_so_sanh
from ui_controller import create_gui, predict, forecast_10_days, reshow_so_sanh

# Tải dữ liệu từ file và huấn luyện mô hình ban đầu
df_data = load_data()
model, scaler, mae, rmse, r2, y_test, y_pred = train_model(df_data)

# Tạo hàm callback phù hợp với giao diện GUI
predict_callback = lambda: update_predict()
forecast_callback = lambda: forecast_10_days(plot_frame, model, scaler, df_data)
back_callback = lambda: reshow_so_sanh(plot_frame, y_test, y_pred)

# Sau mỗi lần nhấn nút "Dự đoán", gọi predict và cập nhật các biến toàn cục
def update_predict():
    global model, scaler, mae, rmse, r2, y_test, y_pred, df_data
    model, scaler, mae, rmse, r2, y_test, y_pred, df_data = predict(
        entries, result_label, metrics_label, plot_frame, model, scaler, df_data
    )

# Tạo GUI từ các callback
window, entries, result_label, metrics_label, plot_frame = create_gui(predict_callback, forecast_callback, back_callback, mae, rmse, r2)

# Hiển thị biểu đồ so sánh
fig, ax = plt.subplots(figsize=(8, 4))
bieu_do_so_sanh(ax, y_test, y_pred, n=15)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.draw()
canvas.get_tk_widget().pack()

# Chạy chương trình
window.mainloop()