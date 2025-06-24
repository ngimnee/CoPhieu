import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from data_utils import append_new_data, load_data
from model_utils import train_model, du_doan_xu_huong
from chart import bieu_do_so_sanh, bieu_do_du_bao

# Hàm tạo giao diện
def create_gui(predict_callback, forecast_callback, back_callback, mae, rmse, r2):
    window = tk.Tk()
    window.title("Dự báo giá đóng cửa cổ phiếu FPT")
    window.configure(bg="white")

    # Tạo khung chính để chứa các phần: form nhập và biểu đồ
    main_frame = ttk.Frame(window)
    main_frame.pack(padx=10, pady=10, fill='both', expand=True)

    # Khung bên trái: form nhập liệu
    form_frame = ttk.Frame(main_frame)
    form_frame.grid(row=0, column=0, sticky="nw")

    # Khung bên phải: biểu đồ
    plot_frame = ttk.Frame(main_frame)
    plot_frame.grid(row=0, column=1, sticky="ne", padx=20)

    # Cấu hình font chữ
    style = ttk.Style()
    style.configure("TLabel", font=("Arial", 11), background="white")
    style.configure("TButton", font=("Arial", 11, "bold"))
    style.configure("TEntry", font=("Arial", 11))

    # Tạo các ô nhập liệu và nhãn tương ứng
    entries = {}  # Dictionary để lưu các Entry widget
    labels = ["Giá mở cửa", "Giá cao nhất", "Giá thấp nhất", "KL giao dịch (M)", "% Thay đổi"]
    for i, label in enumerate(labels):
        ttk.Label(form_frame, text=label + ":").grid(row=i, column=0, sticky="e", padx=10, pady=5)
        entry = ttk.Entry(form_frame)
        entry.grid(row=i, column=1, padx=10, pady=5)
        entries[label] = entry

    # Label hiển thị kết quả dự đoán
    result_label = ttk.Label(form_frame, text="")
    result_label.grid(row=6, column=0, columnspan=2, pady=10)

    # Label hiển thị các chỉ số đánh giá mô hình
    metrics_label = ttk.Label(
        form_frame,
        text=f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}"
    )
    metrics_label.grid(row=7, column=0, columnspan=2, pady=5)

    # Tạo frame chứa 2 nút ngang hàng
    button_row = ttk.Frame(form_frame)
    button_row.grid(row=8, column=0, columnspan=2, pady=10)

    # Nút dự đoán thủ công
    ttk.Button(
        form_frame,
        text="Dự đoán",
        command=predict_callback
    ).grid(row=5, column=0, columnspan=2, pady=10)

    # Nút dự báo 10 ngày tới
    ttk.Button(
        button_row,
        text="Dự đoán xu hướng",
        command=forecast_callback
    ).pack(side="left", padx=5)

    # Nút xem biểu đồ so sánh
    ttk.Button(
        button_row,
        text="Biểu đồ so sánh",
        command=back_callback
    ).pack(side="left", padx=5)

    forecast_plot_frame = ttk.Frame(main_frame)
    forecast_plot_frame.grid(row=1, column=0, columnspan=2, pady=10)

    return window, entries, result_label, metrics_label, plot_frame


# Hàm xử lý khi người dùng nhấn nút Dự đoán
def predict(entries, result_label, metrics_label, plot_frame, model, scaler, df_data):
    try:
        # Lấy dữ liệu từ ô nhập
        mo = float(entries["Giá mở cửa"].get())
        cao = float(entries["Giá cao nhất"].get())
        thap = float(entries["Giá thấp nhất"].get())
        kl = float(entries["KL giao dịch (M)"].get()) * 1_000_000
        pt = float(entries["% Thay đổi"].get())

        # Chuẩn hóa dữ liệu đầu vào trước khi đưa vào mô hình
        data = pd.DataFrame([[mo, cao, thap, kl, pt]],
                            columns=['Mo', 'Cao', 'Thap', 'KL', 'Phan_tram'])
        data_scaled = scaler.transform(data)

        # Dự đoán giá đóng cửa từ mô hình đã huấn luyện và lưu
        ket_qua = round(model.predict(data_scaled)[0], 2)
        result_label.config(text=f"Giá đóng cửa dự đoán: {ket_qua}", foreground="blue")
        append_new_data(mo, cao, thap, kl, pt, ket_qua)

        # Nạp lại toàn bộ dữ liệu để huấn luyện lại mô hình
        df_data = load_data()
        model, scaler, mae, rmse, r2, y_test, y_pred = train_model(df_data)
        metrics_label.config(text=f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

        # Sau khi train lại model:
        global du_bao_cache
        du_bao_cache = None  # Xóa cache để dự đoán lại lần sau

        # Hiển thị biểu đồ xu hướng dự đoán 10 ngày tới
        forecast_10_days(plot_frame, model, scaler, df_data)
        return model, scaler, mae, rmse, r2, y_test, y_pred, df_data

    except Exception as e:
        result_label.config(text=f"Lỗi: {e}", foreground="red")
        return model, scaler, mae, rmse, r2, y_test, y_pred, df_data


du_bao_cache = None  # Cache xu hướng

# Hàm hiển thị biểu đồ dự báo xu hướng giá cổ phiếu trong 10 ngày tiếp theo
def forecast_10_days(plot_frame, model, scaler, df_data):
    global du_bao_cache
    if du_bao_cache is None:
        print("→ Tính toán xu hướng mới...")
        du_bao_cache = du_doan_xu_huong(model, scaler, df_data, n=10)   # Gọi mô hình để dự đoán 10 ngày tiếp theo
    # Vẽ biểu đồ dựa trên kết quả dự đoán
    render_chart(lambda ax: bieu_do_du_bao(ax, du_bao_cache), plot_frame)


# Hàm vẽ lại biểu đồ so sánh
def reshow_so_sanh(plot_frame, y_test, y_pred):
    render_chart(lambda ax: bieu_do_so_sanh(ax, y_test, y_pred, n=30), plot_frame)


def render_chart(ax_func, plot_frame):
    # Xóa toàn bộ widget cũ trong vùng hiển thị biểu đồ
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Tạo một biểu đồ matplotlib với kích thước cố định
    fig, ax = plt.subplots(figsize=(8, 4))
    ax_func(ax)

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Giải phóng bộ nhớ: đóng figure sau khi nhúng
    plt.close(fig)
