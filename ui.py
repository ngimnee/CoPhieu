import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from model_utils import train_model
from data_utils import load_data


# Hàm tạo giao diện
def create_gui(predict_callback, mae, rmse, r2):
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

    # Nút dự đoán thủ công
    ttk.Button(
        form_frame,
        text="Dự đoán",
        command=predict_callback
    ).grid(row=5, column=0, columnspan=2, pady=10)

    return window, entries, result_label, metrics_label, plot_frame
