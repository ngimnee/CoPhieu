import pandas as pd
import matplotlib.pyplot as plt
from model_utils import train_model
from data_utils import load_data
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Hàm vẽ biểu đồ so sánh
def bieu_do_so_sanh(ax, y_test, y_pred, n=15):
    x_range = range(n)
    ax.plot(x_range, y_test[-n:], label='Giá thực tế', color='#1f77b4', linewidth=2, marker='o', markersize=4)
    ax.plot(x_range, y_pred[-n:], label='Giá dự đoán', color='#ff7f0e', linestyle='--', linewidth=2, marker='x', markersize=4)
    ax.set_title(f"Giá thực tế và giá dự đoán trong {n} ngày", fontsize=12)
    ax.set_ylabel("Giá đóng cửa")
    ax.set_xlabel("Ngày")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.figure.tight_layout()


# Hàm vẽ biểu đồ dự báo xu hướng
def bieu_do_du_bao(ax, gia_du_bao):
    ax.clear()
    x_range = list(range(1, len(gia_du_bao) + 1))
    ax.plot(x_range, gia_du_bao, marker='o', linestyle='--', color='green')
    ax.set_title("Xu hướng giá cổ phiếu 10 ngày tới", fontsize=12)
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá đóng cửa dự đoán")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.figure.tight_layout()


# Hàm show lại biểu đồ so sánh
def show_bieu_do_so_sanh(plot_frame, y_test, y_pred):
    # Xóa biểu đồ cũ trong plot_frame
    for widget in plot_frame.winfo_children():
        widget.destroy()

    # Vẽ lại biểu đồ so sánh
    from chart import bieu_do_so_sanh
    fig, ax = plt.subplots(figsize=(8, 4))
    bieu_do_so_sanh(ax, y_test, y_pred, n=15)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
