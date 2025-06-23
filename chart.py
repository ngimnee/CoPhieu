import pandas as pd
import matplotlib.pyplot as plt
from model_utils import train_model
from data_utils import load_data

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