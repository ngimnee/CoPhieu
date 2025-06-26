import pandas as pd
import matplotlib.pyplot as plt
from model_utils import train_model
from data_utils import load_data_goc
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Hàm vẽ biểu đồ so sánh
def bieu_do_so_sanh(ax, y_test, y_pred, n=30):
    x_range = range(n)
    ax.plot(x_range, list(y_test)[:n], label="Giá thực tế")
    ax.plot(x_range, list(y_pred)[:n], label="Giá dự đoán")
    ax.set_title(f"Giá thực tế và giá dự đoán trong {n} ngày", fontsize=12)
    ax.set_ylabel("Giá đóng cửa")
    ax.set_xlabel("Ngày")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.figure.tight_layout()


# Hàm vẽ biểu đồ dự báo xu hướng
def bieu_do_du_bao(ax, gia_du_bao, df_data):
    # Lấy giá đóng cửa của ngày cuối cùng từ dữ liệu lịch sử
    last_historical_price = df_data['Dong_cua'].iloc[0]
    # Kết hợp điểm lịch sử và các điểm dự đoán
    prices = [last_historical_price] + gia_du_bao
    x_range = list(range(0, len(prices)))  # Bắt đầu từ 0 để bao gồm điểm lịch sử

    # Vẽ điểm lịch sử (màu đỏ) và các điểm dự đoán (màu xanh lá)
    ax.plot(x_range[:1], prices[:1], marker='o')
    ax.plot(x_range[1:], prices[1:], linestyle='--')

    ax.set_title("Xu hướng cổ phiếu", fontsize=12)
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá đóng cửa")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.figure.tight_layout()
