import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from data_utils import append_new_data, load_data
from model_utils import train_model, du_bao_xu_huong
from chart import bieu_do_so_sanh, bieu_do_du_bao

def create_gui(predict_callback, forecast_callback, back_callback, mae, rmse, r2):
    window = tk.Tk()
    window.title("Dự báo giá đóng cửa cổ phiếu FPT")
    window.configure(bg="white")

    main_frame = ttk.Frame(window)
    main_frame.pack(padx=10, pady=10, fill='both', expand=True)

    form_frame = ttk.Frame(main_frame)
    form_frame.grid(row=0, column=0, sticky="nw")

    plot_frame = ttk.Frame(main_frame)
    plot_frame.grid(row=0, column=1, sticky="ne", padx=20)

    style = ttk.Style()
    style.configure("TLabel", font=("Arial", 11), background="white")
    style.configure("TButton", font=("Arial", 11, "bold"))
    style.configure("TEntry", font=("Arial", 11))

    entries = {}
    labels = ["Giá mở cửa", "Giá cao nhất", "Giá thấp nhất", "KL giao dịch (M)", "% Thay đổi"]
    for i, label in enumerate(labels):
        ttk.Label(form_frame, text=label + ":").grid(row=i, column=0, sticky="e", padx=10, pady=5)
        entry = ttk.Entry(form_frame)
        entry.grid(row=i, column=1, padx=10, pady=5)
        entries[label] = entry

    result_label = ttk.Label(form_frame, text="")
    result_label.grid(row=6, column=0, columnspan=2, pady=10)

    metrics_label = ttk.Label(
        form_frame,
        text=f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}"
    )
    metrics_label.grid(row=7, column=0, columnspan=2, pady=5)

    button_row = ttk.Frame(form_frame)
    button_row.grid(row=8, column=0, columnspan=2, pady=10)

    ttk.Button(
        form_frame,
        text="Dự đoán",
        command=predict_callback
    ).grid(row=5, column=0, columnspan=2, pady=10)

    ttk.Button(
        button_row,
        text="Dự đoán xu hướng",
        command=forecast_callback
    ).pack(side="left", padx=5)

    ttk.Button(
        button_row,
        text="Biểu đồ so sánh",
        command=back_callback
    ).pack(side="left", padx=5)

    forecast_plot_frame = ttk.Frame(main_frame)
    forecast_plot_frame.grid(row=1, column=0, columnspan=2, pady=10)

    return window, entries, result_label, metrics_label, plot_frame

def _get_float_input(entries, key):
    """Chuyển đổi giá trị từ ô nhập thành float, trả về None nếu không hợp lệ."""
    try:
        return float(entries[key].get())
    except (ValueError, TypeError):
        return None

def predict(entries, result_label, metrics_label, plot_frame, model, scaler, df_data):
    try:
        mo = _get_float_input(entries, "Giá mở cửa")
        cao = _get_float_input(entries, "Giá cao nhất")
        thap = _get_float_input(entries, "Giá thấp nhất")
        kl = _get_float_input(entries, "KL giao dịch (M)") * 1_000_000
        pt = _get_float_input(entries, "% Thay đổi")

        if any(x is None for x in [mo, cao, thap, kl, pt]):
            raise ValueError("Vui lòng nhập đầy đủ và đúng định dạng số.")
        if thap > cao:
            raise ValueError("Giá thấp nhất không thể lớn hơn giá cao nhất.")
        if any(x < 0 for x in [mo, cao, thap, kl]):
            raise ValueError("Giá trị không được âm.")

        data = pd.DataFrame([[mo, cao, thap, kl, pt]],
                            columns=['Mo', 'Cao', 'Thap', 'KL', 'Phan_tram'])
        data_scaled = scaler.transform(data)
        data_scaled_df = pd.DataFrame(data_scaled, columns=['Mo', 'Cao', 'Thap', 'KL', 'Phan_tram'])
        ket_qua = round(model.predict(data_scaled_df)[0], 2)
        result_label.config(text=f"Giá đóng cửa dự đoán: {ket_qua}", foreground="blue")
        append_new_data(mo, cao, thap, kl, pt, ket_qua)

        df_data = load_data()
        # Gọi train_model với các tham số phù hợp
        model, scaler, mae, rmse, r2, y_test, y_pred = train_model(
            df_data,
            input_cols=['Mo', 'Cao', 'Thap', 'KL', 'Phan_tram'],
            target_col='Dong_cua'
        )
        metrics_label.config(text=f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

        global du_bao_cache
        du_bao_cache = None
        # Đã xóa: forecast_10_days(plot_frame, model, scaler, df_data)
        return {
            'model': model,
            'scaler': scaler,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred,
            'df_data': df_data
        }

    except Exception as e:
        result_label.config(text=f"Lỗi: {e}", foreground="red")
        return {
            'model': model,
            'scaler': scaler,
            'mae': mae if 'mae' in locals() else 0.0,
            'rmse': rmse if 'rmse' in locals() else 0.0,
            'r2': r2 if 'r2' in locals() else 0.0,
            'y_test': y_test if 'y_test' in locals() else pd.Series(),
            'y_pred': y_pred if 'y_pred' in locals() else pd.Series(),
            'df_data': df_data
        }


def forecast(plot_frame, model, scaler, df_data):
    print("→ Tính toán xu hướng...")
    gia_du_bao = du_bao_xu_huong(df_data, n=5)
    render_chart(lambda ax: bieu_do_du_bao(ax, gia_du_bao, df_data), plot_frame)

def reshow_so_sanh(plot_frame, y_test, y_pred):
    render_chart(lambda ax: bieu_do_so_sanh(ax, y_test, y_pred, n=30), plot_frame)


def render_chart(ax_func, plot_frame):
    print("→ Bắt đầu vẽ biểu đồ...")
    for widget in plot_frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax_func(ax)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    plt.close('all')
