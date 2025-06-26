from data_utils import load_data, load_data_goc
from model_utils import train_model
from ui_controller import create_gui, predict, forecast, reshow_so_sanh

# Load dữ liệu
df_data = load_data()        # Dùng để train mô hình chung
df_goc = load_data_goc()     # Dùng cho biểu đồ so sánh

# Train trên tất cả dữ liệu
model, scaler, mae, rmse, r2, y_test, y_pred = train_model(
    df_data,
    input_cols=['Mo', 'Cao', 'Thap', 'KL', 'Phan_tram'],
    target_col='Dong_cua'
)

# Train riêng trên dữ liệu gốc để vẽ biểu đồ so sánh chuẩn
_, _, _, _, _, y_test_goc, y_pred_goc = train_model(
    df_goc,
    input_cols=['Mo', 'Cao', 'Thap', 'KL', 'Phan_tram'],
    target_col='Dong_cua'
)

# Biến trạng thái
state = {
    'model': model,
    'scaler': scaler,
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'y_test': y_test,
    'y_pred': y_pred,
    'y_test_goc': y_test_goc,
    'y_pred_goc': y_pred_goc,
    'df_data': df_data,
    'df_goc': df_goc
}

# Tạo GUI
window, entries, result_label, metrics_label, plot_frame = create_gui(
    predict_callback=lambda: state.update(
        predict(entries, result_label, metrics_label, plot_frame,
                state['model'], state['scaler'], state['df_data'])
    ),
    forecast_callback=lambda: forecast(
        plot_frame, state['model'], state['scaler'], state['df_data']
    ),
    back_callback=lambda: reshow_so_sanh(
        plot_frame, state['y_test_goc'], state['y_pred_goc']
    ),
    mae=mae, rmse=rmse, r2=r2
)

# Gọi luôn biểu đồ xu hướng
forecast(plot_frame, state['model'], state['scaler'], state['df_data'])

# Chạy GUI
window.mainloop()
