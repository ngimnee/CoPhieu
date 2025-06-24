from data_utils import load_data
from model_utils import train_model
from ui_controller import create_gui, predict, forecast_7_days, reshow_so_sanh

# Load dữ liệu và train mô hình ban đầu
df_data = load_data()
model, scaler, mae, rmse, r2, y_test, y_pred = train_model(
    df_data,
    input_cols=['Mo', 'Cao', 'Thap', 'KL', 'Phan_tram'],
    target_col='Dong_cua'
)

# Biến chứa mô hình và dữ liệu
state = {
    'model': model,
    'scaler': scaler,
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'y_test': y_test,
    'y_pred': y_pred,
    'df_data': df_data
}

# Tạo GUI
window, entries, result_label, metrics_label, plot_frame = create_gui(
    predict_callback=lambda: state.update(
        predict(entries, result_label, metrics_label, plot_frame,
                state['model'], state['scaler'], state['df_data'])
    ),
    forecast_callback=lambda: forecast_7_days(
        plot_frame, state['model'], state['scaler'], state['df_data']
    ),
    back_callback=lambda: reshow_so_sanh(
        plot_frame, state['y_test'], state['y_pred']
    ),
    mae=mae, rmse=rmse, r2=r2
)

# Gọi luôn biểu đồ xu hướng ngay khi mở app
forecast_7_days(plot_frame, state['model'], state['scaler'], state['df_data'])

# Chạy GUI
window.mainloop()
