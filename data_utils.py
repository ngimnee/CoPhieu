import pandas as pd
import os
from datetime import datetime

DATA_FILE = "data/du_lieu_fpt.csv"
USER_DATA_FILE = "data/du_lieu_du_doan.csv"

# Hàm chuyển đổi chuỗi thành số, xử lý các định dạng như 'M', '%', '.', ','
def chuyen_doi_so(chuoi):
    if isinstance(chuoi, str):
        chuoi = chuoi.replace('M', '').replace('%', '').replace('.', '').replace(',', '.')
    try:
        return float(chuoi)
    except:
        return None


# Hàm đọc và xử lý dữ liệu từ tệp CSV
def load_data():
    df = pd.read_csv(DATA_FILE)
    # Nếu có tệp dữ liệu dự đoán, hợp nhất với dữ liệu gốc
    if os.path.exists(USER_DATA_FILE):
        df_user = pd.read_csv(USER_DATA_FILE)
        df = pd.concat([df, df_user], ignore_index=True)

    # Lấy các cột cần thiết từ dữ liệu
    df = df.iloc[:, [0, 1, 2, 3, 4, 5, 6]]
    # Đặt tên cột cho dữ liệu
    df.columns = ['Ngay', 'Dong_cua', 'Mo', 'Cao', 'Thap', 'KL', 'Phan_tram']

    # Áp dụng hàm chuyển đổi cho các cột số
    for cot in ['Dong_cua', 'Mo', 'Cao', 'Thap', 'KL', 'Phan_tram']:
        df[cot] = df[cot].apply(chuyen_doi_so)

    # Loại bỏ các hàng có giá trị thiếu
    df.dropna(inplace=True)
    # Lọc dữ liệu, chỉ giữ giá đóng cửa dưới 1000
    df = df[df['Dong_cua'] < 1000]
    return df


# Hàm lưu dữ liệu dự đoán mới vào tệp
def append_new_data(mo, cao, thap, kl, pt, ket_qua):
    try:
        today = datetime.today().strftime('%Y-%m-%d')   # Ngày hiện tại
        # Tạo DataFrame cho dữ liệu mới
        df_new = pd.DataFrame([[today, ket_qua, mo, cao, thap, kl, pt]],
                              columns=['Ngay', 'Dong_cua', 'Mo', 'Cao', 'Thap', 'KL', 'Phan_tram'])

        # Nếu tệp tồn tại, hợp nhất với dữ liệu cũ
        if os.path.exists(USER_DATA_FILE):
            df_existing = pd.read_csv(USER_DATA_FILE)
            df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_updated = df_new

        # Lưu dữ liệu vào tệp
        df_updated.to_csv(USER_DATA_FILE, index=False)
    except Exception as e:
        print("Không thể lưu dữ liệu dự đoán mới:", e)
