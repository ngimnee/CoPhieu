# CoPhieu
Bài tập lớn thuộc học phần Học máy
Xây dựng mô hình dự đoán Xu hướng giá cổ phiếu dựa trên mô hình Hồi quy tuyến tính

Đây là một chương trình Python kết hợp GUI (tkinter) với mô hình học máy (LinearRegression) để dự báo giá đóng cửa cổ phiếu FPT dựa trên các chỉ số trong ngày.

1. Xử lý dữ liệu: chuyen_doi_so(chuoi)
  - Đọc file data. Lấy các cột làm X đặc trưng đầu vào: giá mở cửa, cao, thấp, khối lượng, % thay đổi.
  - Chuyển đổi chuỗi thành số, xử lý các định dạng "M", "&", "," ...
2. Huấn luyện mô hình: train_model(df)
  - Đặc trưng đầu vào: Mo, Cao, Thap, KL, Phan_tram.
  - Dự đoán đầu ra: Dong_cua của ngày tiếp theo.
  - Chuẩn hóa đặc trưng (StandardScaler), tách train/test 80-20.
  - Huấn luyện LinearRegression, tính các chỉ số:
    + MAE: sai số tuyệt đối
    + RMSE: sai số bình phương
    + R2: độ chính xác của mô hình
3. Dự đoán và lưu kết quả: predict()
  - Lấy giá trị từ GUI.
  - Nhân KL với 1 000 000 để lấy giá trị thật (Đầu vào nhập là định dạng đơn vị triệu).
  - Chuẩn hóa và dự đoán bằng mô hình.
  - Hiển thị kết quả dự đoán.
  - Lưu data dự đoán vào file.
  - Huấn luyện lại mô hình.
4. Giao diện
  - Form nhập: mở cửa, cao nhất, thấp nhất, khối lượng, phần trăm.
  - Nút "Dự đoán" -> predict().
  - Show kết quả và các chỉ số đánh giá.
5. Biểu đồ
  - Trục x:
    + 0 - 30: giá thực tế 30 ngày gần nhất
    + 30 - 60: giá dự báo 30 ngày tiếp
