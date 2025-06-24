# CoPhieu
Bài tập lớn thuộc học phần Học máy
Xây dựng mô hình dự đoán Xu hướng giá cổ phiếu dựa trên mô hình Hồi quy tuyến tính

Đây là một chương trình Python kết hợp GUI (tkinter) với mô hình học máy (LinearRegression) để dự báo giá đóng cửa cổ phiếu FPT dựa trên các chỉ số trong ngày.

---📁 Cấu trúc thư mục---
├── data
  ├── du_lieu_fpt.csv       # Dữ liệu chính (giá cổ phiếu)
  └── du_lieu_du_doan.csv   # Lưu dự đoán người dùng
├── src
  ├── chart.py              # Vẽ biểu đồ so sánh & xu hướng
  ├── data_utils.py         # Load dữ liệu & cập nhật CSV
  ├── model_utils.py        # Huấn luyện & dự đoán mô hình
  ├── ui_controller.py      # Logic xử lý GUI & biểu đồ
  └── main.py               # Khởi chạy ứng dụng
└── README.md             # Tài liệu hướng dẫn

---🚀 Cách chạy ứng dụng---
1. Cài thư viện:
  - pip install pandas numpy scikit-learn matplotlib
2. Chạy ứng dụng:
  - python main.py

---🧠 Chức năng chính---
  - Dự đoán giá đóng cửa từ thông số nhập tay (giá mở cửa, cao nhất, thấp nhất, khối lượng, phần trăm thay đổi).
  - Tự động cập nhật file CSV sau mỗi lần dự đoán.
  - Huấn luyện lại mô hình để cải thiện độ chính xác.
  - Biểu đồ so sánh giá thực tế vs dự đoán.
  - Biểu đồ xu hướng 10 ngày tiếp theo.
  - Bộ nhớ cache biểu đồ xu hướng giúp cố định xu hướng trong mỗi phiên.

---📈 Giao diện---
  - Dự đoán thủ công với các ô nhập liệu.
  - Nút "Xu hướng 10 ngày tới" để xem biểu đồ tự động.
  - Nút "Biểu đồ so sánh" để quay lại đánh giá mô hình.

---🛠 Kỹ thuật sử dụng---
- LinearRegression từ Scikit-Learn
- StandardScaler để chuẩn hóa dữ liệu
- Tkinter cho GUI
- Matplotlib để hiển thị biểu đồ


---🔍 Các hàm chính---
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
  - So sánh giá thực tế và giá dự đoán trong 15 ngày gần nhất.
  - Biểu đồ dự đoán xu hướng cổ phiếu trong 10 ngày tiếp theo.
