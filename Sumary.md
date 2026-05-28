# 🌀 PROJECT DOCUMENTATION: TROPICAL CYCLONE TRACK FORECASTING (GBDT FRAMEWORK)

## 1. General Introduction

- **Concept:** Một khung học máy (Machine Learning framework) chuyên biệt để dự báo quỹ đạo xoáy thuận nhiệt đới (TC) tại khu vực Tây Bắc Thái Bình Dương (WNP). Hệ thống sử dụng thuật toán **LightGBM (LGBM)** để học các quy luật di chuyển phức tạp và phi tuyến tính của bão.
- **Objective:** Dự báo chính xác mức độ dịch chuyển của tâm bão ($\Delta Lat$, $\Delta Lon$) tại các mốc thời gian tương lai (ví dụ: 12h, 24h). Mô hình tập trung đánh giá giai đoạn bão trưởng thành để đạt độ tin cậy vật lý cao nhất.
- **Data Context:** Tích hợp đa nguồn dữ liệu với hơn 88 đặc trưng (features):
  - _Động lực học:_ Lịch sử quỹ đạo từ IBTrACS.
  - _Khí hậu vĩ mô:_ Chỉ số ENSO đa biến (MEIv2).
  - _Hoàn cảnh môi trường:_ Dữ liệu lưới khí quyển ERA5 từ ECMWF ($u10, v10, msl, sst, vimdf$).
- **Formal Problem Formulation:** Xây dựng hai mô hình hồi quy độc lập:
  - $Target\_Lat: \Delta Y = LAT_{T+24} - LAT_{T_0}$
  - $Target\_Lon: \Delta X = LON_{T+24} - LON_{T_0}$
  - $\hat{Position}_{T+24} = Position_{T_0} + (\Delta X, \Delta Y)$

---

## 2. Feature Engineering & Dataset Structure

Hệ thống sử dụng bộ 88 đặc trưng được thiết kế để cung cấp toàn cảnh về quán tính và môi trường của bão:

1.  **Dynamic & Kinematic Features:** Các biến trễ ($T-6$ đến $T-24$) của vị trí, áp suất, sức gió. Các biến phái sinh như Vận tốc ($V\_Zonal, V\_Meridional$), Gia tốc, và Phương vị (Azimuth) giúp mô hình nhận diện các pha bão đổi hướng (recurvature).
2.  **Expanding Window Climatology:** Các đặc trưng khí hậu (X79, X81, X82, X84) được tính toán bằng phương pháp "cửa sổ mở rộng" - với bão năm $Y$, hệ thống chỉ lấy trung bình của các cơn bão từ năm $Y-1$ trở về trước tại cùng ô lưới. **(Ngăn chặn rò rỉ dữ liệu toàn cục)**.
3.  **Environmental Forcing (ERA5):** Các biến môi trường đóng vai trò là "lực đẩy" (steering flow) điều hướng bão:
    - `u10`, `v10`: Vector gió bề mặt ($m/s$).
    - `msl`: Áp suất mực nước biển ($hPa$).
    - `sst`: Nhiệt độ mặt nước biển ($K$ hoặc $^\circ C$).
    - `vimdf`: Độ phân kỳ ẩm ($kg \cdot m^{-2}s^{-1}$).

---

## 3. The "Anti-Leakage" Core (Quy tắc Chống Rò Rỉ Dữ Liệu)

Mô hình đảm bảo kết quả dự báo phản ánh đúng năng lực thực tế, tuân thủ 4 chốt chặn nghiêm ngặt:

1.  **Strict Time-based Split:** Tuyệt đối không chia ngẫu nhiên (random split).
    - **Training:** 2000 – 2020 (Học quy luật).
    - **Validation:** 2021 – 2022 (Dừng sớm - Early Stopping).
    - **Testing:** 2023 – 2024 (Kiểm chứng thực tế).
2.  **Safe Imputation (Xử lý NaN):** Áp dụng `.groupby('SID').bfill()`. Việc điền khuyết bị cô lập hoàn toàn trong từng cơn bão, không để dữ liệu của bão năm 2000 tràn sang bão năm 2024.
3.  **Target Cleansing:** Tự động loại bỏ (dropna) các mốc thời gian cuối vòng đời của bão vì không tồn tại tọa độ thực tế ở $T+12$, $T+24$.
4.  **No Future Look-ahead:** Việc tích hợp dữ liệu GRIB (ERA5) và MEIv2 được đồng bộ chính xác theo trục thời gian (UTC), đảm bảo tại thời điểm $T$, mô hình không "nhìn thấy" trạng thái khí quyển của $T+6$.

---

## 4. Model Training Optimization

Sử dụng **GradientBoostingRegressor**, **XGBoost**, và **LightGBM** để tận dụng khả năng tự động xử lý giá trị thiếu (Sparsity-aware) từ lưới khí quyển.

- **Hyperparameters (Đối với GradientBoostingRegressor) (Dựa trên Jinkai Tan et al.):**
  - `max_depth`: **3** (Điểm ngọt để tránh Over-fitting).
  - `max_features` (`colsample_bytree`): **~0.7** (Chọn ngẫu nhiên khoảng 60/88 biến mỗi lần rẽ nhánh để chống đa cộng tuyến).
  - `learning_rate`: **0.01 - 0.1**.
  - `n_estimators`: **1000 - 5000** kết hợp với **Early Stopping** trên tập Validation (dừng nếu sai số không giảm sau 50 vòng).
- **XGBoost:**
  - `max_depth`: **3**
  - `learning_rate`: **0.01**
  - `n_estimators`: **5000**
  - `subsample`: **0.8**
  - `colsample_bytree`: **0.7**
  - `n_jobs`: **-1**
- **LightGBM:**
  - `learning_rate`: **0.01**,
  - `n_estimators`: **3000**,
  - `max_depth`: **8**,
  - `num_leaves`: **7**,
  - `feature_fraction`: **0.6**,
  - `lambda_l1`: **0.1**,
  - `lambda_l2`: **1**,

---

## 5. Benchmark & Evaluation Strategy

Hiệu suất của mô hình trên tập Test (2022-2024) được đánh giá bằng 3 độ đo khắt khe chuyên biệt cho quỹ đạo bão:

1.  **DPE (Direct Positional Error):** Đo sai số khoảng cách thực tế (km) bằng công thức lượng giác cầu Haversine. Kỳ vọng đạt mức trung bình ~**65 km** cho dự báo 12h và ~**145 km** cho dự báo 24h.
2.  **Directional Stability (Độ ổn định hướng):** Đánh giá tỷ lệ phần trăm mô hình dự báo đúng hướng di chuyển (Đông/Tây, Nam/Bắc). Kỳ vọng đạt ~**85%** trở lên.
3.  **Moving Velocity Sensitivity (Độ nhạy vận tốc):** Đánh giá khả năng của AI trong việc bắt đúng các pha bão tăng tốc/giảm tốc đột ngột. (Lưu ý: Đây thường là điểm yếu của các mô hình Tree-based, mức kỳ vọng thường nằm ở khoảng **<60%**).

---

**💡 Lời khuyên cho bạn:** Với phạm vi đồ án chỉ tập trung vào **Quỹ đạo (Track)**, cấu trúc trên là cực kỳ hoàn hảo. Khi ra Hội đồng bảo vệ, bạn chỉ cần nhấn mạnh vào **Mục số 2 (Các tính năng vật lý & ERA5)** và **Mục số 3 (Tuyệt đối không có Data Leakage)** là đã đủ để thuyết phục và thể hiện tư duy Khoa học dữ liệu (Data Science) rất xuất sắc của mình rồi!

```typhoon-track-prediction/
│
├── data/                               # Quản lý dữ liệu dự án (Cần thêm vào .gitignore)
│   ├── raw/                            # Dữ liệu bão thô ban đầu (từ NOAA, JMA, JTWC...)
│   │   └── hurricanes_raw.csv
│   ├── processed/                      # Dữ liệu sau khi làm sạch, xử lý khuyết thiếu, nội suy
│   │   └── typhoons_cleaned.csv
│   └── final/                          # Dữ liệu sau feature engineering, sẵn sàng để đưa vào mô hình
│       ├── features_matrix.parquet     # Lưu định dạng parquet để đọc/ghi siêu nhanh
│       └── target_metadata.json        # Lưu thông tin metadata của các cột target
│
├── notebooks/                          # Nơi thử nghiệm nhanh và trực quan hóa (Jupyter Notebooks)
│   ├── 01_data_exploration.ipynb       # Khám phá dữ liệu, vẽ biểu đồ đường đi của bão thực tế
│   ├── 02_feature_engineering.ipynb    # Thử nghiệm tạo các đặc trưng trễ (Lag), sai phân (Diff)
│   ├── 03_exploratory_direct.ipynb     # Chạy thử nhanh kỹ thuật Direct với LightGBM
│   └── 04_exploratory_recursive.ipynb  # Chạy thử nhanh kỹ thuật Recursive với XGBoost
│
├── src/                                # MÃ NGUỒN CHÍNH CỦA HỆ THỐNG
│   ├── __init__.py
│   ├── config.py                       # Lưu toàn bộ tham số mô hình, đường dẫn, danh sách mốc thời gian
│   │
│   ├── data_loader/                    # Phân hệ nạp và xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── preprocess.py               # Hàm xử lý nhiễu, chuẩn hóa tọa độ địa lý
│   │   └── features.py                 # "Xương sống": Hàm tạo Lag features, Diff features, Rolling stats
│   │
│   ├── models/                         # Phân hệ quản lý cấu hình các mô hình học máy
│   │   ├── __init__.py
│   │   └── factory.py                  # Model Factory: Cấp phát instance GBDT, XGBoost hoặc LightGBM dựa trên tên
│   │
│   ├── strategies/                     # PHÂN HỆ QUAN TRỌNG: ĐÓNG GÓI 2 KỸ THUẬT DỰ BÁO
│   │   ├── __init__.py
│   │   ├── base_strategy.py            # Lớp trừu tượng cha định nghĩa luật chung cho các chiến lược
│   │   ├── direct_strategy.py          # Cài đặt thuật toán Dự báo Trực tiếp (Huấn luyện N mô hình song song)
│   │   └── recursive_strategy.py       # Cài đặt thuật toán Dự báo Đệ quy (Huấn luyện 1 mô hình, dự báo cuộn chiếu)
│   │
│   ├── evaluation/                     # Phân hệ đánh giá và kiểm thử mô hình
│   │   ├── __init__.py
│   │   └── metrics.py                  # Hàm tính khoảng cách Haversine (km), tính Across-track / Along-track error
│   │
│   └── compare.py                      # SCRIPT ĐẦU MỐI: Chạy ma trận so sánh (3 Mô hình x 2 Kỹ thuật)
│
├── experiments/                        # NƠI LƯU TRỮ TOÀN BỘ KẾT QUẢ THỬ NGHIỆM
│   ├── checkpoints/                    # Thư mục lưu các mô hình sau khi train xong thành file vật lý
│   │   ├── direct/                     # Nơi lưu file model của kỹ thuật Direct (Ví dụ: xgb_6h.json, lgbm_12h.txt...)
│   │   └── recursive/                  # Nơi lưu file model của kỹ thuật Recursive (Ví dụ: gbdt_step1.pkl...)
│   ├── feature_importance/             # Lưu ảnh đồ thị (.png) sinh ra từ hàm feature_importance của 3 mô hình
│   └── results/                        # Nơi xuất file báo cáo cuối cùng
│       ├── summary_metrics.csv         # Bảng so sánh sai số km của các bên phục vụ viết báo cáo đồ án
│       └── prediction_plots/           # Lưu ảnh trực quan đường đi dự báo vs đường đi thực tế của bão
│
├── scripts/                            # Các script tiện ích chạy bằng lệnh terminal nhanh
│   ├── download_data.sh                # Script tự động cào/tải dữ liệu khí tượng từ server (nếu có)
│   └── run_all.py                      # Lệnh "một chạm" chạy từ tiền xử lý, train, đến xuất báo cáo so sánh
│
├── tests/                              # Phân hệ Unit Test (Đảm bảo code không lỗi logic)
│   ├── __init__.py
│   ├── test_features.py                # Test xem hàm tạo Lag features có bị rò rỉ dữ liệu (Data Leakage) không
│   └── test_strategies.py              # Test xem hàm dự báo cuốn chiếu đệ quy có chạy đúng vòng lặp không
│
├── .gitignore                          # Liệt kê các file nặng, file rác cần bỏ qua không đẩy lên GitHub (data/, checkpoints/, .venv/)
├── README.md                           # Trang hướng dẫn cài đặt môi trường, cấu trúc dự án và cách chạy code
├── main.py                             # File giao diện chính của chương trình ứng dụng (hoặc gọi API/CLI)
└── requirements.txt                    # Danh sách các thư viện cần cài (scikit-learn, xgboost, lightgbm, pandas, numpy, geopy, matplotlib)
```
