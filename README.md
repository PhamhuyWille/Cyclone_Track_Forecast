# 🌀 Hệ Thống Dự Báo Quỹ Đạo Xoáy Thuận Nhiệt Đới

> Một khung học máy (Machine Learning framework) chuyên biệt để dự báo chuyển động của xoáy thuận nhiệt đới tại khu vực Tây Bắc Thái Bình Dương (WNP) sử dụng các kỹ thuật gradient boosting nâng cao.

**Khóa học:** DS102 - Đồ án Cuối Kỳ | **Trường:** Đại học Công nghệ Thông tin | **Học kỳ:** 4  
**Năm:** 2024 | **Giấy phép:** Xem tệp [LICENSE](LICENSE)

---

## 📋 Mục Lục

- [Tổng Quan](#tổng-quan)
- [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
- [Đặc Điểm Chính](#đặc-điểm-chính)
- [Nguồn Dữ Liệu](#nguồn-dữ-liệu)
- [Quy Trình Xử Lý Dữ Liệu](#quy-trình-xử-lý-dữ-liệu)
- [Các Mô Hình & Thuật Toán](#các-mô-hình--thuật-toán)
- [Chiến Lược Chống Rò Rỉ Dữ Liệu](#chiến-lược-chống-rò-rỉ-dữ-liệu)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Hiệu Suất Mô Hình](#hiệu-suất-mô-hình)
- [Chi Tiết Kỹ Thuật](#chi-tiết-kỹ-thuật)

---

## Tổng Quan

Dự án này triển khai một khung học máy chuyên biệt để dự báo quỹ đạo (trajectory) của các xoáy thuận nhiệt đới tại khu vực Tây Bắc Thái Bình Dương. Hệ thống sử dụng **các thuật toán gradient boosting nâng cao** (GradientBoosting, XGBoost, và LightGBM) để học các mô hình chuyển động phức tạp và phi tuyến tính của bão.

### Phát Biểu Bài Toán

Cho trạng thái hiện tại của xoáy thuận nhiệt đới và các điều kiện môi trường, dự báo sự dịch chuyển vị trí trong tương lai:

- **Biến Đích:**
  - $\Delta Lat = LAT_{T+24} - LAT_{T_0}$ (Thay đổi vĩ độ)
  - $\Delta Lon = LON_{T+24} - LON_{T_0}$ (Thay đổi kinh độ)
- **Chân Trời Dự Báo:** 6h, 12h, 24h
- **Vị Trí Dự Báo:** $Position_{T+24} = Position_{T_0} + (\Delta X, \Delta Y)$

### Mục Tiêu Chính

1. ✅ Dự báo vị trí xoáy với độ chính xác cao (mục tiêu: ~65 km cho 12h, ~145 km cho 24h)
2. ✅ Xác định độ ổn định hướng di chuyển (mục tiêu: ~85%+ độ chính xác)
3. ✅ Bắt được các thay đổi vận tốc và gia tốc
4. ✅ Đảm bảo zero data leakage trong dự báo theo thời gian

---

## Cấu Trúc Thư Mục

```
Cyclone_Track_Forecast/
│
├── 📄 README.md                      # Tệp này
├── 📄 LICENSE                        # Giấy phép dự án
├── 📄 Sumary.md                      # Tóm tắt chi tiết (Tiếng Việt)
│
├── data/                             # Quản lý dữ liệu
│   ├── raw/                          # Dữ liệu thô từ các nguồn bên ngoài
│   │   ├── ibtracs.csv              # Dữ liệu quỹ đạo bão từ IBTrACS
│   │   ├── EU(00-15)/               # Dữ liệu ERA5 2000-2015
│   │   │   ├── data.grib
│   │   │   └── data.grib.5b7b6.idx
│   │   └── EU(16-24)/               # Dữ liệu ERA5 2016-2024
│   │       ├── data.grib
│   │       └── data.grib.5b7b6.idx
│   │
│   ├── processed/                   # Dữ liệu đã làm sạch và xử lý
│   │   ├── climate_feature.csv      # Đặc trưng chỉ số ENSO
│   │   ├── dynamic_feature.csv      # Đặc trưng động học
│   │   ├── enso_clean.csv           # Dữ liệu ENSO đã xử lý
│   │   └── final_data.csv           # Dữ liệu kết hợp
│   │
│   └── final/                       # Ma trận đặc trưng và biến đích cuối cùng
│       ├── features.csv             # 88+ đặc trưng đã kỹ thuật hóa (có SID)
│       └── target.csv               # Biến đích (Δlat, Δlon) (có SID)
│
├── notebook/                         # Jupyter notebooks cho EDA & thử nghiệm
│   ├── data_note/                   # Notebooks xử lý dữ liệu
│   │   ├── concation.ipynb          # Ghép dữ liệu
│   │   ├── enso_clean.ipynb         # Làm sạch dữ liệu ENSO
│   │   ├── eu_file_clean.ipynb      # Trích xuất dữ liệu ERA5
│   │   ├── ibtracs_clean.ipynb      # Tiền xử lý IBTrACS
│   │   └── target.ipynb             # Tạo biến đích
│   │
│   └── model/                       # Thử nghiệm mô hình
│       ├── 6h/                      # Thử nghiệm dự báo 6 giờ
│       │   ├── GDBT.ipynb
│       │   ├── LightGBM.ipynb
│       │   └── XGBoost.ipynb
│       ├── 12h/                     # Thử nghiệm dự báo 12 giờ
│       │   ├── GBDT.ipynb
│       │   ├── LightGBM.ipynb
│       │   └── XGBoost.ipynb
│       └── 24h/                     # Thử nghiệm dự báo 24 giờ
│           ├── GBDT_24h.ipynb
│           ├── LightGBM_24h.ipynb
│           └── XGBoost_24h.ipynb
│
├── experiments/                      # Theo dõi thử nghiệm & kết quả
│   ├── final_comparison_report.csv  # So sánh hiệu suất mô hình
│   └── final_comparison_report_ver02.csv
│
└── src/                              # Mã nguồn
    ├── config.py                     # Cấu hình và siêu tham số
    ├── main.py                       # Điểm vào chính của quy trình
    │
    ├── data_loader/                  # Các mô-đun tiền xử lý dữ liệu
    │   ├── __init__.py
    │   ├── _01_ibtracs_clean.py     # Tiền xử lý dữ liệu IBTrACS
    │   ├── _02_enso_clean.py        # Xử lý dữ liệu ENSO
    │   ├── _03_ERA5_clean.py        # Trích xuất & xử lý ERA5
    │   ├── _04_feature_engineering.py # Quy trình kỹ thuật hóa đặc trưng
    │   └── _05_target_split.py      # Tạo biến đích
    │
    ├── models/                       # Các mô hình học máy
    │   ├── __init__.py
    │   ├── model.py                 # Giao diện mô hình cơ sở
    │   ├── gdbt.py                  # Triển khai GBDT từ Scikit-learn
    │   ├── xgboost.py               # Mô hình XGBoost
    │   └── lightgbm.py              # Mô hình LightGBM
    │
    ├── evaluation/                   # Các chỉ số đánh giá
    │   └── metrics.py               # Chỉ số tùy chỉnh (DPE, ổn định hướng, v.v.)
    │
    └── pipeline/                     # Các chiến lược dự báo
        ├── compare.py               # Chạy so sánh thử nghiệm
        ├── normal.py                # Chiến lược dự báo độc lập
        ├── regressor_chain.py       # Chiến lược chuỗi hồi quy
        └── cliper.py                # Phương pháp cơ sở CLIPER
```

---

## Đặc Điểm Chính

### 1. **Tích Hợp Dữ Liệu Đa Nguồn**

- **IBTrACS:** Quỹ đạo bão lịch sử (2000-2024)
- **ERA5:** Các điều kiện khí quyển (gió, áp suất, SST, độ ẩm)
- **MEIv2:** Chỉ số ENSO đa biến (lực sinh khí hậu)

### 2. **Kỹ Thuật Hóa Đặc Trưng Toàn Diện** (88+ Đặc Trưng)

#### Đặc Trưng Động Học & Động Lực:

- Vị trí lịch sử tại T-6, T-12, T-18, T-24
- Thành phần vận tốc (zonal, meridional)
- Gia tốc và hướng chuyển động
- Azimuth và góc quay

#### Đặc Trưng Môi Trường:

- **Gió:** u10, v10 (thành phần gió bề mặt)
- **Áp suất:** msl (áp suất mực nước biển)
- **Đại dương:** sst (nhiệt độ mặt nước biển)
- **Độ ẩm:** vimdf (phân kỳ ẩm tích hợp theo chiều dọc)

#### Đặc Trưng Khí Hậu:

- Chỉ số ENSO (climatology cửa sổ mở rộng)
- Thống kê được bảo vệ chống rò rỉ dữ liệu

### 3. **Nhiều Chiến Lược Dự Báo**

- **Độc Lập:** Mô hình riêng cho Δlat và Δlon
- **Chuỗi Hồi Quy:** Dự báo tuần tự có phản hồi
- **CLIPER:** Phương pháp cơ sở thống kê

### 4. **Ba Mô Hình Hiện Đại**

- **GBDT** (Gradient Boosting từ Scikit-learn)
- **XGBoost** (Extreme Gradient Boosting)
- **LightGBM** (Light Gradient Boosting Machine)

---

## Nguồn Dữ Liệu

| Nguồn       | Mô Tả                                | Phạm Vi   | Đặc Trưng                |
| ----------- | ------------------------------------ | --------- | ------------------------ |
| **IBTrACS** | Kho lưu trữ quỹ đạo tốt nhất quốc tế | 2000-2024 | Vị trí, áp suất, sức gió |
| **ERA5**    | Tái phân tích ECMWF                  | 2000-2024 | Gió, áp suất, SST, độ ẩm |
| **MEIv2**   | Chỉ số ENSO của NOAA                 | 2000-2024 | Lực sinh khí hậu         |

---

## Quy Trình Xử Lý Dữ Liệu

### Bước 1: Làm Sạch IBTrACS (`_01_ibtracs_clean.py`)

- Tải quỹ đạo bão lịch sử
- Loại bỏ các mục nhập không hợp lệ
- Nội suy các quan sát bị thiếu
- Đầu ra: `ibtracs_clean.csv`

### Bước 2: Xử Lý ENSO (`_02_enso_clean.py`)

- Tải/tải xuống chỉ số MEI v2
- Đồng bộ hóa với dấu thời gian bão
- Tạo đặc trưng climatology cửa sổ mở rộng
- Đầu ra: `enso_clean.csv` + `climate_feature.csv`

### Bước 3: Trích Xuất ERA5 (`_03_ERA5_clean.py`)

- Phân tích tệp GRIB từ ECMWF
- Trích xuất các biến liên quan (u10, v10, msl, sst, vimdf)
- Nội suy đến vị trí và thời gian bão
- Đầu ra: `dynamic_feature.csv`

### Bước 4: Kỹ Thuật Hóa Đặc Trưng (`_04_feature_engineering.py`)

- Kết hợp tất cả các nguồn dữ liệu
- Tạo đặc trưng trễ (6h, 12h, 18h, 24h)
- Tính toán các đặc trưng dẫn xuất (vận tốc, gia tốc)
- Xử lý giá trị thiếu bằng phương pháp nội suy an toàn
- Đầu ra: `features.csv`

### Bước 5: Tạo Biến Đích (`_05_target_split.py`)

- Tính toán sự thay đổi vị trí: Δlat và Δlon
- Tạo biến đích cho chân trời 6h, 12h, 24h
- Loại bỏ các mẫu có biến đích không hoàn chỉnh
- Đầu ra: `target.csv`

---

## Các Mô Hình & Thuật Toán

### 1. **Gradient Boosting Decision Trees (GBDT)**

```python
GBDT_PARAMS = {
    'n_estimators': 1000,
    'max_depth': 3,
    'max_features': 60,
    'learning_rate': 0.1,
}
```

### 2. **XGBoost**

```python
XGB_PARAMS = {
    'max_depth': 3,
    'learning_rate': 0.01,
    'n_estimators': 5000,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
}
```

### 3. **LightGBM**

```python
LGBM_PARAMS = {
    'learning_rate': 0.01,
    'n_estimators': 3000,
    'max_depth': 8,
    'num_leaves': 7,
    'feature_fraction': 0.6,
    'lambda_l1': 0.1,
    'lambda_l2': 1,
}
```

---

## Chiến Lược Chống Rò Rỉ Dữ Liệu

Dự án này triển khai các giao thức **toàn vẹn dữ liệu thời gian** nghiêm ngặt để đảm bảo dự báo hợp lệ:

### 1. **Chia Tập Dữ Liệu Dựa Trên Thời Gian**

```
Huấn luyện:  2000-2020 (Học mô hình)
Xác thực:    2021-2022 (Dừng sớm)
Kiểm tra:    2023-2024 (Xác thực thực tế)
```

❌ **KHÔNG sáo trộn ngẫu nhiên** - Thứ tự thời gian rất quan trọng

### 2. **Giao Thức Nội Suy An Toàn**

- Các giá trị thiếu được điền **cho từng xoáy** sử dụng `.groupby('SID').bfill()`
- Không pha trộn dữ liệu giữa các cơn bão
- Cô lập với vòng đời bão cá nhân

### 3. **Làm Sạch Biến Đích**

- Tự động loại bỏ các quan sát cuối cùng (không có biến đích tương lai)
- Đảm bảo Δlat và Δlon hợp lệ cho tất cả các mẫu

### 4. **Không Nhìn Trước Tương Lai**

- Dữ liệu ERA5 và MEIv2 được đồng bộ chặt chẽ đến dấu thời gian UTC
- Tại thời điểm T, mô hình chỉ thấy dữ liệu T trở về quá khứ
- Không thể tiếp cận các điều kiện khí quyển của T+6 hoặc T+24

---

## Hướng Dẫn Sử Dụng

### Yêu Cầu Tiên Quyết

```bash
Python >= 3.8
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0
lightgbm >= 3.3.0
eccodes >= 2.20.0  # Để đọc tệp GRIB
```

### Cài Đặt

1. **Clone kho lưu trữ**

   ```bash
   git clone https://github.com/yourusername/Cyclone_Track_Forecast.git
   cd Cyclone_Track_Forecast
   ```

2. **Cài đặt các gói phụ thuộc**

   ```bash
   pip install -r requirements.txt
   ```

3. **Chuẩn bị dữ liệu**
   - Đặt dữ liệu IBTrACS trong `data/raw/`
   - Đặt tệp GRIB ERA5 trong `data/raw/EU(00-15)/` và `data/raw/EU(16-24)/`
   - Tải chỉ số ENSO từ NOAA

### Chạy Quy Trình

#### Tùy Chọn 1: Quy Trình Đầy Đủ

```bash
python src/main.py
```

Điều này thực hiện tất cả các bước:

1. ✅ Tiền xử lý IBTRACS
2. ✅ Xử lý ENSO
3. ✅ Trích xuất ERA5
4. ✅ Kỹ thuật hóa đặc trưng
5. ✅ Chia biến đích
6. ✅ Huấn luyện & đánh giá mô hình

#### Tùy Chọn 2: Các Bước Riêng Lẻ

**Chuẩn bị dữ liệu:**

```bash
python src/data_loader/_01_ibtracs_clean.py
python src/data_loader/_02_enso_clean.py
python src/data_loader/_03_ERA5_clean.py
python src/data_loader/_04_feature_engineering.py
python src/data_loader/_05_target_split.py
```

**Huấn luyện mô hình:**

```bash
python src/pipeline/compare.py
```

#### Tùy Chọn 3: Jupyter Notebooks

Điều hướng đến thư mục `notebook/` và chạy các tệp `.ipynb` riêng lẻ để phân tích khám phá và thử nghiệm.

---

## Hiệu Suất Mô Hình

### Chỉ Số Đánh Giá

1. **Sai Số Vị Trí Trực Tiếp (DPE)**
   - Sai số khoảng cách tính bằng km sử dụng công thức Haversine
   - Mục tiêu: ~65 km (12h), ~145 km (24h)

2. **Độ Ổn Định Hướng**
   - Phần trăm dự báo hướng chính xác (N/S/E/W)
   - Mục tiêu: ~85%+

3. **Độ Nhạy Vận Tốc**
   - Khả năng bắt các mô hình tăng tốc/giảm tốc
   - Kỳ vọng: <60% (khó khăn cho mô hình dựa trên cây)

### Tóm Tắt Kết Quả

Các kết quả mới nhất được lưu trữ tại:

- `experiments/final_comparison_report.csv`
- `experiments/final_comparison_report_ver02.csv`

So sánh hiệu suất trên:

- **Mô hình:** GBDT vs XGBoost vs LightGBM
- **Chân trời:** 6h, 12h, 24h
- **Chiến lược:** Độc lập, Chuỗi, CLIPER

#### Bảng So Sánh Tóm Tắt

| Mô Hình  | Chiến Lược | Chân Trời | Sai Số (km) | Ổn Định (%) | Vận Tốc (%) |
| -------- | ---------- | --------- | ----------- | ----------- | ----------- |
| LightGBM | Normal     | 12h       | 61.17       | 84.68       | 13.33       |
| XGBoost  | Normal     | 12h       | 61.30       | 85.55       | 11.11       |
| GBDT     | Normal     | 12h       | 66.14       | 84.83       | 20.00       |
| LightGBM | Chain      | 12h       | 61.03       | 85.26       | 11.11       |
| CLIPER   | Baseline   | 12h       | 67.11       | 84.25       | 8.89        |

---

## Chi Tiết Kỹ Thuật

### Lựa Chọn Đặc Trưng

Từ 88+ đặc trưng ứng viên, hệ thống chọn những đặc trưng phù hợp dựa trên:

- Mối tương quan với biến đích
- Tầm quan trọng cây từ các mô hình ban đầu
- Kiến thức tên miền (liên quan về mặt vật lý)
- Phân tích đa cộng tuyến

### Xử Lý Dữ Liệu Thiếu

**Chiến lược:** Nội suy an toàn, cục bộ

- Nhóm theo ID xoáy (SID)
- Điền lùi trong mỗi cơn bão
- Loại bỏ các hàng có NaN còn lại trong biến đích
- Ngăn chặn ô nhiễm giữa các cơn bão

### Tinh Chỉnh Siêu Tham Số

Được sử dụng trong phát triển mô hình:

- Tìm kiếm lưới trên tập xác thực (2021-2022)
- Dừng sớm trên các chỉ số xác thực
- Xác thực chéo trong tập huấn luyện
- Không liên quan đến tập kiểm tra

### Hiệu Quả Tính Toán

- **LightGBM:** Huấn luyện nhanh nhất (dựa trên histogram)
- **XGBoost:** Cân bằng tốc độ/hiệu suất
- **GBDT:** Chậm nhất nhưng cơ sở vững chắc

### Cấu Trúc Đầu Ra

```
├── features.csv          # Hình dạng: (N, 88+)
├── target.csv            # Hình dạng: (N, 6) - 3 chân trời × 2 biến
├── predictions/
│   ├── GBDT_12h.csv
│   ├── XGBoost_12h.csv
│   └── LightGBM_12h.csv
└── metrics/
    └── evaluation_report.csv
```

---

## Tệp Dữ Liệu Chính Chứa SID

Hai tệp CSV chính chứa ID xoáy (SID) để theo dõi và ghép dữ liệu:

1. **`data/final/features.csv`** (88+ đặc trưng)
   - Cột SID: Định danh xoáy duy nhất
   - Cột khác: Các đặc trưng động học, môi trường, khí hậu
   - Số lượng hàng: ~2,000+ mẫu

2. **`data/final/target.csv`** (Biến đích)
   - Cột SID: Định danh xoáy duy nhất
   - Cột DELTA*LAT\*\*h & DELTA*LON\*\*h: Sự thay đổi vị trí
   - Đồng bộ hoàn toàn với features.csv qua SID

---

## Tài Liệu Tham Khảo Chính

### Nguồn Dữ Liệu

- **IBTrACS**: https://www.ncei.noaa.gov/products/international-best-track-archive-and-search
- **ERA5**: https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
- **MEI**: https://origin.cpc.ncei.noaa.gov/products/analysis-monitoring/ensoyears.shtml

### Tài Liệu Khoa Học

- Jinkai Tan et al. - GBDT dự báo quỹ đạo
- Chen, T., & Guestrin, C. - XGBoost (2016)
- Ke, G., et al. - LightGBM (2017)
- Công thức Haversine cho khoảng cách vòng tròn lớn

---

## Đóng Góp

Đây là dự án học tập cho khóa học DS102. Để cải tiến hoặc báo cáo lỗi, vui lòng mở một issue.

---

## Liên Hệ

**Khóa học:** DS102 | **Trường:** Đại học Công nghệ Thông tin  
**Học kỳ:** 4 | **Năm:** 2024

---

**Cập nhật lần cuối:** 28/05/2024  
**Trạng thái:** ✅ Hoàn tất & Sẵn sàng Bảo vệ
