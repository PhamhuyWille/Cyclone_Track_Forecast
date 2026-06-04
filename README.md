# 🌀 Dự báo quỹ đạo của bão nhiệt đới khu vực Tây Thái Bình Dương bằng học máy và phương pháp Chuỗi hồi quy

> Một hệ thống ML chuyên biệt để dự báo quỹ đạo của xoáy thuận nhiệt đới tại khu vực Tây Bắc Thái Bình Dương (WNP). Hệ thống sử dụng các thuật toán **Machine Learning tiên tiến**, chiến lược **Chuỗi Hồi Quy (Regressor Chain)**, kết hợp cùng **Hệ thống Web Hiện Đại (FastAPI + Vite/Leaflet)** để trình diễn kết quả trực quan trên bản đồ.

## GVHD:
PSG.TS.Nguyễn Lưu Thùy Ngân

TS. Dương Ngọc Hảo

## Thành viên:
1. Phạm Đình Quang Huy - MSSV: 24520689 

2. Huỳnh Lâm Bảo Khang - MSSV: 24520738



---

## 📋 Mục Lục

- [Tổng Quan](#tổng-quan)
- [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
- [Hệ Thống Web App Trực Quan (Mới)](#hệ-thống-web-app-trực-quan-mới)
- [Các Mô Hình Học Máy & Thuật Toán](#các-mô-hình-học-máy--thuật-toán)
- [Nguồn Dữ Liệu & Tiền Xử Lý](#nguồn-dữ-liệu--tiền-xử-lý)
- [Hướng Dẫn Cài Đặt & Khởi Chạy](#hướng-dẫn-cài-đặt--khởi-chạy)
- [Hiệu Suất Mô Hình](#hiệu-suất-mô-hình)

---

## Tổng Quan

Dự án này ban đầu là một framework học máy chuyên sâu nghiên cứu về các mô hình chuyển động phức tạp của bão, nay đã được nâng cấp thành một ứng dụng Fullstack hoàn chỉnh:
1. **Core ML:** Dùng thuật toán Gradient Boosting (LightGBM, XGBoost) và chiến lược Regressor Chain để giải quyết bài toán chuỗi thời gian không gian.
2. **Backend:** Sử dụng **FastAPI** phục vụ truy vấn mô hình. Tự động tính toán sai số DPE (Direct Position Error) trung bình.
3. **Frontend:** Giao diện người dùng sử dụng **Vite + Vanilla JS**, CSS thuần thiết kế theo phong cách **Glassmorphism & Dark Mode** cao cấp. Tích hợp bản đồ Leaflet chuẩn chủ quyền Tiếng Việt.

### Phát Biểu Bài Toán Machine Learning

Cho trạng thái hiện tại của bão và các điều kiện môi trường, dự báo sự dịch chuyển vị trí trong tương lai:
- **Biến Đích:** $\Delta Lat = LAT_{T+H} - LAT_{T_0}$ và $\Delta Lon = LON_{T+H} - LON_{T_0}$
- **Chân Trời Dự Báo:** +6h, +12h, +24h
- **Chiến lược:** Hồi quy độc lập (Independent) hoặc Chuỗi hồi quy (Regressor Chain) học sự phụ thuộc giữa các biến đích.

---

## Cấu Trúc Thư Mục

```text
Cyclone_Track_Forecast/
│
├── 📄 README.md                      # Tệp tài liệu này
├── 📄 LICENSE                        # Giấy phép dự án
│
├── api/                              # (MỚI) Backend Server
│   ├── main.py                       # FastAPI Endpoints (/api/storms, /api/predict)
│   ├── requirements.txt              # Thư viện cho Backend
│   └── models/                       # Nơi lưu trữ mô hình đã huấn luyện (.pkl) và test_data.csv
│
├── web/                              # (MỚI) Frontend Web Application
│   ├── index.html                    # Giao diện chính (Dark Mode, Glassmorphism)
│   ├── main.js                       # Logic gọi API, vẽ bản đồ Leaflet, Animations
│   ├── style.css                     # CSS Animations, Invert Map Filters
│   └── package.json                  # Cấu hình Vite
│
├── data/                             # Quản lý dữ liệu (Raw, Processed, Final)
├── notebook/                         # Jupyter notebooks cho EDA & Data Processing
├── experiments/                      # Theo dõi kết quả báo cáo thực nghiệm
│
└── src/                              # Mã nguồn Machine Learning Core
    ├── config.py                     # Siêu tham số & Đường dẫn
    ├── data_loader/                  # Tiền xử lý (IBTrACS, ENSO, ERA5)
    ├── models/                       # Triển khai GBDT, XGBoost, LightGBM
    ├── evaluation/                   # Tính sai số DPE, độ nhạy vận tốc
    └── pipeline/                     # Chạy thực nghiệm và chiến lược (Chain/Independent)
        ├── compare.py               
        └── export_model.py           # Xuất mô hình tốt nhất ra thư mục api/
```

---

## Hệ Thống Web App Trực Quan 

Điểm nhấn của hệ thống là ứng dụng Web được xây dựng hoàn toàn từ số không nhằm mục đích demo đồ án một cách chuyên nghiệp nhất:

1. **Giao Diện dễ dùng (UI/UX):** Sử dụng thiết kế Glassmorphism trên nền Dark Mode. Cung cấp trải nghiệm như hệ thống điều khiển trung tâm của đài khí tượng.
2. **Trình Diễn Hoạt Ảnh Dự Báo:** Vẽ từng bước đường đi của bão (+6h, +12h, +24h) với hiệu ứng Ping (CSS PopIn) và đường kẻ chuyển động (SVG stroke animations), camera bản đồ bay (flyTo) theo thời gian thực mô phỏng bản tin Thời sự.
3. **Hiển Thị Quỹ Đạo Quá Khứ:** Khi chọn một mốc thời gian T0, toàn bộ quỹ đạo hình thành bão trước đó sẽ được tự động vẽ ra dưới dạng đường đứt nét.
4. **Bản Đồ Bản Địa Hóa:** Tích hợp TileLayer của Google Maps với cấu hình `hl=vi` và dùng kĩ thuật Invert CSS để đổi bản đồ sáng thành Dark map. Đảm bảo hiển thị đầy đủ và chính xác tên địa danh lãnh thổ Tiếng Việt: **Biển Đông, Quần đảo Hoàng Sa, Quần đảo Trường Sa**.

---

## Các Mô Hình Học Máy & Thuật Toán

Dự án áp dụng 3 thuật toán Boosting hàng đầu:
- **LightGBM:** Huấn luyện nhanh nhất, độ chính xác cao nhất (được chọn làm Best Model).
- **XGBoost:** Cân bằng tốc độ và hiệu suất, kháng nhiễu tốt.
- **GBDT (Scikit-learn):** Mô hình cơ sở (baseline) mạnh mẽ.

**Chiến Lược Regressor Chain:** 
Thay vì dự báo độc lập từng mốc, hệ thống dự báo chuỗi: Tọa độ 6h được dùng làm "tri thức" (Feature) để mô hình tiếp theo dự đoán 12h, và tương tự lên 24h. Kỹ thuật này giúp mô hình bắt được quán tính di chuyển của cơn bão.

---

## Hướng Dẫn Cài Đặt & Khởi Chạy

### Yêu Cầu
- Python >= 3.8
- Node.js >= 16 (Dành cho Frontend)

### Bước 1: Cài đặt và Chuẩn bị Model
```bash
git clone https://github.com/yourusername/Cyclone_Track_Forecast.git
cd Cyclone_Track_Forecast
pip install -r requirements.txt

# Chạy script huấn luyện và xuất mô hình tốt nhất cho Backend
python src/pipeline/export_model.py
```
> Script này sẽ huấn luyện `LightGBM` + `Regressor Chain` và tạo ra tệp `api/models/best_model.pkl` cùng `test_data.csv`.

### Bước 2: Khởi động Backend (FastAPI)
Mở một cửa sổ Terminal mới:
```bash
cd Cyclone_Track_Forecast
pip install fastapi uvicorn
uvicorn api.main:app --reload --port 8000
```
> Khi có dòng chữ `Application startup complete` thì Backend đã sẵn sàng tính toán Quỹ đạo và DPE.

### Bước 3: Khởi động Frontend (Vite)
Mở thêm một cửa sổ Terminal thứ hai:
```bash
cd Cyclone_Track_Forecast/web
npm install
npm run dev
```
> Click vào đường link xuất hiện trên Terminal để mở trang Web Dự Báo Bão. F5 để làm mới nếu cần.

---

## Hiệu Suất Mô Hình

Sử dụng 88+ đặc trưng (Khí quyển, Hải dương học, Lịch sử dịch chuyển):
- **Sai Số Vị Trí Trực Tiếp (DPE 12h):** ~61.03 km (Vượt mục tiêu 65 km)
- **Sai Số Vị Trí Trực Tiếp (DPE 24h):** ~145 km
- **Độ Ổn Định Hướng:** Đạt ~85.26% khả năng đoán đúng phương hướng (Đông/Tây/Nam/Bắc).

---

## Đóng Góp & Liên Hệ

Đây là dự án thực tế phục vụ đồ án Cuối kỳ. Mọi góp ý về kiến trúc mô hình hay UI/UX Web vui lòng mở Issue trực tiếp trên kho lưu trữ.

**Cập nhật lần cuối:** 04/06/2026

**Trạng thái:** ✅ Hoàn tất & Sẵn sàng Bảo vệ. Hệ thống đã tích hợp API Backend và Dashboard Frontend.
