# 🚗 Vietnamese License Plate Detector — Realtime OCR

Ứng dụng phát hiện và nhận dạng biển số xe Việt Nam theo thời gian thực sử dụng camera, EasyOCR và OpenCV.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![EasyOCR](https://img.shields.io/badge/EasyOCR-1.7-orange)
![License](https://img.shields.io/badge/License-UET-yellow)

## ✨ Tính năng

- 📷 **Camera Realtime** — Quét biển số xe trực tiếp từ webcam
- 🇻🇳 **Hỗ trợ biển số Việt Nam** — Nhận dạng biển 1 dòng & 2 dòng (ví dụ: `29A-123.45`)
- 🔍 **Gom nhóm thông minh** — Tự động ghép các dòng text gần nhau thành biển số hoàn chỉnh
- 💾 **Lưu trữ tự động** — Ảnh crop biển số + log CSV vào folder `database/`
- 🎨 **Giao diện chuyên nghiệp** — Dark theme, hiển thị biển số dạng bảng vàng, history cards
- ⚡ **FPS Counter** — Theo dõi hiệu suất realtime
- 🔄 **Chống trùng lặp** — Cooldown 5 giây tránh lưu trùng

## 📸 Giao diện

| Camera Feed | Biển số phát hiện |
|:-----------:|:-----------------:|
| Khung trái hiển thị camera realtime với bounding box xanh bao quanh biển số | Khung phải hiển thị biển số mới nhất, ảnh crop, và lịch sử phát hiện |

## 🛠️ Cài đặt

### Yêu cầu
- Python 3.10+
- Webcam

### Cài đặt dependencies

```bash
pip install opencv-python easyocr numpy Pillow
```

### Chạy ứng dụng

```bash
python license_plate_detector.py
```

## 📁 Cấu trúc project

```
OCR/
├── license_plate_detector.py   # Ứng dụng chính — GUI + Camera + OCR
├── easyOCR.py                  # Script OCR cơ bản (đọc text từ ảnh)
├── database/                   # Thư mục lưu kết quả phát hiện
│   ├── plate_*.jpg             # Ảnh crop biển số
│   └── plates_log.csv          # Log tổng hợp (timestamp, text, confidence)
├── englishquote.jpg            # Ảnh test mẫu
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 Hướng dẫn sử dụng

1. **Nhấn `▶ BẮT ĐẦU`** để bật camera
2. **Đưa biển số xe** vào vùng quan sát camera
3. Hệ thống tự động:
   - Phát hiện & khoanh vùng biển số (khung xanh)
   - Đọc text biển số & hiển thị ở khung phải
   - Lưu ảnh crop + log vào `database/`
4. **Nhấn `■ DỪNG`** để tắt camera
5. **Nhấn `✕ XÓA`** để xóa lịch sử

## ⚙️ Cấu hình

Chỉnh sửa các hằng số trong `license_plate_detector.py`:

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `CAMERA_INDEX` | `0` | Index camera (0 = webcam mặc định) |
| `OCR_INTERVAL` | `1.0` | Giây giữa các lần quét OCR |
| `CONFIDENCE_THRESHOLD` | `0.3` | Ngưỡng tin cậy tối thiểu |
| `CROP_PADDING` | `15` | Padding (px) quanh biển số khi crop |
| `DUPLICATE_COOLDOWN` | `5` | Giây chờ trước khi lưu lại biển trùng |

## 📝 License

UET License — Đồ án sinh viên Trường Đại học Công nghệ (UET), Đại học Quốc gia Hà Nội (VNU).

