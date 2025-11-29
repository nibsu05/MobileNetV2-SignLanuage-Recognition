# Sign Language Letters Recognition

Nhận diện chữ cái ngôn ngữ ký hiệu sử dụng MobileNetV2 

## Demo
[![Demo Video](thumbnail.png)](demo.mp4)


## Dữ liệu (Dataset)
Bộ dữ liệu đã được xử lý và đóng gói. Bạn có thể tải về tại đây:
- **Google Drive**: [Download ASL Dataset.zip](https://drive.google.com/file/d/1yLIR-gLw5P_qvDDpT73ebCQ8rb0bCVol/view?usp=sharing)
- Sau khi tải về, giải nén vào thư mục gốc của dự án (sẽ tạo ra thư mục `data/`).

## Kết quả & Trực quan hóa

### 1. Quy trình tiền xử lý (MediaPipe -> Gray -> CLAHE)
![Preprocessing Pipeline](results/preprocessing_comparison.png)
*So sánh ảnh gốc, ảnh xám và ảnh sau khi cân bằng sáng CLAHE.*

### 2. Mẫu dữ liệu sau khi xử lý
![Sample Batch](results/sample_batch.png)
*Các mẫu ảnh tay đã được cắt và xóa nền.*

### 3. Phân phối dữ liệu
![Class Distribution](results/data_train_class_distribution.png)
*Phân phối số lượng ảnh giữa các lớp trong tập huấn luyện.*

### 4. Quá trình huấn luyện (MobileNetV2)
![Training Plot](results/moblienetv2_training_plot.png)
*Biểu đồ Loss và Accuracy qua các Epochs.*

### 5. Đánh giá trên tập Test
**Ma trận nhầm lẫn (Confusion Matrix):**
![Confusion Matrix](results/test_confusion_matrix.png)
*Hiển thị độ chính xác chi tiết cho từng lớp ký tự.*

**Dự đoán mẫu (Test Predictions):**
![Test Predictions](results/test_prediction.png)
*Một số kết quả dự đoán thực tế của model trên tập test.*

## Cấu trúc dự án

### 1. Ứng dụng chính
- **`app.py`**: Chương trình nhận diện thời gian thực qua Webcam.
  - Tích hợp MediaPipe để xóa nền.
  - Giao diện trực quan: Webcam + Model Input + Gõ chữ.

### 2. Huấn luyện & Đánh giá Model
- **`train_mobilenet.ipynb`**: Notebook huấn luyện model MobileNetV2 từ đầu (Scratch).
- **`test_mobilenet.ipynb`**: Notebook đánh giá model trên tập test (Accuracy, Confusion Matrix).
- **`asl_mobilenet_v2.pth`**: File trọng số model đã huấn luyện.

### 3. Xử lý dữ liệu
- **`preprocess_remove_bg.py`**: Xử lý dữ liệu thô:
  - Dùng MediaPipe cắt tay và xóa nền đen.
  - Tự động Augmentation (xoay, chỉnh sáng) để cân bằng dữ liệu (500 ảnh/lớp).
- **`split_data.py`**: Chia dữ liệu thành 3 tập: Train (80%), Valid (10%), Test (10%).
- **`visualize_preprocessing.py`**: Biểu đồ hóa dữ liệu:
  - Phân phối lớp (Class Distribution).
  - So sánh trước/sau tiền xử lý (Grayscale, CLAHE).

### 4. Thu thập dữ liệu
- **`collect_data_webcam.py`**: Tool tự thu thập dữ liệu từ Webcam cá nhân.
- **`crawl_data.py`**: Tool tải ảnh từ internet (Bing) dùng thư viện `icrawler`.

### 5. Khác
- **`check_gpu.py`**: Kiểm tra PyTorch có nhận GPU (CUDA) không.
- **`requirements.txt`**: Danh sách các thư viện cần thiết.




