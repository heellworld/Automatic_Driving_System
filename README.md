# 🚗 Autonomous Driving System

> Hệ thống lái xe tự động thông minh sử dụng AI và Computer Vision

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ Tính năng chính

- 🧠 **Deep Learning**: CNN model dự đoán góc lái từ camera
- 👁️ **Computer Vision**: Phát hiện làn đường real-time  
- 🎮 **Simulator Support**: Tương thích với Unity simulators
- 📱 **Webcam Testing**: Test trực tiếp qua webcam

## 🚀 Quick Start

### 1. Cài đặt nhanh
```bash
# Clone và setup
git clone https://github.com/heellworld/Automatic_Driving_System.git
cd Automatic_Driving_System
pip install -r requirements.txt
```

### 2. Demo nhanh
```bash
# Chạy interactive demo
python scripts/quick_start.py

# Hoặc test ngay với webcam
python drive_realtime.py --approach lane --interface webcam
```

### 3. Simulator Mode
```bash
# Khởi động simulator trước
./beta_simulator_windows/beta_simulator.exe

# Kết nối autonomous driving
python drive_realtime.py --approach cnn --interface simulator
```

## 🧠 Hai phương pháp chính

| Phương pháp | Ưu điểm | Nhược điểm | Use Case |
|-------------|---------|------------|----------|
| **CNN** | Accurate, End-to-end learning | Cần training data lớn | Complex scenarios |
| **Computer Vision** | Fast, Explainable | Limited to lane following | Simple road conditions |
| **Hybrid** | Best of both worlds | More complex | Production ready |

## 📊 Performance

- **Real-time**: 30+ FPS
- **Accuracy**: 95%+ trên test set
- **Model size**: 16MB (CNN)
- **Latency**: <50ms per frame

## 🛠️ Scripts chính

| Script | Mô tả | Ví dụ |
|--------|--------|--------|
| `train_cnn.py` | Training CNN model | `python train_cnn.py --epochs 20` |
| `test_model.py` | Test và đánh giá model | `python test_model.py --webcam` |
| `drive_realtime.py` | Driving real-time | `python drive_realtime.py --approach hybrid` |
| `scripts/quick_start.py` | Interactive demo | `python scripts/quick_start.py` |

## 📁 Cấu trúc dự án

```
📦 Autonomous_Driving_System/
├── 🧠 src/                    # Core source code
│   ├── models/               # AI models
│   ├── config.py            # Configurations  
│   └── utils.py             # Utilities
├── 🏋️ train_cnn.py           # Training script
├── 🧪 test_model.py          # Testing script
├── 🚗 drive_realtime.py      # Real-time driving
├── 📊 Data/                  # Training dataset (~28K images)
└──  🎮 beta_simulator_windows/ # Unity simulator
```

## ⚙️ Configuration

Chỉnh sửa `src/config.py` để tuning:

```python
# Training
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Simulator  
SPEED_LIMIT = 20
PORT = 4567

# Lane Detection
CANNY_LOW = 150
CANNY_HIGH = 200
```

## 🔧 Advanced Usage

### Training từ đầu
```bash
# Chuẩn bị data trong data/
python train_cnn.py --epochs 50 --batch-size 64 --model-name my_model.h5
```

### Testing comprehensive
```bash
# Test accuracy
python test_model.py --cnn-model models/cnn_model.h5 --test-data data/driving_log.csv

# So sánh models
python test_model.py --compare --test-images data/IMG/
```

### Custom approaches
```bash
# Hybrid với custom weights
python drive_realtime.py --approach hybrid --model models/custom.h5

# Record driving session
python drive_realtime.py --approach cnn --record
```

## 🐛 Troubleshooting

<details>
<summary><b>❌ Common Issues</b></summary>

**Webcam không hoạt động:**
```python
# Thử camera index khác
python drive_realtime.py --camera 0  # thay vì 1
```

**Simulator không kết nối:**
```bash
# Kiểm tra port
netstat -an | findstr 4567
```

**Model loading error:**
```bash
# Kiểm tra TensorFlow version
pip install tensorflow==2.8.0
```

</details>

---

<div align="center">

**⭐ Star this repo if you find it helpful! ⭐**

Made with ❤️ for autonomous driving enthusiasts

</div> 