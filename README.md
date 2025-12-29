# 🚗 Real-Time Object Detection for Autonomous Vehicles

An end-to-end **real-time object detection system** for autonomous driving applications using **YOLO11**, featuring **model screening, fine-tuning, MLOps tracking with MLflow**, and **real-time deployment via Streamlit**.

---

## 📌 Project Overview

This project implements a complete **computer vision + MLOps pipeline** to detect critical road objects such as cars, pedestrians, cyclists, and trucks in real-time.  
The system is trained and evaluated on the **KITTI Vision Benchmark dataset** and deployed as an interactive **Streamlit application** supporting webcam, image, and video inference.

---

## 🧠 Key Features

- YOLO11-based object detection (n / s / m variants)
- Automated model screening with accuracy–speed tradeoff
- Transfer learning & fine-tuning
- MLflow experiment tracking & model registry
- Real-time inference deployment using Streamlit
- Supports webcam, image, and video input

---

## 🛠️ Tech Stack

| Category | Tools |
|--------|------|
| Language | Python |
| Framework | Ultralytics YOLO |
| Dataset | KITTI Object Detection |
| Metrics | mAP50, mAP50–95, Precision, Recall, FPS |
| MLOps | MLflow |
| Deployment | Streamlit |
| Visualization | OpenCV, Matplotlib |
| Platform | Kaggle GPU Notebook |

---

## 📂 Dataset

- **Dataset:** KITTI Vision Benchmark Suite (Object Detection)
- **Classes:**  
  `car, van, truck, pedestrian, cyclist, Person_sitting, tram, misc`
- **Format:** YOLO annotation format
- **Challenges:** occlusion, lighting variation, small objects, real-world traffic scenes

YOLO11 automatically handles:
- Image resizing
- Normalization
- Data augmentation
- Tensor conversion during training

---

## ⚙️ Methodology

### 🔹 Model Screening
Three YOLO variants were evaluated:
- **YOLO11n** (2.5M parameters)
- **YOLO11s** (6.0M parameters)
- **YOLO11m** (18.0M parameters)

Each model was:
- Trained for **3 epochs**
- Evaluated using **mAP50, mAP50–95**
- Benchmarked for **FPS**

**Performance Score Formula:**
