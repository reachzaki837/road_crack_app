# 🚗 Road Crack & Pothole Detection System

A Flask-based web application for AI-powered road infrastructure analysis using TensorFlow and OpenCV, designed to detect road cracks and potholes in real time. The system is optimized for both Raspberry Pi 4 edge deployments and standard laptop/PC environments.

## 📋 Overview

This application captures or accepts road surface images, runs computer vision inference to detect cracks and potholes, computes a **Road Health Index (RHI)**, and presents results through an interactive web dashboard. It is intended for field engineers, municipal authorities, and researchers monitoring road conditions.

## 🌟 Features

### 🔍 **Intelligent Detection**
- **Road Crack Detection**: Identifies linear cracks in road surfaces
- **Pothole Detection**: Detects circular and irregular potholes
- **Road Health Index (RHI)**: Composite score summarizing overall road condition
- **Real-time Analysis**: Fast inference optimized for edge devices

### 📱 **Multiple Input Methods**
- **Image Upload**: Support for common image formats (JPG, PNG, GIF, BMP, TIFF)
- **Camera Capture**: Live camera integration with configurable capture intervals
- **Perspective Calibration**: Interactive 4-point calibration tool for accurate road measurements

### 🧠 **Hardware-Adaptive Models**
- **Raspberry Pi 4 Optimization**: TensorFlow Lite quantized models (INT8)
- **Standard PC Support**: Full TensorFlow Float32 models
- **Automatic Hardware Detection**: Smart model selection based on device capabilities

### 📊 **Reporting & Visualization**
- **Live Dashboard**: Displays RHI, crack percentage, and pothole count with color-coded severity badges
- **Historical Reports**: Tabular logs and trend graphs (Road Health over time)
- **Predictive Analytics**: Forecasts road health for the next 5 capture intervals
- **Result Overlays**: Processed images with detection bounding boxes and confidence scores

### ⚙️ **CI/CD Pipeline**
- **Jenkinsfile** included with Build, Test, and SonarQube Analysis stages
- Auto-heal branches managed automatically for pipeline failure recovery

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8+ (tested with Python 3.11)
- 4 GB+ RAM (optimized for Raspberry Pi 4)
- Camera module (optional, for live capture)

### **Installation & Running**

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python app.py
   ```

3. **Access the web interface**:
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
road_crack_app/
├── app.py                 # Main Flask application (detection logic & routes)
├── requirements.txt       # Python dependencies
├── Jenkinsfile            # CI/CD pipeline (Build → Test → SonarQube)
├── README.md              # This file
│
├── templates/
│   ├── index.html         # Live dashboard (RHI, crack %, potholes, capture control)
│   ├── calibrate.html     # Perspective calibration interface (4-point selection)
│   └── report.html        # Historical log report with trend graph & predictions
│
└── static/
    ├── uploads/           # Uploaded input images
    └── results/           # Detection output images
```

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| Flask | 3.0.0 | Web framework |
| TensorFlow | 2.15.0 | AI inference engine |
| OpenCV | 4.8.1.78 | Image processing |
| NumPy | 1.24.3 | Numerical computation |
| Pillow | 10.1.0 | Image I/O |
| psutil | 5.9.6 | Hardware detection |
| Werkzeug | 3.0.1 | WSGI utilities |
| gunicorn | 21.2.0 | Production WSGI server |

## 🌿 Branches

| Branch | Description |
|---|---|
| `main` | Stable production branch |
| `copilot/update-readme-description` | README documentation update |
| `autoheal-*` | Auto-generated branches for CI/CD pipeline failure recovery |

## 🎯 Key Features Implemented

### **✅ Flask Web Application**
- Responsive UI using Bootstrap 5
- Real-time hardware detection and model selection
- Configurable camera capture intervals via the dashboard
- Comprehensive error handling and logging

### **✅ Detection & Analysis**
- Image upload and live camera capture support
- Road crack percentage and pothole count computation
- RHI-based condition classification (Good / Warning / Critical) with color badges

### **✅ Calibration**
- Interactive perspective calibration via 4-point canvas selection
- Upload a road image and save calibration for accurate spatial measurements

### **✅ Reporting**
- Bootstrap-styled tabular log of historical readings
- Matplotlib trend graph rendered inline
- 5-step ahead RHI prediction table

## 🔧 Hardware-Specific Optimization

### **Raspberry Pi 4 (4 GB)**
- **Model**: TensorFlow Lite quantized (INT8)
- **Memory**: Optimized for 4 GB RAM constraint
- **Performance**: ~100 ms inference time
- **Features**: NEON acceleration, batch size = 1

### **Standard Laptop / PC**
- **Model**: TensorFlow Lite standard (Float32)
- **Memory**: Standard memory usage
- **Performance**: ~50 ms inference time
- **Features**: CPU optimization
