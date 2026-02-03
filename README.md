# 🚗 Road Crack & Pothole Detection System

A Flask-based web application for AI-powered road infrastructure analysis using TensorFlow Lite, optimized for both Raspberry Pi 4 and GPU-enabled systems.

## 🌟 Features

### 🔍 **Intelligent Detection**
- **Road Crack Detection**: Identifies linear cracks in road surfaces
- **Pothole Detection**: Detects circular and irregular potholes
- **Real-time Analysis**: Fast inference optimized for edge devices

### 📱 **Dual Input Methods**
- **Image Upload**: Support for JPG, PNG, GIF, BMP, TIFF formats
- **Camera Capture**: Real-time camera integration with device camera
- **Drag & Drop**: Intuitive file upload interface

### 🧠 **Hardware-Adaptive Models**
- **Raspberry Pi 4 Optimization**: TensorFlow Lite quantized models
- **GPU Acceleration**: Full TensorFlow models for powerful systems
- **Automatic Hardware Detection**: Smart model selection based on capabilities

### 📊 **Comprehensive Results**
- **Visual Detection**: Bounding boxes with confidence scores
- **Detailed Statistics**: Defect counts and classification
- **Result Visualization**: Processed images with overlays

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8+ (tested with Python 3.11)
- 4GB+ RAM (optimized for Raspberry Pi 4)
- Camera support (optional, for real-time capture)

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
/app/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── templates/
│   ├── index.html         # Main application interface
│   └── calibrate.html     # Calibration interface
|   └── report.html        # Report Output interface
│
├── static/
│   ├── uploads/           # Uploaded images
│   └── results/           # Detection results
```

## 🎯 Key Features Implemented

### **✅ Flask Web Application**
- Modern responsive UI with Bootstrap 5
- Real-time hardware detection and optimization
- Comprehensive error handling and logging

### **✅ Dual Detection Methods**
- Image upload with drag-and-drop support
- Camera capture with real-time processing
- Multiple image format support

### **✅ Hardware Optimization**
- Automatic Raspberry Pi detection
- GPU capability assessment
- Memory-aware model selection
- Performance-optimized inference

## 🔧 Hardware-Specific Optimization

### **Raspberry Pi 4 (4GB)**
- **Model**: TensorFlow Lite quantized (INT8)
- **Memory**: Optimized for 4GB RAM constraint
- **Performance**: ~100ms inference time
- **Features**: NEON acceleration, batch size = 1

### **Standard Laptop/PC**
- **Model**: TensorFlow Lite standard (Float32)  
- **Memory**: Standard memory usage
- **Performance**: ~50ms inference time
- **Features**: CPU optimization
