# 🎯 Detectra — Object Disappearance Detection System

Detectra is a professional **AI-powered desktop application** designed to detect and document the disappearance of objects in CCTV and video footage. Version 3.0.0 introduces a massive performance and feature update, including hybrid tracking, multi-video support, and advanced forensic visualization.

The system allows users to select an object within a video and intelligently track it across frames using high-performance computer vision. If the object disappears from the scene, Detectra automatically captures high-resolution evidence snapshots, extracts OCR timestamps, and visualizes the object's path for forensic analysis.

---

# ✨ Key Features

### 🚀 Modern UI & UX
*   **Professional Branding**: Custom icons, high-DPI scaling support, and a polished dark-themed interface (Catppuccin).
*   **Splash Screen**: Smooth initial loading experience with real-time status updates for models and dependencies.
*   **Responsive Canvas**: Dynamic video scaling with interactive bounding-box and OCR region selection.

### 🧠 Intelligent Hybrid Tracking
*   **YOLOv8 + CSRT**: Combines state-of-the-art AI detection with OpenCV's CSRT tracker for smooth, reliable, and validated monitoring.
*   **Resume from Disappearance**: Re-draw boxes to track objects that reappear, maintaining forensic continuity.
*   **Multi-Video Queue**: Batch process multiple CCTV files with automated advancement.

### 🔍 Forensic Evidence & Path Visualization
*   **Breadcrumb Investigation**: Visualizes the object's trajectory with colored markers (Start: Green, Path: Yellow, Last-seen: Red).
*   **Precision OCR Selection**: Manually define OCR regions to accurately extract timestamps from any DVR layout.
*   **Night Mode (CLAHE)**: Integrated contrast enhancement for improved detection in low-light or grainy footage.

### ⚡ Performance & Portability
*   **Variable Speed (1x to 30x)**: Physically seeks through frames for ultra-fast processing of long-duration footage without UI lag.
*   **Standalone EXE**: Packaged as a single-file portable Windows executable with localized model storage.

---

# 🛠️ System Requirements

*   **OS**: Windows 10 or 11 (64-bit)
*   **RAM**: 8 GB (16 GB Recommended)
*   **GPU**: NVIDIA GPU with CUDA support recommended (but runs on CPU)
*   **Dependencies**: The standalone version requires an internet connection on the *first launch only* to download models.

---

# 🚀 Installation & Build

### Option 1 — Run the Executable (Recommended)
1. Download **`Detectra_v3.0.0.exe`** from the [Releases](https://github.com/NEIL-DANIEL-A/Detectra/releases) page.
2. Run the file directly. No installation is required.

### Option 2 — Developer Setup (Source Code)
1. Clone the repo: `git clone https://github.com/NEIL-DANIEL-A/Detectra.git`
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install requirements: `pip install -r requirements.txt`
5. Run: `python main.py`

---

# 📂 Project Structure

| File | Description |
| :--- | :--- |
| **`main.py`** | Application entry point, UI management, and Splash Screen. |
| **`tracker.py`** | Core logic for Hybrid Tracking (YOLOv8+CSRT) and OCR. |
| **`requirements.txt`** | Python dependencies for the developer environment. |
| **`icon.ico`** | High-resolution application branding. |

---

# 🧪 Technologies Used

*   **Python 3.10+ & Tkinter** — Core application framework
*   **Ultralytics YOLOv8** — State-of-the-art object detection
*   **OpenCV & CSRT** — Advanced video processing and tracking
*   **EasyOCR** — Optical Character Recognition for forensic timestamps
*   **PyInstaller** — Secure executable distribution

---

# 📈 Current Stage & Versioning

**Current version: v3.0.0**

### What's New in v3.0.0:
- [x] **Hybrid Tracker**: Replaced basic YOLO tracking with a validated **YOLO + CSRT** engine.
- [x] **Multi-File Queue**: Support for batch processing multiple videos.
- [x] **Path Visualization**: Real-time breadcrumb trails and forensic path capture.
- [x] **OCR Region Tool**: Custom selection of timestamp areas for better accuracy.
- [x] **Resume Feature**: Ability to continue tracking from a disappearance point.
- [x] **CLAHE Preprocessing**: Improved visibility for dark and low-contrast footage.

---

## 👥 Contributors

- **[Neil Daniel A](https://github.com/NEIL-DANIEL-A)**   
- **[Monica B](https://github.com/Monica-403)**

---

# 📄 License

This project is developed for educational and research purposes.
