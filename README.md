# 🎯 Detectra — Object Disappearance Detection System

Detectra is a professional **AI-powered desktop application** designed to detect and document the disappearance of objects in CCTV and video footage. Version 2.1.0 introduces a comprehensive UI/UX overhaul, featuring a sleek modern design, automated setup, and enhanced stability.

The system allows users to select an object within a video and intelligently track it across frames using high-performance computer vision. If the object disappears from the scene, Detectra automatically captures high-resolution evidence snapshots for forensic analysis.

---

# ✨ Key Features

### 🚀 Modern UI & UX
*   **Professional Branding**: Custom icons, high-DPI scaling support, and a polished dark-themed interface.
*   **Splash Screen**: Smooth initial loading experience with real-time status updates.
*   **Responsive Canvas**: Dynamic video scaling and interactive bounding-box selection.

### 🧠 Intelligent AI Tracking
*   **YOLOv8 & ByteTrack**: State-of-the-art object detection and persistent tracking for reliable monitoring.
*   **Live Preview**: Real-time visual feedback of the tracking process.
*   **Variable Speed**: Adjustable processing speeds (1x to 10x) to handle long-duration CCTV footage efficiently.

### 🔍 Forensic Evidence Capture
*   **Automatic Snapshots**: Captures "Before" and "After" frames the moment an object is no longer detected.
*   **OCR Timestamp Extraction**: Automatically extracts on-screen timestamps from CCTV feeds (Alpha).
*   **Export Management**: Organized results export with timestamped folders.

### 📦 Standalone Portability
*   **Zero-Install EXE**: Packaged as a single-file portable Windows executable.
*   **Automated Setup**: On first launch, Detectra automatically downloads and configures the required AI models.

---

# 🛠️ System Requirements

*   **OS**: Windows 10 or 11 (64-bit)
*   **RAM**: 8 GB (16 GB Recommended)
*   **Dependencies**: The standalone version requires an active internet connection on the *first launch only* to download models.

---

# 🚀 Installation & Build

### Option 1 — Run the Executable (Recommended)
1. Download **`Detectra_v2.1.0.exe`** from the [Releases](https://github.com/NEIL-DANIEL-A/Detectra/releases) page.
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
| **`tracker.py`** | Core AI logic for YOLOv8 detection and object tracking. |
| **`setup.py`** | Automated first-run downloader for AI models and weights. |
| **`icon.ico`** | High-resolution application branding. |

---

# 🧪 Technologies Used

*   **Python 3.10+ & Tkinter** — Core application framework
*   **Ultralytics YOLOv8** — State-of-the-art object detection
*   **OpenCV & PIL** — Advanced image and video processing
*   **EasyOCR** — Optical Character Recognition for timestamps
*   **PyInstaller** — Secure executable distribution

---

# 📈 Current Stage & Versioning

**Current version: v2.1.0**

### What's New in v2.1.0:
- [x] Complete UI redesign with **Catppuccin** inspired color palette.
- [x] Added **Windows Taskbar integration** and AppUserModelID fix.
- [x] Integrated **Automated Model Setup** (no more manual downloads).
- [x] Implemented **Splash Screen** with loading progress tracking.
- [x] Fixed DPI scaling issues for high-resolution displays.

---

## 👥 Contributors

- **[Neil Daniel A](https://github.com/NEIL-DANIEL-A)**   
- **[Monica B](https://github.com/Monica-403)**

---

# 📄 License

This project is developed for educational and research purposes.
