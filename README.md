# Detectra — Object Disappearance Detection System

Detectra is an **AI-powered desktop application** designed to detect and document the disappearance of objects in CCTV and video footage.

The system allows users to select an object within a video and intelligently track it across frames. If the object disappears from the scene, Detectra automatically captures and stores evidence frames for analysis.

Detectra is distributed as a **portable Windows executable**, so it can run without installing Python or additional dependencies.

# Features

## Intelligent AI Tracking

Detectra uses modern computer vision techniques to ensure reliable object tracking.

* Object detection powered by **Ultralytics YOLOv8**
* Persistent tracking using ByteTrack
* Manual object selection through a drag-and-select interface
* Robust tracking across motion, partial occlusion, and scene changes

## Disappearance Detection

Detectra continuously monitors the tracked object and detects when it leaves the frame or can no longer be detected.

Capabilities include:

* Automatic disappearance detection
* Real-time visual tracking feedback
* Frame-skipping logic for faster processing
* Adjustable tracking speed for long CCTV footage

## Evidence Capture and Reporting

When a disappearance event occurs, Detectra automatically saves evidence frames.

Captured outputs include:

* Frame before disappearance
* Frame after disappearance
* Timestamp of disappearance
* OCR extraction of timestamps from the video

Saved outputs are organized into unique folders to prevent overwriting.

## Desktop Application

Detectra is packaged as a standalone Windows application using **PyInstaller**.

Key benefits:

* No Python installation required
* All dependencies bundled into a single executable
* Portable and easy to distribute

# System Requirements

Recommended specifications:

* Windows 10 or Windows 11
* Minimum 8 GB RAM
* Recommended 16 GB RAM
* CPU inference supported
* GPU optional but improves performance

# Installation

You can run Detectra in two ways.

## Option 1 — Run the Executable (Recommended)

1. Go to the **Releases** page of this repository.
2. Download the latest release.
3. Run:

```
Detectra.exe
```

No installation is required.

## Option 2 — Run from Source Code

Clone the repository:

```
git clone https://github.com/NEIL-DANIEL-A/Detectra.git
cd detectra
```

Create a virtual environment:

```
python -m venv venv
```

Activate the environment:

Windows

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
python main.py
```


# Usage

Using Detectra is simple.

1. Launch the application.
2. Upload a video file.
3. Select the object to track by drawing a bounding box.
4. Click **Start Tracking**.
5. Detectra will monitor the object throughout the video.

If the object disappears, Detectra automatically records the event and saves evidence frames.

# Project Structure

```
Detectra/
│
main.py
tracker.py
requirements.txt
yolov8n.pt
README.md
```

Description:

| File       | Purpose                                           |
| ---------- | ------------------------------------------------- |
| main.py    | Application entry point                           |
| tracker.py | Object tracking and disappearance detection logic |
| yolov8n.pt | YOLO model weights                                |


# Technologies Used

Detectra is built using the following technologies:

* **Ultralytics YOLOv8** — Object detection
* **PyTorch** — Deep learning framework
* **OpenCV** — Video processing
* **EasyOCR** — Timestamp extraction
* **PyInstaller** — Desktop application packaging


# Known Limitations

* Extremely small objects may be harder to track reliably
* Crowded scenes with similar objects may cause tracking switches
* Large videos may require longer processing times

# Future Improvements

Planned features include:

* Multi-object tracking support
* Faster inference optimization
* Real-time camera monitoring
* Improved user interface


# Release

Current version:

```
v1.1.1
```

See the **Releases** page for downloadable builds.


# License

This project is released for educational and research purposes.
