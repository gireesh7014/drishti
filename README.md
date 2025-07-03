# Drishti

A computer vision project for face detection and analysis.

## Project Structure

- `AI/` - Core AI and computer vision algorithms
- `backend/` - Server-side code and APIs
- `frontend/` - User interface and client-side code

## Current Features

- Face detection using MediaPipe
- Real-time video processing

## Setup

1. Create virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate virtual environment:
   ```bash
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install opencv-python mediapipe
   ```

4. Run face detection:
   ```bash
   cd AI
   python face-detect.py
   ```

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- Webcam for real-time detection
