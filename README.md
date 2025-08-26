# Motion Comparison Dashboard

A Streamlit app to compare two athletes (or two trials of the same athlete) side-by-side from video.

## ✨ Features
- Upload two videos
- Pose detection with **MediaPipe**
- Skeleton overlay and annotated videos
- Joint angles, speed, stride length (approx), jump height (approx)
- Interactive charts and **radar** comparison
- Optional rough calibration to meters by entering athlete height

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📝 Notes
- Distances/speeds are **pixel-based** unless you provide athlete height to calibrate (rough).
- For reliable stride estimates, use a **side-view** running video.
- To add OpenPose, create a function that outputs the same DataFrame fields as `mediapipe_pose_process()`.