# Motion Comparison Dashboard (Streamlit)
# ---------------------------------------
# Features:
# - Upload two videos
# - Pose extraction with MediaPipe
# - Skeleton overlay & annotated video export
# - Compute joint angles, speed, stride length, jump height
# - Timeline alignment & normalization
# - Visual comparisons (curves + radar) with Plotly
# Run: streamlit run app.py

import os
import io
import math
import tempfile
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Try to import mediapipe; show friendly message if missing.
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception as e:
    MP_AVAILABLE = False

st.set_page_config(
    page_title="Motion Comparison Dashboard",
    layout="wide"
)

# --------------------------
# Utility Math
# --------------------------
def angle_between(p1, p2, p3):
    """
    Returns the angle at p2 made by points p1-p2-p3 in degrees.
    Points are (x, y) tuples in pixels.
    """
    if p1 is None or p2 is None or p3 is None:
        return np.nan
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return np.nan
    cosang = np.clip(np.dot(a, b) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def euclidean(p1, p2):
    if p1 is None or p2 is None:
        return np.nan
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def moving_average(x, k=5):
    if len(x) < 3 or k <= 1:
        return np.array(x, dtype=float)
    return np.convolve(x, np.ones(k)/k, mode='same')

def detect_peaks(x, mph=None, mpd=10):
    """
    Very simple local peak detector.
    x: 1D numpy array
    mph: minimum peak height (optional)
    mpd: minimum peak distance (samples)
    Returns indices of peaks.
    """
    x = np.asarray(x)
    if len(x) < 3:
        return np.array([], dtype=int)
    dx = np.diff(x)
    # candidate peaks: derivative crosses from + to -
    candidates = np.where((np.hstack([dx, 0]) <= 0) & (np.hstack([0, dx]) > 0))[0]
    if mph is not None:
        candidates = candidates[x[candidates] >= mph]
    # enforce minimum peak distance
    filtered = []
    last = -np.inf
    for c in candidates:
        if len(filtered) == 0 or (c - last) >= mpd:
            filtered.append(c)
            last = c
    return np.array(filtered, dtype=int)

# --------------------------
# Pose Extraction & Video Annotation
# --------------------------

LANDMARK_NAMES = [
    'NOSE','LEFT_EYE_INNER','LEFT_EYE','LEFT_EYE_OUTER','RIGHT_EYE_INNER','RIGHT_EYE','RIGHT_EYE_OUTER',
    'LEFT_EAR','RIGHT_EAR','MOUTH_LEFT','MOUTH_RIGHT','LEFT_SHOULDER','RIGHT_SHOULDER','LEFT_ELBOW',
    'RIGHT_ELBOW','LEFT_WRIST','RIGHT_WRIST','LEFT_PINKY','RIGHT_PINKY','LEFT_INDEX','RIGHT_INDEX',
    'LEFT_THUMB','RIGHT_THUMB','LEFT_HIP','RIGHT_HIP','LEFT_KNEE','RIGHT_KNEE','LEFT_ANKLE','RIGHT_ANKLE',
    'LEFT_HEEL','RIGHT_HEEL','LEFT_FOOT_INDEX','RIGHT_FOOT_INDEX'
]

POSE_CONNECTIONS = None
if MP_AVAILABLE:
    POSE_CONNECTIONS = list(mp.solutions.pose.POSE_CONNECTIONS)

def mediapipe_pose_process(video_path, out_video_path, min_det=0.5, min_track=0.5, model_complexity=1):
    """
    Runs MediaPipe Pose on a video, writes an annotated video, and returns a DataFrame with landmark xy per frame.
    DataFrame columns: frame, time, landmark_<name>_x, landmark_<name>_y, visibility_<name>
    Also includes derived metrics per frame (speeds and joint angles).
    """
    if not MP_AVAILABLE:
        raise RuntimeError("MediaPipe is not available in this environment. Please install mediapipe.")

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    # Use a video codec that is compatible with web browsers (H.264)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    records = []
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_track
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # Default: no landmarks
            row = {'frame': frame_idx, 'time': frame_idx / fps}

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # Collect landmark coords in pixel space
                pts = {}
                for i, name in enumerate(LANDMARK_NAMES):
                    x = lm[i].x * width
                    y = lm[i].y * height
                    v = lm[i].visibility
                    row[f'landmark_{name}_x'] = x
                    row[f'landmark_{name}_y'] = y
                    row[f'visibility_{name}'] = v
                    pts[name] = (x, y)

                # Derived points
                mid_hip = None
                if pts.get('LEFT_HIP') and pts.get('RIGHT_HIP'):
                    mid_hip = ((pts['LEFT_HIP'][0] + pts['RIGHT_HIP'][0]) / 2.0,
                               (pts['LEFT_HIP'][1] + pts['RIGHT_HIP'][1]) / 2.0)
                mid_shoulder = None
                if pts.get('LEFT_SHOULDER') and pts.get('RIGHT_SHOULDER'):
                    mid_shoulder = ((pts['LEFT_SHOULDER'][0] + pts['RIGHT_SHOULDER'][0]) / 2.0,
                                    (pts['LEFT_SHOULDER'][1] + pts['RIGHT_SHOULDER'][1]) / 2.0)

                # Joint angles (deg)
                L_elbow = angle_between(pts.get('LEFT_SHOULDER'), pts.get('LEFT_ELBOW'), pts.get('LEFT_WRIST'))
                R_elbow = angle_between(pts.get('RIGHT_SHOULDER'), pts.get('RIGHT_ELBOW'), pts.get('RIGHT_WRIST'))
                L_knee = angle_between(pts.get('LEFT_HIP'), pts.get('LEFT_KNEE'), pts.get('LEFT_ANKLE'))
                R_knee = angle_between(pts.get('RIGHT_HIP'), pts.get('RIGHT_KNEE'), pts.get('RIGHT_ANKLE'))
                L_hip = angle_between(pts.get('LEFT_SHOULDER'), pts.get('LEFT_HIP'), pts.get('LEFT_KNEE'))
                R_hip = angle_between(pts.get('RIGHT_SHOULDER'), pts.get('RIGHT_HIP'), pts.get('RIGHT_KNEE'))

                row.update({
                    'L_elbow': L_elbow, 'R_elbow': R_elbow,
                    'L_knee': L_knee, 'R_knee': R_knee,
                    'L_hip': L_hip, 'R_hip': R_hip,
                    'mid_hip_x': mid_hip[0] if mid_hip else np.nan,
                    'mid_hip_y': mid_hip[1] if mid_hip else np.nan,
                    'mid_shoulder_x': mid_shoulder[0] if mid_shoulder else np.nan,
                    'mid_shoulder_y': mid_shoulder[1] if mid_shoulder else np.nan,
                })

                # Draw landmarks
                overlay = frame.copy()
                mp_drawing.draw_landmarks(
                    overlay,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(thickness=2)
                )
                frame = overlay
            else:
                # fill missing landmark cols with NaNs to keep schema stable
                for name in LANDMARK_NAMES:
                    row[f'landmark_{name}_x'] = np.nan
                    row[f'landmark_{name}_y'] = np.nan
                    row[f'visibility_{name}'] = np.nan
                row.update({
                    'L_elbow': np.nan, 'R_elbow': np.nan,
                    'L_knee': np.nan, 'R_knee': np.nan,
                    'L_hip': np.nan, 'R_hip': np.nan,
                    'mid_hip_x': np.nan, 'mid_hip_y': np.nan,
                    'mid_shoulder_x': np.nan, 'mid_shoulder_y': np.nan,
                })

            writer.write(frame)
            records.append(row)
            frame_idx += 1

    cap.release()
    writer.release()

    df = pd.DataFrame.from_records(records)
    # Pixel/s speed based on mid-hip
    df['speed_px_s'] = df['mid_hip_x'].diff().pow(2).add(df['mid_hip_y'].diff().pow(2)).pow(0.5) * (fps)
    df['speed_px_s'] = df['speed_px_s'].fillna(0.0)
    return df

# --------------------------
# Metric Computation
# --------------------------
def compute_stride_metrics(df, fps, which='LEFT'):
    """
    Estimate steps/stride from ankle X trajectory (smoothed).
    Returns dict with step_rate (steps/s), avg_step_length (px), num_steps.
    """
    ankle_x = df.get(f'landmark_{which}_ANKLE_x')
    if ankle_x is None:
        return {'step_rate': np.nan, 'avg_step_length_px': np.nan, 'num_steps': 0}
    x = ankle_x.to_numpy(dtype=float)
    x_s = moving_average(x, k=7)
    # peaks on positive direction
    peaks = detect_peaks(x_s, mph=None, mpd=int(max(5, fps*0.2)))
    num_steps = len(peaks)
    # step lengths as |x[i]-x[i-1]| between consecutive peaks
    step_lengths = []
    for i in range(1, len(peaks)):
        step_lengths.append(abs(x_s[peaks[i]] - x_s[peaks[i-1]]))
    avg_step_length = float(np.nanmean(step_lengths)) if step_lengths else np.nan
    duration_s = df['time'].iloc[-1] - df['time'].iloc[0] if len(df) > 1 else np.nan
    step_rate = num_steps / duration_s if duration_s and duration_s > 0 else np.nan
    return {'step_rate': step_rate, 'avg_step_length_px': avg_step_length, 'num_steps': num_steps}

def compute_jump_metrics(df):
    """
    Very rough jump height estimate using mid-hip vertical displacement (pixels).
    Positive jump height = max drop from standing baseline (lower y = higher in image).
    Returns jump_height_px and airtime_s (based on velocity sign changes).
    """
    y = df['mid_hip_y'].to_numpy(dtype=float)
    if len(y) == 0 or np.all(np.isnan(y)):
        return {'jump_height_px': np.nan, 'airtime_s': np.nan}
    # baseline = median of lowest 20% (standing)
    finite_y = y[np.isfinite(y)]
    if len(finite_y) == 0:
        return {'jump_height_px': np.nan, 'airtime_s': np.nan}
    baseline = np.percentile(finite_y, 80)
    highest = np.nanmin(y)
    jump_height_px = max(0.0, baseline - highest)

    # airtime by zero-crossings of vertical velocity around the jump arc
    vy = np.gradient(-y)  # up positive
    signs = np.sign(vy)
    zero_crossings = np.where(np.diff(signs) != 0)[0]
    airtime_s = np.nan
    if len(zero_crossings) >= 2:
        # take widest two crossings
        airtime_frames = zero_crossings[-1] - zero_crossings[0]
        # fps must be inferred from df['time']
        times = df['time'].to_numpy()
        if len(times) > 1:
            dt = times[1] - times[0]
            if dt > 0:
                airtime_s = airtime_frames * dt
    return {'jump_height_px': float(jump_height_px), 'airtime_s': float(airtime_s) if airtime_s==airtime_s else np.nan}

def summary_metrics(df, fps, px_per_meter=None):
    """
    Build a summary dict. If px_per_meter is provided, convert to m and m/s.
    """
    speed = df['speed_px_s'].to_numpy(dtype=float)
    speed_s = moving_average(speed, k=7)
    avg_speed_px = float(np.nanmean(speed_s))
    max_speed_px = float(np.nanmax(speed_s)) if np.isfinite(speed_s).any() else np.nan

    stride_L = compute_stride_metrics(df, fps, 'LEFT')
    stride_R = compute_stride_metrics(df, fps, 'RIGHT')
    jump = compute_jump_metrics(df)

    # joint angle ROM
    knee_rom = float((df[['L_knee','R_knee']].max().max() - df[['L_knee','R_knee']].min().min()))
    hip_rom = float((df[['L_hip','R_hip']].max().max() - df[['L_hip','R_hip']].min().min()))
    elbow_rom = float((df[['L_elbow','R_elbow']].max().max() - df[['L_elbow','R_elbow']].min().min()))

    # Convert to metric units if calibration provided
    def conv_len(px):
        if px_per_meter and px_per_meter > 0:
            return px / px_per_meter
        return np.nan

    def conv_speed(px_s):
        if px_per_meter and px_per_meter > 0:
            return px_s / px_per_meter
        return np.nan

    out = {
        'avg_speed_px_s': avg_speed_px,
        'max_speed_px_s': max_speed_px,
        'step_rate_L_steps_s': stride_L['step_rate'],
        'step_rate_R_steps_s': stride_R['step_rate'],
        'avg_step_length_L_px': stride_L['avg_step_length_px'],
        'avg_step_length_R_px': stride_R['avg_step_length_px'],
        'jump_height_px': jump['jump_height_px'],
        'airtime_s': jump['airtime_s'],
        'knee_ROM_deg': knee_rom,
        'hip_ROM_deg': hip_rom,
        'elbow_ROM_deg': elbow_rom,
    }
    # Metric conversions
    out.update({
        'avg_speed_m_s': conv_speed(avg_speed_px),
        'max_speed_m_s': conv_speed(max_speed_px),
        'avg_step_length_L_m': conv_len(stride_L['avg_step_length_px']) if stride_L['avg_step_length_px']==stride_L['avg_step_length_px'] else np.nan,
        'avg_step_length_R_m': conv_len(stride_R['avg_step_length_px']) if stride_R['avg_step_length_px']==stride_R['avg_step_length_px'] else np.nan,
        'jump_height_m': conv_len(jump['jump_height_px']) if jump['jump_height_px']==jump['jump_height_px'] else np.nan,
    })
    return out

def estimate_px_per_meter(df, assumed_body_height_m=None):
    """
    Optional rough calibration: use average pixel distance from mid-shoulder to ankle as proxy for body height.
    """
    if not assumed_body_height_m or assumed_body_height_m <= 0:
        return None
    # Estimate "pixel body height"
    y_sh = df['mid_shoulder_y'].to_numpy(dtype=float)
    y_la = df['landmark_LEFT_ANKLE_y'].to_numpy(dtype=float)
    y_ra = df['landmark_RIGHT_ANKLE_y'].to_numpy(dtype=float)
    # Use whichever ankle is lower (bigger y) to reduce missing data
    y_ankle = np.nanmax(np.vstack([y_la, y_ra]), axis=0)
    valid = np.isfinite(y_sh) & np.isfinite(y_ankle)
    if valid.sum() < 5:
        return None
    px_body = np.nanmedian(y_ankle[valid] - y_sh[valid])
    if px_body <= 0:
        return None
    px_per_meter = px_body / float(assumed_body_height_m)
    return px_per_meter

# --------------------------
# Plotting
# --------------------------
def plot_time_series(df_a, df_b, col, label_a, label_b, title, yaxis):
    fig = go.Figure()
    if col in df_a.columns:
        fig.add_trace(go.Scatter(x=df_a['time'], y=df_a[col], mode='lines', name=label_a))
    if col in df_b.columns:
        fig.add_trace(go.Scatter(x=df_b['time'], y=df_b[col], mode='lines', name=label_b))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title=yaxis, height=350, legend_title_text="Athlete")
    return fig

def plot_radar(metrics_a, metrics_b, label_a, label_b):
    # Choose a consistent set of metrics
    fields = [
        ('avg_speed_px_s', 'Avg Speed (px/s)'),
        ('max_speed_px_s', 'Max Speed (px/s)'),
        ('step_rate_L_steps_s', 'Step rate L (1/s)'),
        ('step_rate_R_steps_s', 'Step rate R (1/s)'),
        ('avg_step_length_L_px', 'Step length L (px)'),
        ('avg_step_length_R_px', 'Step length R (px)'),
        ('jump_height_px', 'Jump height (px)'),
        ('knee_ROM_deg', 'Knee ROM (°)'),
        ('hip_ROM_deg', 'Hip ROM (°)'),
        ('elbow_ROM_deg', 'Elbow ROM (°)'),
    ]
    cats = [f[1] for f in fields]
    a_vals = [metrics_a.get(f[0], np.nan) for f in fields]
    b_vals = [metrics_b.get(f[0], np.nan) for f in fields]

    # Normalize for radar display scale 0..1 per-feature using joint min/max
    all_vals = np.array([a_vals, b_vals], dtype=float)
    mins = np.nanmin(all_vals, axis=0)
    maxs = np.nanmax(all_vals, axis=0)
    denom = np.where((maxs - mins) == 0, 1, (maxs - mins))
    a_norm = (np.array(a_vals) - mins)/denom
    b_norm = (np.array(b_vals) - mins)/denom

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=a_norm, theta=cats, fill='toself', name=label_a))
    fig.add_trace(go.Scatterpolar(r=b_norm, theta=cats, fill='toself', name=label_b))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), height=500,
                      title="Overall Metrics Comparison (normalized)")
    return fig

# --------------------------
# Streamlit UI
# --------------------------
st.title("🏃‍♀️🏃 Motion Comparison Dashboard")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Pose model", ["MediaPipe (recommended)", "OpenPose (not included)"])
    min_det = st.slider("Min detection confidence", 0.1, 0.9, 0.5, 0.05)
    min_track = st.slider("Min tracking confidence", 0.1, 0.9, 0.5, 0.05)
    model_complexity = st.selectbox("Model complexity", [0,1,2], index=1,
                                    help="Higher = more accurate but slower")
    st.markdown("---")
    st.caption("Optional rough calibration (for meters):")
    height_a = st.number_input("Athlete A height (m)", min_value=0.0, max_value=2.5, value=0.0, step=0.01)
    height_b = st.number_input("Athlete B height (m)", min_value=0.0, max_value=2.5, value=0.0, step=0.01)

left, right = st.columns(2)
with left:
    st.subheader("Athlete A Video")
    file_a = st.file_uploader("Upload video A", type=['mp4','mov','avi','mkv'], key="vid_a")
with right:
    st.subheader("Athlete B Video")
    file_b = st.file_uploader("Upload video B", type=['mp4','mov','avi','mkv'], key="vid_b")

process = st.button("▶️ Process & Compare", type="primary", disabled=not(file_a and file_b))

if process:
    if model.startswith("OpenPose"):
        st.error("OpenPose processing isn't bundled in this app. Please switch to MediaPipe or extend the code to integrate OpenPose.")
        st.stop()
    if not MP_AVAILABLE:
        st.error("MediaPipe is not installed. Please install with `pip install mediapipe` and rerun.")
        st.stop()

    # Save uploads to temp files
    tmpdir = tempfile.mkdtemp()
    in_path_a = os.path.join(tmpdir, "input_a.mp4")
    in_path_b = os.path.join(tmpdir, "input_b.mp4")
    with open(in_path_a, "wb") as f:
        f.write(file_a.read())
    with open(in_path_b, "wb") as f:
        f.write(file_b.read())

    out_path_a = os.path.join(tmpdir, "annotated_a.mp4")
    out_path_b = os.path.join(tmpdir, "annotated_b.mp4")

    # Process both videos
    with st.spinner("Processing Athlete A..."):
        df_a = mediapipe_pose_process(in_path_a, out_path_a, min_det=min_det, min_track=min_track, model_complexity=model_complexity)
    with st.spinner("Processing Athlete B..."):
        df_b = mediapipe_pose_process(in_path_b, out_path_b, min_det=min_det, min_track=min_track, model_complexity=model_complexity)

    # Show annotated videos (FIXED)
    c1, c2 = st.columns(2)
    with c1:
        st.header("Annotated Video A")
        with open(out_path_a, "rb") as f:
            video_bytes_a = f.read()
        st.video(video_bytes_a)
    with c2:
        st.header("Annotated Video B")
        with open(out_path_b, "rb") as f:
            video_bytes_b = f.read()
        st.video(video_bytes_b)

    # FPS estimates from time column
    fps_a = 1.0 / np.median(np.diff(df_a['time'])) if len(df_a) > 1 else 30.0
    fps_b = 1.0 / np.median(np.diff(df_b['time'])) if len(df_b) > 1 else 30.0

    # Optional calibration
    pxpm_a = estimate_px_per_meter(df_a, height_a) if height_a and height_a>0 else None
    pxpm_b = estimate_px_per_meter(df_b, height_b) if height_b and height_b>0 else None

    # Summary metrics
    metrics_a = summary_metrics(df_a, fps_a, px_per_meter=pxpm_a)
    metrics_b = summary_metrics(df_b, fps_b, px_per_meter=pxpm_b)

    # Display summary tables
    st.markdown("### 📋 Summary Metrics")
    mdf = pd.DataFrame([metrics_a, metrics_b], index=["Athlete A", "Athlete B"]).T
    st.dataframe(mdf)

    # Time-series comparisons
    st.markdown("### 📈 Time-Series Comparisons")
    ts1 = plot_time_series(df_a, df_b, 'speed_px_s', 'Athlete A', 'Athlete B', 'Speed over time', 'px/s')
    st.plotly_chart(ts1, use_container_width=True)

    angle_cols = [('L_knee','Left Knee (°)'), ('R_knee','Right Knee (°)'),
                  ('L_hip','Left Hip (°)'), ('R_hip','Right Hip (°)'),
                  ('L_elbow','Left Elbow (°)'), ('R_elbow','Right Elbow (°)')]
    for col, label in angle_cols:
        fig = plot_time_series(df_a, df_b, col, 'Athlete A', 'Athlete B', label, 'degrees')
        st.plotly_chart(fig, use_container_width=True)

    # Stride proxies (ankle X)
    st.markdown("### 👟 Stride / Step Proxies (ankle X)")
    fig_L = plot_time_series(df_a, df_b, 'landmark_LEFT_ANKLE_x', 'Athlete A', 'Athlete B', 'Left Ankle X', 'pixels')
    st.plotly_chart(fig_L, use_container_width=True)
    fig_R = plot_time_series(df_a, df_b, 'landmark_RIGHT_ANKLE_x', 'Athlete A', 'Athlete B', 'Right Ankle X', 'pixels')
    st.plotly_chart(fig_R, use_container_width=True)

    # Radar chart
    st.markdown("### 🛡️ Radar (Overall Comparison)")
    radar = plot_radar(metrics_a, metrics_b, "Athlete A", "Athlete B")
    st.plotly_chart(radar, use_container_width=True)

    # Notes
    with st.expander("ℹ️ Notes & Tips"):
        st.write("""
        - All distances and speeds default to *pixels* unless you provide the athlete's height (rough calibration).
        - Stride length and step rate are approximations from ankle horizontal motion. For best results, record from the side view.
        - Jump height is a rough estimate from mid-hip vertical displacement.
        - You can extend this app for OpenPose by adding a similar extractor that returns the same DataFrame schema.
        """)

else:
    st.info("Upload both videos and click **Process & Compare** to begin.")