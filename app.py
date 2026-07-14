# Interactive GUI for this project -- lets someone with no Python experience
# label their own gauge photos/videos, train the model on that data, and run
# the live reader, all from a browser tab instead of editing constants in
# train_model.py/stream_to_core.py by hand.
#
#   streamlit run app.py
#
# Training and streaming are launched as SUBPROCESSES of train_model.py /
# stream_to_core.py (same scripts the README's CLI workflow uses) rather
# than called in-process -- both are long-running and block, and
# stream_to_core.py also pops up its own local preview window, neither of
# which plays well with Streamlit's rerun-on-interaction execution model.
# Running them as subprocesses also means this file never has to duplicate
# their logic, only launch them and tail their log output.
from streamlit_image_coordinates import streamlit_image_coordinates

import os
import re
import subprocess
import sys
import time

import cv2
import numpy as np
import streamlit as st

import dataset_tools
import train_model
from process_image import process_frame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(BASE_DIR, "gui_runtime")
os.makedirs(RUNTIME_DIR, exist_ok=True)

POINT_COLORS = {  # BGR, drawn on the preview so already-clicked points are visible
    "pivot": (0, 255, 255),
    "scale_min": (255, 0, 0),
    "scale_max": (0, 0, 255),
    "needle_tip": (0, 255, 0),
}

VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "m4v"]
IMAGE_TYPES = ["jpg", "jpeg", "png"]


def sanitize_stem(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)


def draw_points(frame, points):
    """Return a copy of frame with a colored dot + label at each already-
    clicked keypoint, so the user gets visual feedback on what they've set
    before moving on to the next point."""
    out = frame.copy()
    for name, (x, y) in points.items():
        color = POINT_COLORS.get(name, (255, 255, 255))
        cv2.circle(out, (int(x), int(y)), 6, color, -1)
        cv2.putText(out, name, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def start_subprocess(cmd, log_path):
    """Launch cmd with stdout+stderr tee'd to log_path, using THIS process's
    own Python interpreter (sys.executable) so subprocesses run inside the
    same virtualenv the GUI itself was started from."""
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=BASE_DIR, text=True)
    log_file.close()  # Popen holds its own dup'd fd; our handle isn't needed after this
    return proc


def read_log(log_path, max_chars=15000):
    if not log_path or not os.path.exists(log_path):
        return ""
    with open(log_path) as f:
        text = f.read()
    return text[-max_chars:]


def process_status(proc):
    if proc is None:
        return None
    return "running" if proc.poll() is None else f"exited ({proc.poll()})"


def init_state():
    defaults = {
        "label_frames": [],       # list of (stem, processed_bgr_frame)
        "label_idx": 0,
        "label_points": {},
        "label_saved_count": 0,
        "train_proc": None,
        "train_log_path": None,
        "stream_proc": None,
        "stream_log_path": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_label_tab():
    st.header("1. Label Data")
    st.write(
        "Add your own gauge photos or a video, click the 4 keypoints on each "
        "frame (**pivot**, **scale_min**, **scale_max**, **needle_tip**), and "
        "they're saved straight into the training set below."
    )

    stats = dataset_tools.dataset_stats()
    c1, c2 = st.columns(2)
    c1.metric("Train images", stats["Train"])
    c2.metric("Val images", stats["Val"])

    if not st.session_state.label_frames:
        source_kind = st.radio("Add from", ["Video", "Photos"], horizontal=True, key="label_source_kind")

        if source_kind == "Video":
            uploaded = st.file_uploader("Upload a gauge video", type=VIDEO_TYPES, key="label_video_upload")
            interval = st.slider("Label every Nth frame", 1, 90, 15, key="label_video_interval")
            if uploaded and st.button("Extract frames"):
                video_path = os.path.join(RUNTIME_DIR, f"label_{uploaded.name}")
                with open(video_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                stem_base = sanitize_stem(os.path.splitext(uploaded.name)[0])
                frames = [
                    (f"{stem_base}_frame_{idx:04d}", frame)
                    for idx, frame in dataset_tools.extract_frames(video_path, frame_interval=interval)
                ]
                if not frames:
                    st.warning("No frames extracted -- try a smaller interval or a different video.")
                st.session_state.label_frames = frames
                st.session_state.label_idx = 0
                st.session_state.label_points = {}
                st.rerun()
        else:
            uploaded_images = st.file_uploader(
                "Upload gauge photos", type=IMAGE_TYPES, accept_multiple_files=True, key="label_image_upload"
            )
            if uploaded_images and st.button("Load photos"):
                frames = []
                for uf in uploaded_images:
                    data = np.frombuffer(uf.getbuffer(), dtype=np.uint8)
                    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                    stem = sanitize_stem(os.path.splitext(uf.name)[0])
                    frames.append((stem, process_frame(bgr)))
                st.session_state.label_frames = frames
                st.session_state.label_idx = 0
                st.session_state.label_points = {}
                st.rerun()
        return

    frames = st.session_state.label_frames
    idx = st.session_state.label_idx

    if idx >= len(frames):
        st.success(f"Done -- labeled {st.session_state.label_saved_count} image(s) this session.")
        if st.button("Label more"):
            st.session_state.label_frames = []
            st.session_state.label_idx = 0
            st.session_state.label_points = {}
            st.rerun()
        return

    stem, frame = frames[idx]
    points = st.session_state.label_points
    st.progress(idx / len(frames))
    st.write(f"Image {idx + 1} of {len(frames)}: `{stem}`")

    remaining = [n for n in dataset_tools.KEYPOINT_ORDER if n not in points]
    if remaining:
        st.info(f"Click **{remaining[0]}** ({len(points) + 1}/4)")
    else:
        st.success("All 4 points set.")

    display_rgb = cv2.cvtColor(draw_points(frame, points), cv2.COLOR_BGR2RGB)
    coords = streamlit_image_coordinates(display_rgb, key=f"label_click_{idx}_{len(points)}")
    if coords and remaining:
        points[remaining[0]] = (coords["x"], coords["y"])
        st.rerun()

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Undo last point", disabled=not points):
        del points[list(points)[-1]]
        st.rerun()
    if b2.button("Reset points", disabled=not points):
        st.session_state.label_points = {}
        st.rerun()
    if b3.button("Skip image (no gauge)"):
        st.session_state.label_idx += 1
        st.session_state.label_points = {}
        st.rerun()
    if b4.button("Save && next", type="primary", disabled=not points):
        dataset_tools.ensure_data_yaml()
        dataset_tools.save_labeled_image(frame, points, stem)
        st.session_state.label_saved_count += 1
        st.session_state.label_idx += 1
        st.session_state.label_points = {}
        st.rerun()


def render_train_tab():
    st.header("2. Train Model")

    stats = dataset_tools.dataset_stats()
    st.write(f"Dataset: **{stats['Train']}** train / **{stats['Val']}** val images")
    if stats["Train"] == 0:
        st.warning("No labeled images yet -- use the **Label Data** tab first.")

    existing = train_model.find_latest_best_weights()
    if existing:
        st.caption(f"Existing checkpoint: `{existing}`")
        mode = st.radio("Mode", ["Fine-tune from latest checkpoint", "Train from scratch"], key="train_mode")
    else:
        st.caption("No trained checkpoint yet -- will train from scratch.")
        mode = "Train from scratch"
    is_finetune = mode.startswith("Fine")

    col1, col2, col3 = st.columns(3)
    epochs = col1.number_input("Epochs", 1, 500, 30 if is_finetune else 100, key="train_epochs")
    patience = col2.number_input("Patience", 1, 100, 10 if is_finetune else 20, key="train_patience")
    imgsz = col3.selectbox("Image size", [320, 416, 640], index=1, key="train_imgsz")

    running = st.session_state.train_proc is not None and st.session_state.train_proc.poll() is None
    if st.button("Start training", disabled=(stats["Train"] == 0 or running), type="primary"):
        dataset_tools.ensure_data_yaml()
        log_path = os.path.join(RUNTIME_DIR, "train.log")
        cmd = [
            sys.executable, os.path.join(BASE_DIR, "train_model.py"),
            "--mode", "finetune" if is_finetune else "scratch",
            "--epochs", str(epochs), "--patience", str(patience), "--imgsz", str(imgsz),
        ]
        st.session_state.train_proc = start_subprocess(cmd, log_path)
        st.session_state.train_log_path = log_path
        st.rerun()

    if st.session_state.train_proc is not None:
        proc = st.session_state.train_proc
        status = process_status(proc)
        st.write(f"Status: **{status}**")
        st.code(read_log(st.session_state.train_log_path) or "(no output yet)", language="text")

        cols = st.columns(2)
        if cols[0].button("Stop training", disabled=status != "running"):
            proc.terminate()
            st.rerun()
        auto_refresh = cols[1].checkbox("Auto-refresh", value=True, key="train_auto_refresh")
        if status == "running" and auto_refresh:
            time.sleep(2)
            st.rerun()


def render_run_tab():
    st.header("3. Run && Stream to Nominal")

    checkpoint = train_model.find_latest_best_weights()
    if not checkpoint:
        st.warning("No trained model yet -- label data and train one first.")
        return
    st.caption(f"Using checkpoint: `{checkpoint}`")

    source_kind = st.radio("Source", ["Upload video", "Webcam", "Path on disk"], horizontal=True, key="run_source_kind")
    source_arg = None
    if source_kind == "Upload video":
        uploaded = st.file_uploader("Video to read", type=VIDEO_TYPES, key="run_video_upload")
        if uploaded:
            source_arg = os.path.join(RUNTIME_DIR, f"stream_{uploaded.name}")
            with open(source_arg, "wb") as f:
                f.write(uploaded.getbuffer())
    elif source_kind == "Webcam":
        source_arg = str(st.number_input("Webcam device index", 0, 10, 0, key="run_webcam_index"))
    else:
        source_arg = st.text_input("Video file or image-directory path", key="run_source_path") or None

    col1, col2 = st.columns(2)
    min_value = col1.number_input("Gauge min value", value=0.0, key="run_min_value")
    max_value = col2.number_input("Gauge max value", value=2.5, key="run_max_value")

    running = st.session_state.stream_proc is not None and st.session_state.stream_proc.poll() is None
    if st.button("Start streaming", disabled=(not source_arg or running), type="primary"):
        log_path = os.path.join(RUNTIME_DIR, "stream.log")
        cmd = [
            sys.executable, os.path.join(BASE_DIR, "stream_to_core.py"),
            "--source", str(source_arg), "--min", str(min_value), "--max", str(max_value),
        ]
        st.session_state.stream_proc = start_subprocess(cmd, log_path)
        st.session_state.stream_log_path = log_path
        st.rerun()

    if st.session_state.stream_proc is not None:
        proc = st.session_state.stream_proc
        status = process_status(proc)
        st.write(f"Status: **{status}**")
        st.caption(
            "A local preview window (\"Gauge Reading -- Original | Processed\") opens "
            "outside the browser. Press 'q' in that window, or click Stop below, to end the stream."
        )
        st.code(read_log(st.session_state.stream_log_path) or "(no output yet)", language="text")

        cols = st.columns(2)
        if cols[0].button("Stop streaming", disabled=status != "running"):
            proc.terminate()
            st.rerun()
        auto_refresh = cols[1].checkbox("Auto-refresh", value=True, key="run_auto_refresh")
        if status == "running" and auto_refresh:
            time.sleep(2)
            st.rerun()


def main():
    st.set_page_config(page_title="Gauge Reader", layout="wide")
    init_state()
    dataset_tools.ensure_data_yaml()

    st.title("Gauge Reader")
    st.caption("Label your own gauge data, train the model, and run it -- all from the browser.")

    tab1, tab2, tab3 = st.tabs(["1. Label Data", "2. Train Model", "3. Run & Stream"])
    with tab1:
        render_label_tab()
    with tab2:
        render_train_tab()
    with tab3:
        render_run_tab()


main()
