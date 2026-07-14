# Highest-level entry point for this project.
#
#   python3 stream_to_core.py
#
# Trains a YOLO pose model if one hasn't been trained yet (see
# train_model.py), then uses it to read gauge measurements from a video/
# webcam feed and streams both the live video and the numeric reading to
# Nominal Core.
#
# One-time setup required before streaming works:
#   pip install "nominal[video]"           (video streaming needs the extra)
#   brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly libnice-gstreamer
#   nom config profile add default -t <your-api-token> -u <your-api-url>
# using the API key/URL from Settings > API keys in the Nominal app. See
# https://docs.nominal.io/core/sdk/python-client/authentication and
# https://docs.nominal.io/core/sdk/python-client/video/live-video-streaming
from ultralytics import YOLO
from get_measurement import get_reading
from process_image import process_frame
from train_model import train_model, find_latest_best_weights
from read_measurement import extract_keypoints, iter_image_paths, probe_frame_size
from nominal.core import NominalClient
from nominal.experimental.video import VideoStream, Src

import argparse
import cv2
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "gauge-readings-processed", "images", "Val")
INFERENCE_OUTPUT_DIR = os.path.join(BASE_DIR, "inference_output")
DEFAULT_STREAM_SOURCE = "/Users/lchanpaibool/.cache/kagglehub/datasets/juliusgrassme/pressure-gauge-reader-data/versions/1/Data/2 Test videos/edited videos/man2cropclipscale.mp4"

# Quick-switch videos, selectable from the command line with --1/--3/--4
# (e.g. `python3 stream_to_core.py --3`) instead of editing
# DEFAULT_STREAM_SOURCE by hand every time you want to try a different clip.
# These (and DEFAULT_STREAM_SOURCE above) are convenience shortcuts to this
# project's own demo videos -- to point at YOUR OWN video/webcam/min-max
# instead, use --source/--min/--max (see parse_args() below), which is what
# the GUI's "Run & Stream" tab does under the hood.
MAN_VIDEOS_DIR = "/Users/lchanpaibool/.cache/kagglehub/datasets/juliusgrassme/pressure-gauge-reader-data/versions/1/Data/2 Test videos/edited videos"
DEFAULT_MIN_VALUE = 0.0
DEFAULT_MAX_VALUE = 2.5

# Each entry is (video path, min_value, max_value) -- every physical gauge
# in these clips has its own scale, so the reading range has to travel with
# the video instead of being one hardcoded pair for all of them.
VIDEO_CHOICES = {
    "1": (os.path.join(MAN_VIDEOS_DIR, "man1cropclipscale.mp4"), 0.0, 6.0),
    "3": (os.path.join(MAN_VIDEOS_DIR, "man3cropclipscale.mp4"), 0.0, 2.5),
    "4": (
        "/Users/lchanpaibool/.cache/kagglehub/datasets/juliusgrassme/pressure-gauge-reader-data/versions/1/Data/3 Misc handheld videos/handheld4.mp4",
        -1.0,
        3.0,
    ),
}


def parse_args():
    """Parse either one of --1/--3/--4 (this project's own demo clips), or a
    generic --source/--min/--max for streaming YOUR OWN video file, image
    directory, or webcam. At most one of --1/--3/--4 may be given; --source
    is independent of that group and takes priority if both are somehow
    passed. With nothing given at all, resolve_stream_config() falls back to
    DEFAULT_STREAM_SOURCE/DEFAULT_MIN_VALUE/DEFAULT_MAX_VALUE.
    """
    parser = argparse.ArgumentParser(description="Train (if needed), read gauge measurements from a video, and stream to Nominal Core.")
    group = parser.add_mutually_exclusive_group()
    for key, (path, _min_value, _max_value) in VIDEO_CHOICES.items():
        group.add_argument(f"--{key}", action="store_true", help=f"Stream {os.path.basename(path)}")
    parser.add_argument("--source", default=None,
                         help="Video file, image directory, or webcam index (e.g. 0) to stream from.")
    parser.add_argument("--min", dest="min_value", type=float, default=None, help="Gauge value at scale_min.")
    parser.add_argument("--max", dest="max_value", type=float, default=None, help="Gauge value at scale_max.")
    return parser.parse_args()


def resolve_stream_config(args):
    """Return (source, min_value, max_value): --source (with --min/--max)
    wins if given; otherwise whichever VIDEO_CHOICES entry matches --1/--3/
    --4; otherwise the DEFAULT_* values.

    A --source that's purely digits (e.g. "0") is treated as a webcam device
    index rather than a file path -- cv2.VideoCapture accepts either an int
    index or a string path, but argparse always hands us a string.
    """
    if args.source is not None:
        source = int(args.source) if args.source.isdigit() else args.source
        min_value = args.min_value if args.min_value is not None else DEFAULT_MIN_VALUE
        max_value = args.max_value if args.max_value is not None else DEFAULT_MAX_VALUE
        return source, min_value, max_value

    for key, (path, min_value, max_value) in VIDEO_CHOICES.items():
        if getattr(args, key):
            return path, min_value, max_value
    return DEFAULT_STREAM_SOURCE, DEFAULT_MIN_VALUE, DEFAULT_MAX_VALUE

NOMINAL_PROFILE = "default"
NOMINAL_ASSET_RID = "ri.scout.cerulean-staging.asset.1abf9d05-57f4-47a3-b4f9-0945594d3ede"
NOMINAL_DATASET_RID = "ri.catalog.cerulean-staging.dataset.dcdc95db-858f-4c38-a66c-542defcfff41"
NOMINAL_CHANNEL_NAME = "gauge_reading_bar"
NOMINAL_VIDEO_NAME = "Gauge Reading Live Video"

# Refnames ("data scope" names) the dataset and video get attached to the
# asset under. Once attached, later runs find the SAME dataset/video through
# the asset instead of creating a fresh resource every time.
NOMINAL_DATA_SCOPE_NAME = "gauge_reading_data"
NOMINAL_VIDEO_SCOPE_NAME = "gauge_reading_video"


def get_nominal_client():
    """Connect to Nominal using the locally configured profile."""
    return NominalClient.from_profile(NOMINAL_PROFILE)


def get_nominal_asset():
    """Fetch the target asset by RID. Both the reading dataset and the live
    video get attached here so they show up together in the asset's
    workbooks instead of floating around as unrelated resources."""
    return get_nominal_client().get_asset(NOMINAL_ASSET_RID)


def get_or_attach_dataset(asset):
    """Reuse the dataset already attached to this asset under
    NOMINAL_DATA_SCOPE_NAME if one exists; otherwise fetch our fixed
    NOMINAL_DATASET_RID and attach it under that name so future runs find it
    the same way."""
    try:
        return asset.get_dataset(NOMINAL_DATA_SCOPE_NAME)
    except Exception:
        dataset = get_nominal_client().get_dataset(NOMINAL_DATASET_RID)
        asset.add_dataset(dataset=dataset, data_scope_name=NOMINAL_DATA_SCOPE_NAME)
        return dataset


def get_or_create_video(asset):
    """Reuse the video already attached to this asset under
    NOMINAL_VIDEO_SCOPE_NAME if one exists -- so repeated runs stream into
    the SAME video resource instead of creating a new one every time.
    Otherwise create a new video and attach it under that name."""
    try:
        return asset.get_video(NOMINAL_VIDEO_SCOPE_NAME)
    except Exception:
        video = get_nominal_client().create_video(NOMINAL_VIDEO_NAME)
        asset.add_video(NOMINAL_VIDEO_SCOPE_NAME, video)
        return video


def load_or_train_model():
    """The 'train it if it doesn't exist yet' step -- reuses the most
    recent checkpoint if one exists, otherwise trains a new one from
    scratch via train_model.py before returning a model ready for
    inference. This is what makes stream_to_core.py runnable standalone
    on a totally fresh checkout: no model required up front."""
    existing = find_latest_best_weights()
    if existing:
        print(f"Loading existing checkpoint: {existing}")
        return YOLO(existing)
    print("No trained checkpoint found -- training a new one.")
    return train_model()


def run_inference(model, source, output_dir=INFERENCE_OUTPUT_DIR):
    """Run detection on an image/video source (a single file OR a directory
    of images -- YOLO's predict() natively globs every image in a directory)
    and save each frame with predicted boxes drawn on it. stream=True returns
    a generator -- it does nothing until you iterate it.

    Returns a list of dicts, one per image: {"image": stem, "points": {...}}
    where "points" is whatever extract_keypoints() found -- this is the data
    get_angle() (and eventually a full get_reading()) will consume.

    Every valid reading (None is skipped -- Nominal channels are numeric)
    also gets streamed live to the dataset attached to NOMINAL_ASSET_RID
    under NOMINAL_CHANNEL_NAME, timestamped at the moment each image was
    processed.

    Each image is grayscale/contrast-enhanced via process_frame() before the
    model ever sees it -- this replaces the old "let YOLO glob the directory
    itself" behavior, since preprocessing means we need to load and touch
    every frame ourselves first.
    """
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    with_detections = 0
    results = []
    asset = get_nominal_asset()
    dataset = get_or_attach_dataset(asset)

    with dataset.get_write_stream() as stream:
        for i, image_path in enumerate(iter_image_paths(source)):
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            processed_frame = process_frame(frame)
            r = model.predict(source=processed_frame, verbose=False)[0]
            annotated = r.plot()  # BGR numpy array with boxes/labels drawn

            # Reuse the original filename (e.g. frame_0025_jpg.rf....jpg) so
            # results are easy to match back to their source image.
            stem = os.path.splitext(os.path.basename(image_path))[0]
            filename = os.path.join(output_dir, f"{stem}_pred.jpg")

            points = extract_keypoints(r)
            reading = get_reading(points, min_value=0.0, max_value=6.0)
            results.append({"image": stem, "points": points, "reading": reading})

            if reading is not None:
                stream.enqueue(channel_name=NOMINAL_CHANNEL_NAME, timestamp=datetime.now(), value=reading)

            # Burn the computed reading into the saved image too, so you can
            # visually sanity-check it against the actual dial in the same file.
            reading_text = f"Reading: {reading:.2f}" if reading is not None else "Reading: --"
            cv2.putText(annotated, reading_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imwrite(filename, annotated)

            num_boxes = len(r.boxes)
            total += 1
            with_detections += 1 if num_boxes else 0
            print(f"Saved {filename} -- {num_boxes} detection(s) -- {reading_text}")
            for name, (x, y, conf) in points.items():
                print(f"    {name:12s} conf={conf:.2f}  xy=({x:.1f}, {y:.1f})")

    print(f"\nDone: {with_detections}/{total} images had at least one detection.")
    return results


def stream_gauge_video(model, source, min_value=0.0, max_value=4.0, frame_skip=15):
    """Shared implementation behind run_video() and live_stream() -- both
    used to carry an identical copy of this loop (one defaulting to a video
    file, one to a webcam), so any tweak had to be made twice. Now there's
    one body and the two public functions just call it with their own
    default `source`.

    Runs live detection on a video/webcam source, overlaying the computed
    reading in real time. Press 'q' in the window to stop early.

    Note: this does its own cv2.imshow instead of predict(show=True) --
    ultralytics' built-in display doesn't know about our reading overlay, so
    we draw the frame ourselves after adding the reading text.

    Streams two things to Nominal in parallel, sharing the same wall-clock
    timestamp per frame so they line up in a workbook -- both attached to
    NOMINAL_ASSET_RID so they show up together on the same asset:
      - the plain color video (just the reading text burned in, no
        boxes/keypoints) as a live video under NOMINAL_VIDEO_SCOPE_NAME --
        reused across runs instead of creating a new video resource every
        time.
      - every valid reading (None/"--" is skipped -- Nominal channels are
        numeric) to the dataset under NOMINAL_DATA_SCOPE_NAME, channel
        NOMINAL_CHANNEL_NAME.

    Reads frames ourselves via cv2.VideoCapture (instead of handing the raw
    source to model.predict) so each frame can be grayscale/contrast
    enhanced with process_frame() before the model sees it.

    model.predict() only runs every `frame_skip`-th frame -- inference is
    the real per-frame cost on CPU, so skipping it on most frames cuts total
    inference work substantially. Every frame still gets shown/streamed at
    full rate for smooth video; the reading text just holds over its last
    computed value on the frames where the model didn't run, instead of
    blanking to "--" in between.

    Locally (not sent to Nominal) this also shows the original color frame
    side by side with the processed B&W frame + detected boxes/keypoints,
    so you can see both what the camera saw and what the model saw/found.
    """
    asset = get_nominal_asset()
    dataset = get_or_attach_dataset(asset)
    video = get_or_create_video(asset)

    width, height = probe_frame_size(source)
    print(f"Streaming to asset {NOMINAL_ASSET_RID} -- video RID: {video.rid}")
    video_src = Src.app(width=width, height=height, format="RGB", auto_timestamp=False)

    cap = cv2.VideoCapture(source)
    frame_count = 0
    reading = None
    processed_display = None
    with dataset.get_write_stream() as data_stream, VideoStream.create(video, video_src) as video_stream:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                processed_frame = process_frame(frame)
                r = model.predict(source=processed_frame, conf=0.05, verbose=False)[0]
                points = extract_keypoints(r)
                reading = get_reading(points, min_value=min_value, max_value=max_value)
                # Boxes/keypoints drawn on the processed (B&W) frame the
                # model actually ran on -- this is for the local side-by-side
                # display ONLY. It never gets sent to Nominal below, which
                # still only ever receives the plain color frame + reading.
                processed_display = r.plot(img=processed_frame)
            frame_count += 1

            now = datetime.now()
            reading_text = f"Reading: {reading:.2f}" if reading is not None else "Reading: --"
            cv2.putText(frame, reading_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # --- Nominal stream: unchanged -- plain color frame + reading ---
            # Nominal's video source expects RGB bytes; OpenCV frames are BGR.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_stream.send_frame(rgb_frame.tobytes(), timestamp_ns=int(now.timestamp() * 1e9))

            if reading is not None:
                data_stream.enqueue(channel_name=NOMINAL_CHANNEL_NAME, timestamp=now, value=reading)

            # --- Local display only: original | processed+boxes, side by side ---
            right = processed_display if processed_display is not None else frame
            if right.shape[:2] != frame.shape[:2]:
                right = cv2.resize(right, (frame.shape[1], frame.shape[0]))
            combined = cv2.hconcat([frame, right])
            cv2.imshow("Gauge Reading -- Original | Processed", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    source, min_value, max_value = resolve_stream_config(args)
    print(f"Streaming source: {source} (range {min_value}-{max_value})")

    model = load_or_train_model()
    stream_gauge_video(model, source=source, min_value=min_value, max_value=max_value, frame_skip=1)


if __name__ == "__main__":
    main()
