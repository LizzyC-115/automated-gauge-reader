# Gauge Reader

Reads analog gauge measurements (pressure, temperature, etc.) from video using
a fine-tuned YOLO pose model, and streams both the live annotated video and
the computed numeric reading to Nominal Core.

## How it works

A YOLO pose model is trained to find 4 keypoints on a gauge face: `pivot`
(the needle's center), `scale_min` and `scale_max` (the two ends of the
dial), and `needle_tip`. From those 4 points, simple trigonometry converts
the needle's angle into a numeric reading interpolated between the gauge's
min/max values.

Frames are grayscale/contrast-enhanced and reduced to a gauge-circle +
scale-marking trace (via classical CV: CLAHE contrast, Canny edge detection,
Hough Circle Transform for the bezel) before the model ever sees them --
this was the key change that got the model generalizing across different
gauge types instead of overfitting to the exact photos it was trained on.
See the Notion doc linked from this repo for the full story of how that
decision came about.

## Files

| File | Purpose |
|---|---|
| `stream_to_core.py` | **Entry point.** Trains a model if none exists, then reads gauge measurements from a video/webcam feed and streams the video + reading to Nominal Core. Run with `python3 stream_to_core.py`. |
| `train_model.py` | Trains/fine-tunes the YOLO pose model. Runs standalone: `python3 train_model.py`. |
| `process_image.py` | Per-frame preprocessing: grayscale, contrast enhancement, gauge-circle detection (Hough Circle Transform), scale-marking trace (Canny + dilation). |
| `get_measurement.py` | Pure geometry: converts 4 detected keypoints into a numeric reading. |
| `read_measurement.py` | Shared helpers (keypoint extraction, image/video path handling) used by both training data prep and the streaming pipeline. |
| `extract_photos.py` | Extracts frames from source videos for annotation/training. |
| `merge_roboflow_dataset.py` | Merges an externally-annotated (Roboflow-exported) dataset into the training set, remapping class IDs and label format as needed. |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics opencv-python "nominal[video]"
```

Nominal streaming also requires GStreamer and a configured profile:

```bash
brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly libnice-gstreamer
nom config profile add default -t <your-api-token> -u <your-api-url>
```

## Running

```bash
python3 stream_to_core.py       # train (if needed) + read + stream to Nominal
python3 train_model.py          # train/fine-tune only
```

## Data

Training data (`gauge-readings-processed/`, `processed_images/`, source
videos, etc.) is intentionally excluded from this repo via `.gitignore` --
it's large binary/image data, not code. Keep it on disk locally.
