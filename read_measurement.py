# Reusable, Nominal-independent helpers for reading gauge measurements from
# images/video using a trained YOLO pose model.
#
# stream_to_core.py is the top-level entry point for this project -- it
# trains a model if needed, then uses these helpers (plus process_image.py
# and get_measurement.py) to actually read gauges and push results to
# Nominal. Keeping this file free of Nominal/training imports means it can
# be reused (or unit tested) without needing any of that wired up.
import cv2
import glob
import os


def iter_image_paths(source):
    """Yield individual image file paths from `source`, which may be a
    single image file or a directory of images.

    YOLO's predict() can normally glob a whole directory itself, but we need
    each path one at a time so we can load + preprocess (grayscale/contrast)
    every frame ourselves before the model ever sees it.
    """
    if os.path.isdir(source):
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            yield from sorted(glob.glob(os.path.join(source, ext)))
    else:
        yield source


def probe_frame_size(source):
    """Peek at a video/webcam source's frame dimensions using OpenCV before
    YOLO touches it. Src.app() needs a fixed width/height up front to set up
    the Nominal video stream, before the first annotated frame exists."""
    cap = cv2.VideoCapture(source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def extract_keypoints(result):
    """Pull the (x, y, confidence) of each detected landmark out of a single
    prediction result, keyed by class name.

    Each of our 4 classes (pivot, scale_min, scale_max, needle_tip) is its
    own single-point "object" -- kpt_shape=[1, 3] means every detected box
    has exactly one keypoint attached. This collects the highest-confidence
    detection of each class into a dict like:
        {"pivot": (x, y, conf), "needle_tip": (x, y, conf), ...}
    A class with no confident detection in this image just won't be a key --
    always check with e.g. "pivot" in points before indexing into it, since
    get_angle() needs all 4 to produce a real reading.
    """
    points = {}
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return points

    classes = result.boxes.cls.tolist()
    confs = result.boxes.conf.tolist()
    xy = result.keypoints.xy  # tensor shaped (num_boxes, 1, 2) -- pixel coords

    for i, cls_id in enumerate(classes):
        name = result.names[int(cls_id)]
        x, y = xy[i][0].tolist()
        conf = confs[i]
        # If a class fires more than once in one image, keep the best guess.
        if name not in points or conf > points[name][2]:
            points[name] = (x, y, conf)

    return points
