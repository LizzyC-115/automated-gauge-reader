# Shared helpers behind the labeling GUI (app.py): turning a raw video/photo
# into a labeled YOLO-pose training sample, and keeping the dataset's
# data.yaml / Train-Val split bookkeeping consistent as samples are added.
#
# Kept separate from app.py so this logic has no Streamlit dependency -- it
# could be reused from a script or tested on its own.
import glob
import os

import cv2
import yaml

from process_image import process_frame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "gauge-readings-processed")

# Order matters: this is the fixed click sequence the labeling UI walks
# through for every frame, and also defines the class id each name maps to
# (pivot=0, scale_min=1, scale_max=2, needle_tip=3) -- must match
# get_measurement.REQUIRED_POINTS' names, though not necessarily their order.
KEYPOINT_ORDER = ("pivot", "scale_min", "scale_max", "needle_tip")

# Width/height of the synthetic box drawn around each clicked point, as a
# fraction of image size. YOLO-pose still trains a box per keypoint even
# though we only care about the point itself; existing Roboflow-labeled
# samples in this dataset use boxes in the 0.045-0.08 range, so this matches
# that scale rather than inventing an unrelated convention.
BOX_FRACTION = 0.06


def class_name_to_id():
    return {name: i for i, name in enumerate(KEYPOINT_ORDER)}


def ensure_data_yaml(dataset_dir=DATASET_DIR):
    """(Re)write data.yaml with an absolute `path` pointing at dataset_dir on
    THIS machine. The dataset gets cloned/copied to different paths by
    whoever runs this GUI, so the yaml has to be regenerated locally instead
    of trusting whatever `path` a previous contributor committed.
    """
    for split in ("Train", "Val"):
        os.makedirs(os.path.join(dataset_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "labels", split), exist_ok=True)

    data = {
        "path": dataset_dir,
        "train": "images/Train",
        "val": "images/Val",
        "kpt_shape": [1, 3],
        "names": {i: name for i, name in enumerate(KEYPOINT_ORDER)},
    }
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return yaml_path


def dataset_stats(dataset_dir=DATASET_DIR):
    """Return {"Train": n, "Val": n} labeled-image counts for the dataset."""
    stats = {}
    for split in ("Train", "Val"):
        images_dir = os.path.join(dataset_dir, "images", split)
        stats[split] = len(glob.glob(os.path.join(images_dir, "*.jpg"))) if os.path.isdir(images_dir) else 0
    return stats


def choose_split(dataset_dir=DATASET_DIR, val_fraction=0.15):
    """Pick "Train" or "Val" for the NEXT labeled image, keeping the running
    split ratio close to val_fraction -- rather than e.g. always appending to
    Train, which would let Val stagnate as the dataset grows.
    """
    stats = dataset_stats(dataset_dir)
    total = stats["Train"] + stats["Val"]
    if total == 0:
        return "Train"
    return "Val" if stats["Val"] / total < val_fraction else "Train"


def extract_frames(video_path, frame_interval=15):
    """Yield (index, processed_bgr_frame) for every frame_interval-th frame
    of video_path, already run through process_frame() -- the SAME
    grayscale/contrast/edge transform train_model.py's dataset and
    stream_to_core.py's live inference apply, so keypoints clicked here land
    on the same pixels the model will actually be trained/run on.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if index % frame_interval == 0:
            yield index, process_frame(frame)
        index += 1
    cap.release()


def save_labeled_image(processed_bgr, points, stem, dataset_dir=DATASET_DIR, split=None):
    """Save one labeled sample: processed_bgr into images/{split}/{stem}.jpg
    and one YOLO-pose label line per clicked keypoint into
    labels/{split}/{stem}.txt.

    points is {"pivot": (x, y), ...} in PIXEL coordinates of processed_bgr,
    using whichever subset of KEYPOINT_ORDER the user actually clicked --
    a class with no click is simply omitted from the label file, same as a
    class YOLO never detects at inference time.

    Returns (image_path, label_path, split) -- split is auto-chosen via
    choose_split() unless the caller passes one explicitly.
    """
    if not points:
        raise ValueError("save_labeled_image() needs at least one clicked keypoint")

    split = split or choose_split(dataset_dir)
    h, w = processed_bgr.shape[:2]
    ids = class_name_to_id()

    lines = []
    for name, (x, y) in points.items():
        cx, cy = x / w, y / h
        lines.append(f"{ids[name]} {cx:.6f} {cy:.6f} {BOX_FRACTION:.6f} {BOX_FRACTION:.6f} {cx:.6f} {cy:.6f} 2")

    images_dir = os.path.join(dataset_dir, "images", split)
    labels_dir = os.path.join(dataset_dir, "labels", split)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    image_path = os.path.join(images_dir, f"{stem}.jpg")
    label_path = os.path.join(labels_dir, f"{stem}.txt")
    cv2.imwrite(image_path, processed_bgr)
    with open(label_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return image_path, label_path, split
