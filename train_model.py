# Train/fine-tune the YOLO pose model used by read_measurement.py.
#
# Kept separate from read_measurement.py so you can kick off a training run
# (`python3 train_model.py`) without needing anything related to inference,
# Nominal streaming, or webcams to be wired up -- and vice versa, reading
# measurements doesn't need this file's dependencies loaded either.
from ultralytics import YOLO

import argparse
import glob
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "gauge-readings-processed", "data.yaml")
RUNS_DIR = os.path.join(BASE_DIR, "runs", "pose")


def train_model(data_yaml=DATA_YAML, epochs=100, patience=20, imgsz=416):
    """Fine-tune YOLO on the gauge-readings dataset. Returns the trained model
    loaded from its best checkpoint (not just the final epoch).

    Uses the -pose checkpoint, not the plain detection one -- data.yaml
    defines kpt_shape (pivot/scale_min/scale_max/needle_tip keypoints), and
    that's only honored by the pose task. Loading "yolo26n.pt" instead runs
    the detect trainer, which tries to parse the extra keypoint columns as
    segmentation polygons and crashes.

    epochs is a ceiling, not a target -- patience stops training early once
    val performance hasn't improved in that many straight epochs. imgsz=416
    (down from the 640 default) is the main speed lever on CPU -- compute
    scales roughly with imgsz^2, so this cuts per-epoch time by more than
    half with a small accuracy tradeoff, which is a fine trade for tiny,
    simple point targets like these.
    """
    model = YOLO(os.path.join(BASE_DIR, "yolo26n-pose.pt"))
    train_results = model.train(data=data_yaml, epochs=epochs, patience=patience, imgsz=imgsz)

    # train_results.save_dir points at runs/pose/trainX -- load the best
    # checkpoint from that run rather than continuing to use `model`, which
    # only holds the final-epoch weights in memory.
    best_weights = train_results.save_dir / "weights" / "best.pt"
    print(f"Best weights saved at: {best_weights}")
    return YOLO(best_weights)


def finetune_model(data_yaml=DATA_YAML, extra_epochs=30, patience=10, imgsz=416):
    """Continue training from the CURRENT best checkpoint instead of
    starting over from the generic pretrained yolo26n-pose.pt.

    Use this after adding new images+labels into gauge-readings/images/Train
    (and a few into images/Val) alongside the existing ones. Because it
    starts from weights that already know this task, it needs far fewer
    epochs to adapt to the new data than train_model()'s from-scratch run.

    Important: data.yaml should point at your FULL train/val set (old + new
    combined), not just the new images alone -- fine-tuning on new data only
    risks the model forgetting what it already learned from the old data.
    """
    existing = find_latest_best_weights()
    if not existing:
        raise RuntimeError("No existing checkpoint found -- run train_model() first.")

    print(f"Continuing fine-tuning from: {existing}")
    model = YOLO(existing)
    train_results = model.train(data=data_yaml, epochs=extra_epochs, patience=patience, imgsz=imgsz)

    best_weights = train_results.save_dir / "weights" / "best.pt"
    print(f"Best weights saved at: {best_weights}")
    return YOLO(best_weights)


def find_latest_best_weights():
    """Return the path to the best.pt from the most recently modified
    runs/pose/train* directory, or None if no trained run exists yet."""
    candidates = glob.glob(os.path.join(RUNS_DIR, "train*", "weights", "best.pt"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def parse_args():
    parser = argparse.ArgumentParser(description="Train/fine-tune the gauge-reading YOLO pose model.")
    parser.add_argument("--data", default=DATA_YAML, help="Path to data.yaml (default: %(default)s)")
    parser.add_argument("--mode", choices=["auto", "scratch", "finetune"], default="auto",
                         help="'auto' (default) fine-tunes from the latest checkpoint if one exists, "
                              "otherwise trains from scratch. 'scratch'/'finetune' force one or the other.")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch ceiling (default: 100 scratch / 30 finetune)")
    parser.add_argument("--patience", type=int, default=None, help="Early-stop patience (default: 20 scratch / 10 finetune)")
    parser.add_argument("--imgsz", type=int, default=416, help="Training image size (default: %(default)s)")
    return parser.parse_args()


def main():
    """Standalone entry point: `python3 train_model.py` fine-tunes/continues
    from the latest checkpoint if one exists, otherwise trains from scratch.
    Also usable non-interactively (e.g. from the GUI, as a subprocess) via
    --mode/--epochs/--patience/--imgsz/--data.
    """
    args = parse_args()
    do_finetune = args.mode == "finetune" or (args.mode == "auto" and find_latest_best_weights())

    if do_finetune:
        finetune_model(data_yaml=args.data, extra_epochs=args.epochs or 30, patience=args.patience or 10, imgsz=args.imgsz)
    else:
        train_model(data_yaml=args.data, epochs=args.epochs or 100, patience=args.patience or 20, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
