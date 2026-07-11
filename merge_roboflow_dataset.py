# Merge a Roboflow-exported YOLO detection dataset (bounding boxes around
# pivot/scale_min/scale_max/needle_tip) into gauge-readings-processed, in
# the pose-label format train_model.py/finetune_model() expect.
#
#   python3 merge_roboflow_dataset.py
#
# Two format differences had to be bridged, not just a file copy:
#
#   1. Class ID order. Roboflow assigns class IDs alphabetically, so this
#      export's data.yaml has 0=needle_tip, 1=pivot, 2=scale_max,
#      3=scale_min -- completely different from our own
#      gauge-readings-processed/data.yaml (0=pivot, 1=scale_min,
#      2=scale_max, 3=needle_tip). Copying labels over as-is would silently
#      swap classes (e.g. a "pivot" box would train as "needle_tip").
#      Fixed by remapping every class ID by NAME, read from each dataset's
#      own data.yaml, rather than assuming any fixed index order.
#
#   2. Label shape. This export has plain detection boxes (5 values per
#      line: class cx cy w h). Our dataset is a pose dataset (kpt_shape
#      [1, 3] -- 8 values per line: class cx cy w h kpt_x kpt_y visibility).
#      Since each box is really just a marker drawn around one point, the
#      box's own center is used as that point's keypoint (visibility=2).
#
# The new dataset's train/valid splits map onto our Train/Val. Its test
# split is intentionally left OUT of the merge -- folding held-out test
# data into training defeats the point of having a test set, so if you want
# to use those images too you'll need to do that on purpose.
import os
import shutil

import yaml

NEW_DATASET_DIR = "/Users/lchanpaibool/Desktop/Hack Week Project/processed_gauge_readings.yolo26"
DATASET_DIR = "/Users/lchanpaibool/Desktop/Hack Week Project/gauge-readings-processed"

# (new dataset split folder name, our dataset split folder name)
SPLIT_MAPPING = [("train", "Train"), ("valid", "Val")]


def load_names(yaml_path):
    """Return {class_id: name} from a data.yaml, handling both the dict
    form our own data.yaml uses and the plain-list form Roboflow uses."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return dict(enumerate(names))


def build_class_remap(new_names, our_names):
    """Map each class ID in the new dataset to the matching class ID in
    ours, by NAME -- not by assuming the two datasets number their classes
    the same way (they don't: Roboflow orders classes alphabetically)."""
    our_name_to_id = {name: class_id for class_id, name in our_names.items()}
    missing = set(new_names.values()) - set(our_name_to_id)
    if missing:
        raise ValueError(f"New dataset has classes we don't recognize: {missing}")
    return {new_id: our_name_to_id[name] for new_id, name in new_names.items()}


def convert_label_file(src_path, dest_path, class_remap):
    """Rewrite one detection-format label file (class cx cy w h) into our
    pose format (class cx cy w h kpt_x kpt_y visibility), using each box's
    own center as its keypoint, and remapping class IDs by name."""
    lines_out = []
    with open(src_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            old_class_id = int(parts[0])
            cx, cy, w, h = parts[1:5]
            new_class_id = class_remap[old_class_id]
            lines_out.append(f"{new_class_id} {cx} {cy} {w} {h} {cx} {cy} 2")

    with open(dest_path, "w") as f:
        f.write("\n".join(lines_out) + "\n")


def merge_split(new_split, our_split, class_remap):
    src_images = os.path.join(NEW_DATASET_DIR, new_split, "images")
    src_labels = os.path.join(NEW_DATASET_DIR, new_split, "labels")
    dest_images = os.path.join(DATASET_DIR, "images", our_split)
    dest_labels = os.path.join(DATASET_DIR, "labels", our_split)
    os.makedirs(dest_images, exist_ok=True)
    os.makedirs(dest_labels, exist_ok=True)

    added = 0
    skipped = 0
    for filename in sorted(os.listdir(src_images)):
        stem = os.path.splitext(filename)[0]
        label_src = os.path.join(src_labels, f"{stem}.txt")
        if not os.path.exists(label_src):
            skipped += 1
            continue

        image_dest = os.path.join(dest_images, filename)
        label_dest = os.path.join(dest_labels, f"{stem}.txt")
        if os.path.exists(image_dest) or os.path.exists(label_dest):
            print(f"  Skipping {filename} -- already exists in {our_split}")
            skipped += 1
            continue

        shutil.copy2(os.path.join(src_images, filename), image_dest)
        convert_label_file(label_src, label_dest, class_remap)
        added += 1

    print(f"{new_split} -> {our_split}: added {added}, skipped {skipped}")
    return added


def main():
    new_names = load_names(os.path.join(NEW_DATASET_DIR, "data.yaml"))
    our_names = load_names(os.path.join(DATASET_DIR, "data.yaml"))
    class_remap = build_class_remap(new_names, our_names)
    print(f"Class remap (new id -> our id): {class_remap}")
    print(f"  (new dataset names: {new_names})")
    print(f"  (our dataset names: {our_names})")
    print()

    total_added = 0
    for new_split, our_split in SPLIT_MAPPING:
        total_added += merge_split(new_split, our_split, class_remap)

    print(f"\nDone: {total_added} new labeled images merged into {DATASET_DIR}.")
    print("The 'test' split was intentionally left out of the merge -- see this file's module docstring.")
    print("Run `python3 train_model.py` next to fine-tune on the combined dataset.")


if __name__ == "__main__":
    main()
