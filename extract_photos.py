
# Extract photos for dataset from videos
import cv2
import os
import re

from process_image import process_frame

KAGGLE_DATA_DIR = "/Users/lchanpaibool/.cache/kagglehub/datasets/juliusgrassme/pressure-gauge-reader-data/versions/1/Data"
CURR_VIDEO_DIR = "/Users/lchanpaibool/Desktop/Hack Week Project"

# DATA_PATHS can mix individual video files AND whole folders of videos --
# any entry that's a directory gets expanded into the video files inside it
# by expand_video_paths() below.
DATA_PATHS = [
    os.path.join(KAGGLE_DATA_DIR, "1 Training videos", "edited videos"),  # directory
    os.path.join(KAGGLE_DATA_DIR, "3 Misc handheld videos"),              # directory
    os.path.join(CURR_VIDEO_DIR, "office-gauge.mov"),
    os.path.join(CURR_VIDEO_DIR, "virtual-gauge-2.mov"),
    os.path.join(CURR_VIDEO_DIR, "virtual-gauge-3.mov"),
    os.path.join(CURR_VIDEO_DIR, "virtual-gauge-4.mov"),
    os.path.join(CURR_VIDEO_DIR, "virtual-gauge.mov"),
]
EXTRACTED_FRAMES_DIR = "/Users/lchanpaibool/Desktop/Hack Week Project/new-frames"
PROCESSED_FRAMES_DIR = "/Users/lchanpaibool/Desktop/Hack Week Project/processed_images"

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def expand_video_paths(paths):
    """Expand any directory entries in `paths` into the individual video
    files they contain; plain file paths pass through unchanged.

    Lets DATA_PATHS mix single video files with whole folders of videos --
    e.g. the Kaggle dataset's "edited videos" folders alongside your own
    individual .mov files.
    """
    expanded = []
    for path in paths:
        if os.path.isdir(path):
            found = sorted(
                os.path.join(path, name)
                for name in os.listdir(path)
                if name.lower().endswith(VIDEO_EXTENSIONS)
            )
            if not found:
                print(f"Warning: no video files found in directory {path}")
            expanded.extend(found)
        else:
            expanded.append(path)
    return expanded


def extract_all_frames(video_path, output_folder, processed_folder=PROCESSED_FRAMES_DIR, frame_interval=10):
    """Save every Nth frame of a video as a .jpg for annotation, and also
    save a process_frame()-processed copy of that same frame (grayscale +
    contrast + gauge-circle/scale-marking trace) into processed_folder.

    Filenames are prefixed with the video's own name (sanitized) so frames
    from several different videos can be extracted into the SAME
    output_folder without colliding -- e.g. handheld1.mp4 and handheld4.mp4
    both have a "frame 0000", but they'll be saved as
    handheld1_frame_0000.jpg and handheld4_frame_0000.jpg instead of one
    overwriting the other. The processed copies use the same naming scheme,
    just in processed_folder instead, so a raw frame and its processed
    counterpart are always easy to match up by filename.
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)

    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    video_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", video_stem)  # "tripod 20 deg" -> "tripod_20_deg"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = f"{video_stem}_frame_{frame_count:04d}.jpg"

            filename = os.path.join(output_folder, frame_name)
            cv2.imwrite(filename, frame)

            processed = process_frame(frame)
            processed_filename = os.path.join(processed_folder, frame_name)
            cv2.imwrite(processed_filename, processed)

            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extraction complete for {video_stem}: {saved_count} frames saved.")
    return saved_count


def extract_frames_from_videos(video_paths, output_folder, processed_folder=PROCESSED_FRAMES_DIR, frame_interval=10):
    """Extract frames (+ their processed copies) from several videos into
    shared output_folder / processed_folder directories.
    Each video is namespaced by its own filename (see extract_all_frames),
    so this is safe to call repeatedly / with overlapping video lists --
    it only ever adds new files, it never overwrites the folder.

    Any single video that fails to open/read is skipped with a warning
    instead of aborting the whole batch -- with DATA_PATHS now pulling in
    entire folders of Kaggle videos, one corrupt/unsupported file shouldn't
    stop every other video from being processed.
    """
    total_saved = 0
    for video_path in video_paths:
        try:
            total_saved += extract_all_frames(video_path, output_folder, processed_folder, frame_interval)
        except Exception as e:
            print(f"Warning: skipping {video_path} -- {e}")
    print(f"\nDone: {total_saved} new frames saved across {len(video_paths)} videos into {output_folder}")
    print(f"Processed copies saved into {processed_folder}")
    return total_saved


def main():
    video_paths = expand_video_paths(DATA_PATHS)
    extract_frames_from_videos(video_paths, EXTRACTED_FRAMES_DIR, PROCESSED_FRAMES_DIR, frame_interval=10)


if __name__ == "__main__":
    main()
