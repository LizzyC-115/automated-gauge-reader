import cv2
from PIL import Image
import os
import math
import numpy as np

IMAGE_PATH = "/Users/lchanpaibool/Desktop/Hack Week Project/new-frames/office-gauge_frame_0230.jpg"
OUTPUT_DIR = "/Users/lchanpaibool/Desktop/Hack Week Project/"


def process_frame(frame):
    """Grayscale + contrast-enhance a single frame that's already in memory
    (a BGR numpy array, straight from cv2.imread/cv2.VideoCapture) and hand
    back a 3-channel BGR image.

    This is the version read_measurement.py calls per-frame in run_inference/
    run_video/live_stream -- no disk I/O, so it's cheap enough to run on
    every video frame. It stays 3-channel (not single-channel grayscale) so
    the array shape still matches what the YOLO model expects, even though
    all three channels now hold the same grayscale value.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # --- Find the ONE gauge bezel circle -------------------------------
    # HoughCircles does its own internal edge detection (param1 is its Canny
    # threshold), so it wants a smoothed GRAYSCALE image, not a binary edge
    # map. A median blur is the standard prep for this transform -- it
    # knocks out salt-and-pepper noise (which HoughCircles is sensitive to)
    # better than Gaussian does.
    #
    # Two bugs from the previous version, both visible in your test output:
    #   1. maxRadius=0 means "no limit" -- with nothing capping it, an
    #      unrelated large arc elsewhere in the shot (like the tank
    #      illustration) can score as a giant, bogus "circle" alongside the
    #      real gauge bezel.
    #   2. Looping over every row of circles[0, :] draws ALL detected
    #      circles. Even after bounding the radius, you generally still get
    #      more than one candidate -- circles[0, :] is ordered strongest
    #      match first, so keeping only circles[0, 0] instead of the whole
    #      loop gets you the single best gauge circle.
    circle_input = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        circle_input,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min(h, w),                # we only ever want one match back
        param1=100,
        param2=30,                        # 50 was too strict and silently found nothing
        minRadius=min(h, w) // 6,         # skip tiny noise circles
        maxRadius=min(h, w) // 2,         # the gauge can't be bigger than the frame itself
    )

    gauge_circle = None
    if circles is not None:
        x, y, r = np.uint16(np.around(circles[0, 0]))
        gauge_circle = (int(x), int(y), int(r))

    # --- Mask everything outside that circle to black ------------------
    # This is what actually removes clutter like the "TANK" text/graphic --
    # without it, the edge detection and line search below run over the
    # WHOLE frame and pick up edges from anything in the shot, not just the
    # gauge's scale markings.
    if gauge_circle is not None:
        cx, cy, r = gauge_circle
        mask = np.zeros_like(gray)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        gray = cv2.bitwise_and(gray, gray, mask=mask)

    # --- Edges + needle-candidate lines, now confined to the gauge face ---
    # Three tweaks specifically to make the tick marks/numbers show up more
    # clearly instead of fading into faint, broken fragments:
    #   1. CLAHE contrast boost before blurring -- thin tick strokes are
    #      often low-contrast against the dial face, especially with glare,
    #      so this pulls them out before Canny ever runs.
    #   2. A smaller blur kernel (3x3 instead of 5x5) -- the tick marks and
    #      number strokes are thin, and a bigger kernel smears them out
    #      before they even reach the edge detector.
    #   3. A light dilation on the Canny output -- Canny often finds the
    #      thin strokes as broken, one-pixel-wide fragments; dilating by one
    #      pass thickens and reconnects them into solid, visible marks
    #      without dilating so much that adjacent ticks merge together.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    dst = cv2.Canny(blurred, 50, 150)
    dst = cv2.dilate(dst, np.ones((2, 2), np.uint8), iterations=1)
    cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    # NOTE: this used to also run cv2.HoughLinesP() here and draw every
    # detected segment in red, 3px thick, directly on top of this same
    # image. That's what was blotting out the scale markings -- a generic
    # Hough Line search over the whole dial face fires on every tick mark
    # and number stroke, not just the needle, so nearly the entire white
    # trace got painted over. Pulled it out entirely so this function just
    # returns the clean scale-marking trace + the single gauge circle.
    #
    # If/when you want needle detection back, it needs to be a much more
    # constrained search rather than "find every line on the whole face" --
    # e.g. only keep HoughLinesP segments that pass close to the gauge
    # center (cx, cy) above, or do the radial angular-scan approach instead
    # (score dark-pixel density along candidate angles from the pivot).
    # That way it isolates the one line that actually radiates from the
    # pivot instead of every tick/number edge that happens to look line-ish.

    # --- Draw the single gauge circle + its center on top --------------
    if gauge_circle is not None:
        cx, cy, r = gauge_circle
        cv2.circle(cdstP, (cx, cy), r, (0, 255, 0), 3)     # outer bezel, green
        cv2.circle(cdstP, (cx, cy), 2, (0, 255, 255), 3)   # center point, yellow -- candidate pivot

    return cdstP


def process(image_path):
    """File-based wrapper around process_frame() -- loads an image from
    disk, grayscale+contrast enhances it, saves the result next to the
    original, and returns the new path. Handy for one-off inspection of a
    single photo; the live pipeline uses process_frame() directly instead
    since writing every video frame to disk would be slow.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    processed = process_frame(frame)

    stem = os.path.splitext(os.path.basename(OUTPUT_DIR))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{stem}_processed.jpg")
    cv2.imwrite(output_path, processed)
    return output_path


def main():
    processed_image = process(IMAGE_PATH)
    print(f"Saved: {processed_image}")
    img = Image.open(processed_image)
    img.show()



if __name__ == "__main__":
    main()
