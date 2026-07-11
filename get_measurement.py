import math

"""
EXAMPLE:

    scale_min    conf=0.89  xy=(717.4, 717.1)
    needle_tip   conf=0.66  xy=(668.9, 601.3)
    pivot        conf=0.63  xy=(913.0, 536.0)
    scale_max    conf=0.51  xy=(1098.6, 733.2)
"""

MIN_VALUE = 0.0
MAX_VALUE = 2.5
REQUIRED_POINTS = ("pivot", "needle_tip", "scale_min", "scale_max")


def get_angle(pt1, pt2, min_value=MIN_VALUE, max_value=MAX_VALUE):
    """Return the angle in degrees from pt1 to pt2, where 0 degrees is
    straight up (negative y direction) and angles increase clockwise.
    """
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    # atan2(dx, -dy) rather than atan2(dy, dx) -- this rotates the reference
    # so "straight up" reads as 0 instead of "pointing right", matching the
    # docstring and making printed angles intuitive to sanity-check by eye.
    angle_rad = math.atan2(dx, -dy)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def get_reading(points, min_value=MIN_VALUE, max_value=MAX_VALUE, coincidence_threshold_deg=5.0):
    """Compute the gauge's numeric reading from detected keypoints.

    points is the dict shape produced by extract_keypoints() in
    read_measurement.py:
        {"pivot": (x, y, conf), "needle_tip": (x, y, conf),
         "scale_min": (x, y, conf), "scale_max": (x, y, conf)}
    min_value/max_value are what scale_min/scale_max correspond to on the
    physical dial (e.g. 0 and 4 bar).

    Returns the interpolated reading, or None if any of the 4 required
    points is missing/undetected, or if needle_tip landed essentially on
    top of scale_min (see below) -- so callers can tell "no reading" apart
    from "reading is 0".
    """
    if not all(name in points for name in REQUIRED_POINTS):
        return None

    pivot = points["pivot"][:2]
    needle_tip = points["needle_tip"][:2]
    scale_min = points["scale_min"][:2]
    scale_max = points["scale_max"][:2]

    angle_needle = get_angle(pivot, needle_tip)
    angle_min = get_angle(pivot, scale_min)
    angle_max = get_angle(pivot, scale_max)

    gap_to_min = (angle_needle - angle_min) % 360
    gap_to_min = min(gap_to_min, 360 - gap_to_min)
    if gap_to_min <= coincidence_threshold_deg:
        return None

    sweep_total = (angle_max - angle_min) % 360
    sweep_needle = (angle_needle - angle_min) % 360

    if sweep_total == 0:
        return None  # degenerate: scale_min and scale_max landed on the same angle

    if sweep_needle > sweep_total:
        past_max = sweep_needle - sweep_total
        before_min = 360 - sweep_needle
        fraction = 1.0 if past_max <= before_min else 0.0
    else:
        fraction = sweep_needle / sweep_total

    return min_value + fraction * (max_value - min_value)


def main():
    pivot = (913.0, 536.0)
    needle_tip = (668.9, 601.3)
    scale_min = (717.4, 717.1)
    scale_max = (1098.6, 733.2)
    
    to_tip = get_angle(pivot, needle_tip)
    to_min = get_angle(pivot, scale_min)
    to_max = get_angle(pivot, scale_max)
    
    print(f"Angle to needle tip: {to_tip:.2f} degrees")
    print(f"Angle to scale min: {to_min:.2f} degrees")
    print(f"Angle to scale max: {to_max:.2f} degrees")

if __name__=="__main__":
    main()