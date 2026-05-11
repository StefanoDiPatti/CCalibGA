"""
Live capture with keypoints visualization and quality check of the pattern.
Key 's' to save, ESC to exit.
"""

import cv2
import numpy as np
import os
import argparse

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
PATTERN_PATH  = "./in/offline_pattern.png"   # <-- choose your pattern image here
SAVE_DIR      = "./in/offline_example"     # where to save the captured frames
MIN_GOOD_MATCHES = 15    # under this threshold the frame is considered "poor"
RATIO_THRESH     = 0.75  # Lowe's ratio test
MAX_DISPLAY_MATCHES = 40 # max lines of matches drawn (avoids visual chaos)

os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Carica il pattern e inizializza il detector
# --------------------------------------------------------------------------
pattern_img = cv2.imread(PATTERN_PATH)
if pattern_img is None:
    raise FileNotFoundError(f"Pattern not found: {PATTERN_PATH}")

pattern_gray = cv2.cvtColor(pattern_img, cv2.COLOR_BGR2GRAY)

detector = cv2.ORB_create(2000)
kp_pattern, des_pattern = detector.detectAndCompute(pattern_gray, None)
print(f"Pattern loaded: {len(kp_pattern)} keypoints detected.")

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def match_frame(frame_gray):
    """Returns (good_matches, kp_frame) for the current frame."""
    kp_frame, des_frame = detector.detectAndCompute(frame_gray, None)
    if des_frame is None or len(kp_frame) < 8:
        return [], kp_frame
    raw = matcher.knnMatch(des_pattern, des_frame, k=2)
    good = [m for m, n in raw if m.distance < RATIO_THRESH * n.distance]
    return good, kp_frame


def quality_color(n_matches):
    """Green/yellow/red based on the number of matches."""
    if n_matches >= MIN_GOOD_MATCHES * 2:
        return (0, 220, 60)    # good
    elif n_matches >= MIN_GOOD_MATCHES:
        return (0, 200, 255)   # acceptable
    else:
        return (0, 60, 220)    # insufficient


def draw_hud(canvas, n_matches, kp_frame, good_matches, saved_count):
    color = quality_color(n_matches)

    if n_matches >= MIN_GOOD_MATCHES:
        for m in good_matches:
            x, y = map(int, kp_frame[m.trainIdx].pt)
            cv2.circle(canvas, (x, y), 5, (0, 200, 60), -1)
            cv2.circle(canvas, (x, y), 5, (255, 255, 255), 1)

        label = f"Matches: {n_matches} OK - press S to save"
    else:
        label = "Pattern not visible"

    cv2.putText(canvas, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    border = 4
    cv2.rectangle(canvas,
                  (border, border),
                  (canvas.shape[1] - border, canvas.shape[0] - border),
                  color, border)

    cv2.putText(canvas, f"Saved: {saved_count}", (10, canvas.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas

def draw_matches_overlay(frame, kp_frame, good_matches):
    """
    Draws the match lines by overlaying the resized pattern
    on the current frame (similar to drawMatches but with pattern on the left).
    """
    ph, pw = pattern_gray.shape[:2]
    fh, fw = frame.shape[:2]

    # Change scale of the pattern to fit the frame height, keeping aspect ratio
    scale  = fh / ph
    pw_scaled = int(pw * scale)
    pat_resized = cv2.resize(pattern_img, (pw_scaled, fh))

    # Canvas with pattern on the left and frame on the right
    canvas = np.zeros((fh, fw + pw_scaled, 3), dtype=np.uint8)
    canvas[:, :pw_scaled] = pat_resized
    canvas[:, pw_scaled:] = frame

    # Draw a subset of matches for readability
    subset = good_matches[:MAX_DISPLAY_MATCHES]
    for m in subset:
        # Keypoint in the pattern (scaled)
        kp_p = kp_pattern[m.queryIdx]
        x1 = int(kp_p.pt[0] * scale)
        y1 = int(kp_p.pt[1] * scale)
        # Keypoint in the frame
        kp_f = kp_frame[m.trainIdx]
        x2 = int(kp_f.pt[0]) + pw_scaled
        y2 = int(kp_f.pt[1])

        col = quality_color(len(good_matches))
        cv2.line(canvas, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 3, (0, 200, 255), -1)
        cv2.circle(canvas, (x2, y2), 3, (0, 200, 255), -1)

    return canvas


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
num = 0
show_matches = False   # toggle with key 'M'

print("Commands: S = save | M = toggle match view | ESC = exit")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_gray          = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    good_matches, kp_fr = match_frame(frame_gray)
    n                   = len(good_matches)

    if show_matches and n > 0:
        display = draw_matches_overlay(frame.copy(), kp_fr, good_matches)
    else:
        display = draw_hud(frame.copy(), n, kp_fr, good_matches, num)

    cv2.imshow("Calibration Capture  [S=save | M=match view | ESC=exit]", display)

    k = cv2.waitKey(5) & 0xFF

    if k == 27:      # ESC
        break

    elif k == ord('s') or k == ord('S'):
        path = os.path.join(SAVE_DIR, f"img{num}.png")
        cv2.imwrite(path, frame)   # save the original frame, not the display
        print(f"[{num}] Saved: {path}  (matches: {n})")
        num += 1

    elif k == ord('m') or k == ord('M'):
        show_matches = not show_matches

cap.release()
cv2.destroyAllWindows()
print(f"\nSession terminated. {num} images saved in '{SAVE_DIR}/'")
