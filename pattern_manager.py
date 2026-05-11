import cv2
import numpy as np


TARGET_SIZE          = 300
MIN_MATCH_THRESHOLD  = 8
MAX_DISPLAY_MATCHES  = 40
RATIO_THRESH         = 0.82
MIN_FEAT_PATTERN     = 15   # minimum matches required to consider the pattern found


class PatternManager:
    def __init__(self):
        self.pattern       = cv2.ccalib.CustomPattern()
        self.roi           = [0, 0, 0, 0]
        self.mdown         = False
        self._pattern_img  = None
        self._pattern_gray = None
        self._kp_pattern   = None
        self._des_pattern  = None
        self._detector     = cv2.ORB_create(2000)
        self._matcher      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._last_valid   = None
        self._miss_count   = 0


    # ------------------------------------------------------------------ ROI
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi[0], self.roi[1] = x, y
            self.roi[2], self.roi[3] = 0, 0
            self.mdown = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.roi[2] = x - self.roi[0]
            self.roi[3] = y - self.roi[1]
            self.mdown = False
        elif event == cv2.EVENT_MOUSEMOVE and self.mdown:
            self.roi[2] = x - self.roi[0]
            self.roi[3] = y - self.roi[1]


    def draw_roi(self, frame):
        x, y, w, h = self.roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame


    # ---------------------------------------------------------- Pattern create
    def set_pattern_img(self, img):
        """Save pattern image and precompute ORB keypoints for visualization."""
        self._pattern_img  = img.copy()
        self._pattern_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._kp_pattern, self._des_pattern = self._detector.detectAndCompute(
            self._pattern_gray, None
        )


    def create_pattern(self, frame):
        """Cut ROI, resize to TARGET_SIZE, create ccalib pattern, save ORB data."""
        x, y, w, h = self.roi
        if w <= 0 or h <= 0:
            return False, None
        roi = frame[y:y+h, x:x+w]
        ph, pw = roi.shape[:2]
        if max(ph, pw) > TARGET_SIZE:
            factor = TARGET_SIZE / max(ph, pw)
            roi = cv2.resize(roi, (int(pw * factor), int(ph * factor)),
                             interpolation=cv2.INTER_AREA)
        out = np.zeros_like(roi)
        success = self.pattern.create(roi, (roi.shape[1], roi.shape[0]), out)
        if success:
            self.set_pattern_img(roi)
        return success, out


    # ----------------------------------------------------------- find_pattern
    def find_pattern(self, frame, ratio=0.65, proj_error=6.0):
        """
        Find pattern in frame using ccalib.
        No persistence buffer — returns False immediately if pattern is lost
        or if matched features fall below MIN_FEAT_PATTERN.
        """
        res, matched_feat, pattern_pts, out, H, corners = self.pattern.findPattern(
            frame, ratio=ratio, proj_error=proj_error, refine_position=False
        )
        if res and matched_feat is not None and len(matched_feat) >= MIN_FEAT_PATTERN:
            self._last_valid = (res, matched_feat, pattern_pts, out, H, corners)
            self._miss_count = 0
            return self._last_valid
        self._miss_count += 1
        self._last_valid = None
        return False, None, None, None, None, None


    # -------------------------------------------------------- visualization
    def _quality_color(self, n):
        if n >= MIN_MATCH_THRESHOLD * 2: return (0, 220, 60)
        elif n >= MIN_MATCH_THRESHOLD:   return (0, 200, 255)
        else:                            return (0, 60, 220)


    def build_match_display(self, frame, matched_feat, pattern_found=False):
        """
        Camera feed full screen, pattern thumbnail fixed top-right,
        match lines only when pattern_found=True (confirmed by find_pattern).
        Status banner always visible at bottom-left.
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        good, kp_frame = [], []
        if self._des_pattern is not None and self._kp_pattern is not None:
            kp_frame, des_frame = self._detector.detectAndCompute(frame_gray, None)
            if des_frame is not None and len(kp_frame) >= 8:
                raw  = self._matcher.knnMatch(self._des_pattern, des_frame, k=2)
                good = [m for m, n in raw
                        if m.distance < RATIO_THRESH * n.distance]

        # Lines and status are gated on find_pattern verdict, not raw ORB count
        n      = len(good) if pattern_found else 0
        color  = self._quality_color(n) if pattern_found else (0, 60, 220)
        status = "TRACKED" if pattern_found else "searching..."
        display = frame.copy()
        fh, fw  = display.shape[:2]

        # --- Thumbnail always visible top-right ---
        if self._pattern_img is not None:
            thumb_w = fw // 4
            ph, pw  = self._pattern_gray.shape[:2]
            thumb_h = int(ph * thumb_w / pw)
            thumb   = cv2.resize(self._pattern_img, (thumb_w, thumb_h),
                                 interpolation=cv2.INTER_LINEAR)
            ox, oy  = fw - thumb_w - 8, 8

            # Semi-transparent dark background behind thumbnail
            overlay = display.copy()
            cv2.rectangle(overlay, (ox - 4, oy - 4),
                          (ox + thumb_w + 4, oy + thumb_h + 4), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
            display[oy:oy + thumb_h, ox:ox + thumb_w] = thumb

            # Match lines — drawn only when find_pattern confirmed the pattern
            if pattern_found:
                scale_t = thumb_w / pw
                for m in good[:MAX_DISPLAY_MATCHES]:
                    x1 = int(self._kp_pattern[m.queryIdx].pt[0] * scale_t) + ox
                    y1 = int(self._kp_pattern[m.queryIdx].pt[1] * scale_t) + oy
                    x2 = int(kp_frame[m.trainIdx].pt[0])
                    y2 = int(kp_frame[m.trainIdx].pt[1])
                    cv2.line(display, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
                    cv2.circle(display, (x1, y1), 2, (0, 200, 255), -1)
                    cv2.circle(display, (x2, y2), 3, (0, 200, 255), -1)

        # --- Status banner bottom-left ---
        cv2.rectangle(display, (0, fh - 44), (420, fh), (0, 0, 0), -1)
        cv2.putText(display, f"{n} matches  |  {status}",
                    (8, fh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return display