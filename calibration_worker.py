import cv2
from cv2 import ccalib
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QTextEdit, QGroupBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from pattern_manager import PatternManager
from camera_calibrator import CameraCalibrator
from utils import ChessboardValidator
from ga_worker import GAWorker

TARGET_SIZE = 300  # px on the longest side — tune between 200-400 if needed

def cv2_to_qpixmap(frame):
    """Convert an OpenCV BGR frame to QPixmap."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

# ---------------------------------------------------------------------------
# Shared base widget: video label + log
# ---------------------------------------------------------------------------
class BaseCalibrationWidget(QWidget):
    def __init__(self, title, description, out_dir="out"):
        super().__init__()
        self.out_dir    = out_dir
        self._ga_worker = None
        layout = QVBoxLayout(self)

        desc = QLabel(f"<b>{title}</b><br><small>{description}</small>")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        content = QHBoxLayout()
        layout.addLayout(content)

        self.video_label = QLabel("No active feed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background: #1a1a1a; color: white; border-radius: 6px;")
        content.addWidget(self.video_label, 3)

        right_panel = QVBoxLayout()
        content.addLayout(right_panel, 1)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(180)
        log_layout.addWidget(self.log)
        right_panel.addWidget(log_group)

        ga_log_group = QGroupBox("🧬 GA Optimizer Log")
        ga_log_layout = QVBoxLayout(ga_log_group)
        self.ga_log = QTextEdit()
        self.ga_log.setReadOnly(True)
        self.ga_log.setMaximumHeight(150)
        ga_log_layout.addWidget(self.ga_log)
        right_panel.addWidget(ga_log_group)

        self.status_label = QLabel("Status: Awaiting")
        self.status_label.setWordWrap(True)
        right_panel.addWidget(self.status_label)
        right_panel.addStretch()

        self.controls = QVBoxLayout()
        right_panel.addLayout(self.controls)

    def log_message(self, msg):
        self.log.append(msg)

    def update_frame(self, pixmap):
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    def _launch_ga(self, obj_points, matched_points, img_w, img_h):
        self.ga_log.clear()
        self.ga_log.append("🧬 Starting GA optimization in background...")
        self.status_label.setText("Status: GA Optimization running...")

        self._ga_worker = GAWorker(obj_points, matched_points, img_w, img_h, self.out_dir)
        self._ga_worker.log_signal.connect(self.ga_log.append)
        self._ga_worker.done_signal.connect(self._on_ga_done)
        self._ga_worker.start()

    def _on_ga_done(self, best_rms, n_active):
        self.ga_log.append(
            f"\n✅ GA optimization completed!\n"
            f"   Best reprojection error : {best_rms:.5f}\n"
            f"   Optimal images selected : {n_active}\n"
            f"   Saved to                : {self.out_dir}/ga_best_calibration.xml"
        )
        self.status_label.setText(
            f"Status: GA done — Best RMS: {best_rms:.5f} | Images: {n_active}"
        )


# ---------------------------------------------------------------------------
# MODE 1: Real-Time Calibration
# ---------------------------------------------------------------------------
class RealtimeWorker(QThread):
    frame_ready      = pyqtSignal(QPixmap)
    log_signal       = pyqtSignal(str)
    status_signal    = pyqtSignal(str)
    calibration_done = pyqtSignal(float)
    ga_ready         = pyqtSignal(list, list, int, int)

    def __init__(self):
        super().__init__()
        self.p_manager     = PatternManager()
        self.calibrator    = CameraCalibrator()
        self.validator     = ChessboardValidator()
        self._running      = False
        self._phase        = "select_roi"
        self._capture_next = False
        self._do_calibrate = False
        self._frame        = None
        self._smooth_rvec = None
        self._smooth_tvec = None

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(0)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
            self._frame = frame.copy()

            SMOOTH_ALPHA = 0.35   # 0.0 = massimo smoothing, 1.0 = nessuno
            MIN_TRACKING_FEAT = 20

            if self._phase == "select_roi":
                canvas = self.p_manager.draw_roi(frame.copy())
                self.frame_ready.emit(cv2_to_qpixmap(canvas))

            elif self._phase == "collect":
                res, matched_feat, pattern_pts, out, H, corners = self.p_manager.find_pattern(frame)
                display = self.p_manager.build_match_display(frame, matched_feat, pattern_found=res)
                if res and self._capture_next:
                    self.calibrator.add_points(pattern_pts, matched_feat)
                    count = len(self.calibrator.obj_points)
                    self.log_signal.emit(f"Frame {count} acquired.")
                    self.status_signal.emit(f"Frames collected: {count}")
                    self._capture_next = False
                self.frame_ready.emit(cv2_to_qpixmap(display))

                if self._do_calibrate:
                    self._do_calibrate = False
                    h, w = frame.shape[:2]
                    rms = self.calibrator.calibrate(w, h)
                    if rms:
                        self.calibration_done.emit(rms)
                        self._phase = "calibrated"
                        self.ga_ready.emit(
                            self.calibrator.obj_points,
                            self.calibrator.matched_points,
                            w, h
                        )
            elif self._phase == "calibrated":
                res, matched_feat, pattern_pts, out, H, corners = self.p_manager.find_pattern(frame)
                if res and len(matched_feat) >= MIN_TRACKING_FEAT:
                    ok, rvec, tvec = self.p_manager.pattern.findRt(
                        pattern_pts, matched_feat,
                        self.calibrator.K,
                        self.calibrator.dist_coeff, None, None
                    )
                    if ok:
                        # Smoothing esponenziale
                        if self._smooth_rvec is None:
                            self._smooth_rvec = rvec.copy()
                            self._smooth_tvec = tvec.copy()
                        else:
                            self._smooth_rvec = SMOOTH_ALPHA * rvec + (1 - SMOOTH_ALPHA) * self._smooth_rvec
                            self._smooth_tvec = SMOOTH_ALPHA * tvec + (1 - SMOOTH_ALPHA) * self._smooth_tvec

                        h_frame, w_frame = frame.shape[:2]
                        axis_len = int(min(h_frame, w_frame) * 0.15)
                        frame = self.p_manager.pattern.drawOrientation(
                            frame, self._smooth_tvec, self._smooth_rvec,
                            self.calibrator.K,
                            self.calibrator.dist_coeff, axis_len, 3
                        )
                else:
                    # Pattern perso — resetta lo smooth per non avere "fantasmi"
                    self._smooth_rvec = None
                    self._smooth_tvec = None
                self.frame_ready.emit(cv2_to_qpixmap(frame))

        cap.release()

    def confirm_roi(self):
        success, out = self.p_manager.create_pattern(self._frame)
        if success:
            self._phase = "collect"
            self.log_signal.emit("Pattern created. Click 'Capture Frame' to collect data.")
            self.status_signal.emit("Data collection mode active.")
        else:
            self.log_signal.emit("Error: Invalid ROI.")

    def capture_frame(self):
        self._capture_next = True

    def trigger_calibrate(self):
        self._do_calibrate = True

    def stop(self):
        self._running = False
        self.wait()


class RealtimeCalibrationWidget(BaseCalibrationWidget):
    def __init__(self, out_dir="out"):
        super().__init__(
            "Real-Time Calibration",
            "Draw the ROI on the pattern with the mouse, then capture frames to calibrate."
            " GA optimization starts automatically after calibration.",
            out_dir=out_dir
        )
        self.worker = None

        self.btn_start       = QPushButton("▶ Start Camera")
        self.btn_confirm_roi = QPushButton("✅ Confirm ROI")
        self.btn_capture     = QPushButton("📸 Capture Frame")
        self.btn_calibrate   = QPushButton("⚙️ Perform Calibration + GA")
        self.btn_stop        = QPushButton("⏹ Stop")

        self.btn_confirm_roi.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.btn_calibrate.setEnabled(False)

        for btn in [self.btn_start, self.btn_confirm_roi, self.btn_capture,
                    self.btn_calibrate, self.btn_stop]:
            btn.setMinimumHeight(36)
            self.controls.addWidget(btn)

        self.btn_start.clicked.connect(self.start_worker)
        self.btn_confirm_roi.clicked.connect(self.confirm_roi)
        self.btn_capture.clicked.connect(self.capture)
        self.btn_calibrate.clicked.connect(self.calibrate)
        self.btn_stop.clicked.connect(self.stop_worker)

        self.video_label.setMouseTracking(True)
        self.video_label.mousePressEvent   = self._mouse_press
        self.video_label.mouseMoveEvent    = self._mouse_move
        self.video_label.mouseReleaseEvent = self._mouse_release

    def start_worker(self):
        self.worker = RealtimeWorker()
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.log_signal.connect(self.log_message)
        self.worker.status_signal.connect(lambda m: self.status_label.setText(f"Status: {m}"))
        self.worker.calibration_done.connect(
            lambda rms: self.log_message(
                f"✅ Calibration completed! RMS = {rms:.4f} — launching GA..."
            )
        )
        self.worker.ga_ready.connect(self._launch_ga)
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_confirm_roi.setEnabled(True)
        self.log_message("Camera started. Draw the ROI on the pattern.")
        self.status_label.setText("Status: ROI Selection")

    def _scale_coords(self, x, y):
        lw, lh = self.video_label.width(), self.video_label.height()
        fw, fh = 640, 480
        scale  = min(lw / fw, lh / fh)
        ox     = (lw - fw * scale) / 2
        oy     = (lh - fh * scale) / 2
        return max(0, int((x - ox) / scale)), max(0, int((y - oy) / scale))

    def _mouse_press(self, event):
        if self.worker and self.worker._phase == "select_roi":
            x, y = self._scale_coords(event.position().x(), event.position().y())
            self.worker.p_manager.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, None, None)

    def _mouse_move(self, event):
        if self.worker and self.worker._phase == "select_roi":
            x, y = self._scale_coords(event.position().x(), event.position().y())
            self.worker.p_manager.on_mouse(cv2.EVENT_MOUSEMOVE, x, y, None, None)

    def _mouse_release(self, event):
        if self.worker and self.worker._phase == "select_roi":
            x, y = self._scale_coords(event.position().x(), event.position().y())
            self.worker.p_manager.on_mouse(cv2.EVENT_LBUTTONUP, x, y, None, None)

    def confirm_roi(self):
        if self.worker:
            self.worker.confirm_roi()
            self.btn_confirm_roi.setEnabled(False)
            self.btn_capture.setEnabled(True)
            self.btn_calibrate.setEnabled(True)

    def capture(self):
        if self.worker:
            self.worker.capture_frame()

    def calibrate(self):
        if self.worker:
            self.worker.trigger_calibrate()

    def stop_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
            self.btn_start.setEnabled(True)


# ---------------------------------------------------------------------------
# MODE 2: Pattern Offline + Real-Time Photos
# ---------------------------------------------------------------------------
class OfflinePatternWorker(QThread):
    frame_ready      = pyqtSignal(QPixmap)
    log_signal       = pyqtSignal(str)
    status_signal    = pyqtSignal(str)
    calibration_done = pyqtSignal(float)
    ga_ready         = pyqtSignal(list, list, int, int)

    def __init__(self, pattern_path):
        super().__init__()
        self.pattern_path  = pattern_path
        self.p_manager     = PatternManager()
        self.calibrator    = CameraCalibrator()
        self._running      = False
        self._phase        = "collect"
        self._capture_next = False
        self._do_calibrate = False
        self._frame        = None
        self.img_w         = 640
        self.img_h         = 480

    def run(self):
        pattern_img = cv2.imread(self.pattern_path)
        if pattern_img is None:
            self.log_signal.emit("Error: Unable to load the pattern.")
            return

        # Resize pattern to match its expected appearance size in the camera feed.
        # Descriptors built at a much larger resolution won't match a small pattern in frame.
        ph, pw = pattern_img.shape[:2]
        if max(ph, pw) > TARGET_SIZE:
            factor = TARGET_SIZE / max(ph, pw)
            pattern_img = cv2.resize(pattern_img,
                                    (int(pw * factor), int(ph * factor)),
                                    interpolation=cv2.INTER_AREA)
            ph, pw = pattern_img.shape[:2]
            self.log_signal.emit(f"Pattern resized to {pw}x{ph} for descriptor matching.")

        out = np.zeros_like(pattern_img)
        if not self.p_manager.pattern.create(pattern_img, (pw, ph), out):
            self.log_signal.emit("Error: Pattern creation failed.")
            return
        self.p_manager.set_pattern_img(pattern_img)
        self.log_signal.emit("Pattern loaded successfully.")
        self._running = True
        cap = cv2.VideoCapture(0)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
            self._frame = frame.copy()
            self.img_h, self.img_w = frame.shape[:2]

            SMOOTH_ALPHA = 0.35   # 0.0 = massimo smoothing, 1.0 = nessuno
            MIN_TRACKING_FEAT = 20

            if self._phase == "collect":
                res, matched_feat, pattern_pts, out_frame, H, corners = self.p_manager.find_pattern(frame)
                display = self.p_manager.build_match_display(frame, matched_feat, pattern_found=res)
                if res and self._capture_next:
                    self.calibrator.add_points(pattern_pts, matched_feat)
                    count = len(self.calibrator.obj_points)
                    self.log_signal.emit(f"Frame {count} acquired.")
                    self.status_signal.emit(f"Frames collected: {count}")
                    self._capture_next = False
                self.frame_ready.emit(cv2_to_qpixmap(display))

                if self._do_calibrate:
                    self._do_calibrate = False
                    rms = self.calibrator.calibrate(self.img_w, self.img_h)
                    if rms:
                        self.calibration_done.emit(rms)
                        self._phase = "calibrated"
                        self.ga_ready.emit(
                            self.calibrator.obj_points,
                            self.calibrator.matched_points,
                            self.img_w, self.img_h
                        )

            elif self._phase == "calibrated":
                res, matched_feat, pattern_pts, out, H, corners = self.p_manager.find_pattern(frame)
                if res and len(matched_feat) >= MIN_TRACKING_FEAT:
                    ok, rvec, tvec = self.p_manager.pattern.findRt(
                        pattern_pts, matched_feat,
                        self.calibrator.K,
                        self.calibrator.dist_coeff, None, None
                    )
                    if ok:
                        # Smoothing esponenziale
                        if self._smooth_rvec is None:
                            self._smooth_rvec = rvec.copy()
                            self._smooth_tvec = tvec.copy()
                        else:
                            self._smooth_rvec = SMOOTH_ALPHA * rvec + (1 - SMOOTH_ALPHA) * self._smooth_rvec
                            self._smooth_tvec = SMOOTH_ALPHA * tvec + (1 - SMOOTH_ALPHA) * self._smooth_tvec

                        h_frame, w_frame = frame.shape[:2]
                        axis_len = int(min(h_frame, w_frame) * 0.15)
                        frame = self.p_manager.pattern.drawOrientation(
                            frame, self._smooth_tvec, self._smooth_rvec,
                            self.calibrator.K,
                            self.calibrator.dist_coeff, axis_len, 3
                        )
                else:
                    # Pattern perso — resetta lo smooth per non avere "fantasmi"
                    self._smooth_rvec = None
                    self._smooth_tvec = None
                self.frame_ready.emit(cv2_to_qpixmap(frame))

        cap.release()

    def capture_frame(self):
        self._capture_next = True

    def trigger_calibrate(self):
        self._do_calibrate = True

    def stop(self):
        self._running = False
        self.wait()


class OfflinePatternCalibrationWidget(BaseCalibrationWidget):
    def __init__(self, out_dir="out"):
        super().__init__(
            "Pattern Offline + Real-Time Photos",
            "Load a pattern image from file, then take photos with the camera to calibrate."
            " GA optimization starts automatically after calibration.",
            out_dir=out_dir
        )
        self.worker       = None
        self.pattern_path = None

        self.btn_load_pattern = QPushButton("📂 Load Pattern")
        self.pattern_info     = QLabel("No pattern loaded")
        self.pattern_info.setStyleSheet("color: gray; font-style: italic;")
        self.btn_start     = QPushButton("▶ Start Camera")
        self.btn_capture   = QPushButton("📸 Capture Frame")
        self.btn_calibrate = QPushButton("⚙️ Perform Calibration + GA")
        self.btn_stop      = QPushButton("⏹ Stop")

        self.btn_start.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.btn_calibrate.setEnabled(False)

        for w in [self.btn_load_pattern, self.pattern_info, self.btn_start,
                  self.btn_capture, self.btn_calibrate, self.btn_stop]:
            if isinstance(w, QPushButton):
                w.setMinimumHeight(36)
            self.controls.addWidget(w)

        self.btn_load_pattern.clicked.connect(self.load_pattern)
        self.btn_start.clicked.connect(self.start_worker)
        self.btn_capture.clicked.connect(self.capture)
        self.btn_calibrate.clicked.connect(self.calibrate)
        self.btn_stop.clicked.connect(self.stop_worker)

    def load_pattern(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Pattern Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if path:
            self.pattern_path = path
            self.pattern_info.setText(f"Pattern: {path.split('/')[-1]}")
            self.pattern_info.setStyleSheet("color: green;")
            self.btn_start.setEnabled(True)
            self.log_message(f"Pattern selected: {path}")

    def start_worker(self):
        self.worker = OfflinePatternWorker(self.pattern_path)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.log_signal.connect(self.log_message)
        self.worker.status_signal.connect(lambda m: self.status_label.setText(f"Status: {m}"))
        self.worker.calibration_done.connect(
            lambda rms: self.log_message(
                f"✅ Calibration completed! RMS = {rms:.4f} — launching GA..."
            )
        )
        self.worker.ga_ready.connect(self._launch_ga)
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_capture.setEnabled(True)
        self.btn_calibrate.setEnabled(True)
        self.status_label.setText("Status: Collecting frames")

    def capture(self):
        if self.worker:
            self.worker.capture_frame()

    def calibrate(self):
        if self.worker:
            self.worker.trigger_calibrate()

    def stop_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
            self.btn_start.setEnabled(True)


# ---------------------------------------------------------------------------
# MODE 3: Offline Calibration
# ---------------------------------------------------------------------------
class FullOfflineWorker(QThread):
    log_signal       = pyqtSignal(str)
    status_signal    = pyqtSignal(str)
    result_frame     = pyqtSignal(QPixmap)
    # NEW: emits list of (QPixmap, bool, filename) once all images are processed
    all_frames_ready = pyqtSignal(list)
    calibration_done = pyqtSignal(float)
    ga_ready         = pyqtSignal(list, list, int, int)

    def __init__(self, pattern_path, image_paths):
        super().__init__()
        self.pattern_path = pattern_path
        self.image_paths  = image_paths
        self.p_manager    = PatternManager()
        self.calibrator   = CameraCalibrator()
        self.img_w = 0
        self.img_h = 0

    def run(self):
        pattern_img = cv2.imread(self.pattern_path)
        if pattern_img is None:
            self.log_signal.emit("Error: Unable to load the pattern.")
            return

        ph, pw = pattern_img.shape[:2]
        if max(ph, pw) > TARGET_SIZE:
            factor = TARGET_SIZE / max(ph, pw)
            pattern_img = cv2.resize(pattern_img,
                                    (int(pw * factor), int(ph * factor)),
                                    interpolation=cv2.INTER_AREA)
            ph, pw = pattern_img.shape[:2]
            self.log_signal.emit(f"Pattern resized to {pw}x{ph} for descriptor matching.")

        out = np.zeros_like(pattern_img)
        if not self.p_manager.pattern.create(pattern_img, (pw, ph), out):
            self.log_signal.emit("Error: Pattern creation failed.")
            return
        self.p_manager.set_pattern_img(pattern_img)
        self.log_signal.emit("Pattern loaded successfully.")

        all_results = []

        for i, path in enumerate(self.image_paths):
            filename = path.split('/')[-1]
            img = cv2.imread(path)
            if img is None:
                self.log_signal.emit(f"[SKIP] Error reading: {filename}")
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, f"Error: {filename}", (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                all_results.append((cv2_to_qpixmap(blank), False, filename))
                continue

            self.img_h, self.img_w = img.shape[:2]
            res, matched_feat, pattern_pts, out_frame, H, corners = \
                self.p_manager.find_pattern(img)

            if res and len(matched_feat) > 3:
                self.calibrator.add_points(pattern_pts, matched_feat)
                n_matches = len(matched_feat)
                self.log_signal.emit(
                    f"[{i+1}/{len(self.image_paths)}] ✅ {filename} — {n_matches} matches"
                )
                display = self.p_manager.build_match_display(img, matched_feat)  # ← CAMBIATO
                cv2.putText(display, f"✅ {n_matches} matches  |  {filename}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)
                pix = cv2_to_qpixmap(display)
                all_results.append((pix, True, filename))
            else:
                self.log_signal.emit(
                    f"[{i+1}/{len(self.image_paths)}] {filename} — pattern not found"
                )
                display = img.copy()
                cv2.putText(display, "Pattern NOT found", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
                cv2.putText(display, filename, (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                pix = cv2_to_qpixmap(display)
                all_results.append((pix, False, filename))

            self.result_frame.emit(all_results[-1][0])

        self.all_frames_ready.emit(all_results)

        n_valid = len(self.calibrator.obj_points)
        self.status_signal.emit(f"Analysis completed. Valid frames: {n_valid}")

        if self.calibrator.can_calibrate():
            rms = self.calibrator.calibrate(self.img_w, self.img_h)
            if rms:
                self.calibration_done.emit(rms)
                self.ga_ready.emit(
                    self.calibrator.obj_points,
                    self.calibrator.matched_points,
                    self.img_w, self.img_h
                )
        else:
            self.log_signal.emit("❌ Not enough valid frames for calibration.")

class FullOfflineCalibrationWidget(BaseCalibrationWidget):
    def __init__(self, out_dir="out"):
        super().__init__(
            "Offline Calibration",
            "Load the pattern and a set of calibration images from disk. No camera required."
            " GA optimization starts automatically after calibration.",
            out_dir=out_dir
        )
        self.worker       = None
        self.pattern_path = None
        self.image_paths  = []
        self._all_frames  = []   # list of (QPixmap, bool, filename)
        self._current_idx = 0

        # --- Load buttons ---
        self.btn_load_pattern = QPushButton("📂 Load Pattern")
        self.pattern_info     = QLabel("No pattern loaded")
        self.pattern_info.setStyleSheet("color: gray; font-style: italic;")

        self.btn_load_images = QPushButton("🗂️ Load Calibration Images")
        self.images_info     = QLabel("No images loaded")
        self.images_info.setStyleSheet("color: gray; font-style: italic;")

        self.btn_run = QPushButton("⚙️ Run Calibration + GA")
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold;"
        )

        for w in [self.btn_load_pattern, self.pattern_info,
                  self.btn_load_images, self.images_info,
                  self.btn_run]:
            if isinstance(w, QPushButton):
                w.setMinimumHeight(36)
            self.controls.addWidget(w)

        # --- Navigation bar (hidden until results are ready) ---
        self.nav_widget = QWidget()
        nav_layout = QHBoxLayout(self.nav_widget)
        nav_layout.setContentsMargins(0, 6, 0, 2)

        self.btn_prev = QPushButton("◀")
        self.nav_label = QLabel("—")
        self.nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.nav_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.nav_status_icon = QLabel("")
        self.nav_status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.nav_status_icon.setStyleSheet("font-size: 16px;")
        self.btn_next = QPushButton("▶")

        for b in [self.btn_prev, self.btn_next]:
            b.setFixedWidth(44)
            b.setMinimumHeight(34)

        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.nav_label, 1)
        nav_layout.addWidget(self.nav_status_icon)
        nav_layout.addWidget(self.btn_next)

        self.nav_filename_label = QLabel("")
        self.nav_filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.nav_filename_label.setStyleSheet(
            "color: #aaaaaa; font-size: 10px; font-style: italic;"
        )
        self.nav_filename_label.setWordWrap(True)

        self.nav_widget.hide()
        self.controls.addWidget(self.nav_widget)
        self.controls.addWidget(self.nav_filename_label)

        # Connections
        self.btn_load_pattern.clicked.connect(self.load_pattern)
        self.btn_load_images.clicked.connect(self.load_images)
        self.btn_run.clicked.connect(self.run_calibration)
        self.btn_prev.clicked.connect(self._prev_frame)
        self.btn_next.clicked.connect(self._next_frame)

    def load_pattern(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select pattern image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if path:
            self.pattern_path = path
            self.pattern_info.setText(f"Pattern: {path.split('/')[-1]}")
            self.pattern_info.setStyleSheet("color: green;")
            self.log_message(f"Pattern: {path}")
            self._check_ready()

    def load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select calibration images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if paths:
            self.image_paths = paths
            self.images_info.setText(f"{len(paths)} images loaded")
            self.images_info.setStyleSheet("color: green;")
            self.log_message(f"Loaded {len(paths)} images.")
            self._check_ready()

    def _check_ready(self):
        if self.pattern_path and self.image_paths:
            self.btn_run.setEnabled(True)

    def run_calibration(self):
        self.btn_run.setEnabled(False)
        self._all_frames  = []
        self._current_idx = 0
        self.nav_widget.hide()
        self.nav_filename_label.setText("")

        self.worker = FullOfflineWorker(self.pattern_path, self.image_paths)
        self.worker.log_signal.connect(self.log_message)
        self.worker.status_signal.connect(
            lambda m: self.status_label.setText(f"Status: {m}")
        )
        # Live preview during processing
        self.worker.result_frame.connect(self.update_frame)
        # Full navigation gallery once all images are done
        self.worker.all_frames_ready.connect(self._on_all_frames_ready)
        self.worker.calibration_done.connect(
            lambda rms: (
                self.log_message(
                    f"✅ Calibration completed! RMS = {rms:.4f} — launching GA..."
                ),
                self.btn_run.setEnabled(True),
            )
        )
        self.worker.ga_ready.connect(self._launch_ga)
        self.worker.start()
        self.status_label.setText("Status: Processing images…")

    def _on_all_frames_ready(self, results):
        """Called once the worker has processed every image."""
        self._all_frames  = results
        self._current_idx = 0
        if results:
            self.nav_widget.show()
            self._show_current_frame()

    def _show_current_frame(self):
        if not self._all_frames:
            return
        pix, found, filename = self._all_frames[self._current_idx]
        total = len(self._all_frames)

        self.nav_label.setText(f"{self._current_idx + 1} / {total}")
        self.nav_status_icon.setText("✅" if found else "❌")
        self.nav_filename_label.setText(filename)
        self.update_frame(pix)

        self.btn_prev.setEnabled(self._current_idx > 0)
        self.btn_next.setEnabled(self._current_idx < total - 1)

    def _prev_frame(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self._show_current_frame()

    def _next_frame(self):
        if self._current_idx < len(self._all_frames) - 1:
            self._current_idx += 1
            self._show_current_frame()
