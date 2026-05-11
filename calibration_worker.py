import cv2
import os
import datetime
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QTextEdit, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from pattern_manager import PatternManager
from camera_calibrator import CameraCalibrator
from utils import ChessboardValidator
from ga_worker import GAWorker

TARGET_SIZE = 300


def cv2_to_qpixmap(frame):
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Shared base widget
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
        self.video_label.setStyleSheet(
            "background: #1a1a1a; color: white; border-radius: 6px;"
        )
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
        self._ga_worker = GAWorker(
            obj_points, matched_points, img_w, img_h, self.out_dir
        )
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
        self._smooth_rvec  = None
        self._smooth_tvec  = None

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(0)

        SMOOTH_ALPHA      = 0.35
        MIN_TRACKING_FEAT = 20

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
            self._frame = frame.copy()

            if self._phase == "select_roi":
                canvas = self.p_manager.draw_roi(frame.copy())
                self.frame_ready.emit(cv2_to_qpixmap(canvas))

            elif self._phase == "collect":
                res, matched_feat, pattern_pts, out, H, corners = \
                    self.p_manager.find_pattern(frame)
                display = self.p_manager.build_match_display(
                    frame, matched_feat, pattern_found=res
                )
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
                res, matched_feat, pattern_pts, out, H, corners = \
                    self.p_manager.find_pattern(frame)
                if (res and self.calibrator.K is not None
                        and len(matched_feat) >= MIN_TRACKING_FEAT):
                    ok, rvec, tvec = self.p_manager.pattern.findRt(
                        pattern_pts, matched_feat,
                        self.calibrator.K,
                        self.calibrator.dist_coeff, None, None
                    )
                    if ok:
                        if self._smooth_rvec is None:
                            self._smooth_rvec = rvec.copy()
                            self._smooth_tvec = tvec.copy()
                        else:
                            self._smooth_rvec = (SMOOTH_ALPHA * rvec
                                                 + (1 - SMOOTH_ALPHA) * self._smooth_rvec)
                            self._smooth_tvec = (SMOOTH_ALPHA * tvec
                                                 + (1 - SMOOTH_ALPHA) * self._smooth_tvec)
                        h_frame, w_frame = frame.shape[:2]
                        axis_len = int(min(h_frame, w_frame) * 0.15)
                        frame = self.p_manager.pattern.drawOrientation(
                            frame,
                            self._smooth_tvec, self._smooth_rvec,
                            self.calibrator.K,
                            self.calibrator.dist_coeff, axis_len, 3
                        )
                else:
                    self._smooth_rvec = None
                    self._smooth_tvec = None
                self.frame_ready.emit(cv2_to_qpixmap(frame))

        cap.release()

    def confirm_roi(self):
        success, out = self.p_manager.create_pattern(self._frame)
        if success:
            self._phase = "collect"
            self.log_signal.emit(
                "Pattern created. Click 'Capture Frame' to collect data."
            )
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

        self.controls.addStretch()

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
        self.worker.status_signal.connect(
            lambda m: self.status_label.setText(f"Status: {m}")
        )
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
        self._smooth_rvec  = None
        self._smooth_tvec  = None
        self.img_w         = 640
        self.img_h         = 480

    def run(self):
        cap = cv2.VideoCapture(0)
        ret, probe = cap.read()
        if not ret:
            cap.release()
            self.log_signal.emit("Error: Cannot open camera.")
            return
        self.img_h, self.img_w = probe.shape[:2]

        pattern_img = cv2.imread(self.pattern_path)
        if pattern_img is None:
            cap.release()
            self.log_signal.emit("Error: Unable to load the pattern.")
            return

        ph, pw = pattern_img.shape[:2]
        max_side = min(self.img_w, self.img_h)
        if max(ph, pw) > max_side:
            factor = max_side / max(ph, pw)
            pattern_img = cv2.resize(
                pattern_img,
                (int(pw * factor), int(ph * factor)),
                interpolation=cv2.INTER_AREA
            )
            ph, pw = pattern_img.shape[:2]
            self.log_signal.emit(
                f"Pattern resized to {pw}x{ph} to match camera resolution."
            )

        out = np.zeros_like(pattern_img)
        if not self.p_manager.pattern.create(pattern_img, (pw, ph), out):
            cap.release()
            self.log_signal.emit("Error: Pattern creation failed.")
            return
        self.p_manager.set_pattern_img(pattern_img)
        self.log_signal.emit("Pattern loaded successfully.")
        self._running = True

        SMOOTH_ALPHA      = 0.35
        MIN_TRACKING_FEAT = 20
        first = True

        while self._running:
            if first:
                frame = probe
                first = False
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            self._frame = frame.copy()
            self.img_h, self.img_w = frame.shape[:2]

            if self._phase == "collect":
                res, matched_feat, pattern_pts, out_frame, H, corners = \
                    self.p_manager.find_pattern(frame)
                display = self.p_manager.build_match_display(
                    frame, matched_feat, pattern_found=res
                )
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
                res, matched_feat, pattern_pts, out, H, corners = \
                    self.p_manager.find_pattern(frame)
                if (res and self.calibrator.K is not None
                        and len(matched_feat) >= MIN_TRACKING_FEAT):
                    ok, rvec, tvec = self.p_manager.pattern.findRt(
                        pattern_pts, matched_feat,
                        self.calibrator.K,
                        self.calibrator.dist_coeff, None, None
                    )
                    if ok:
                        if self._smooth_rvec is None:
                            self._smooth_rvec = rvec.copy()
                            self._smooth_tvec = tvec.copy()
                        else:
                            self._smooth_rvec = (SMOOTH_ALPHA * rvec
                                                 + (1 - SMOOTH_ALPHA) * self._smooth_rvec)
                            self._smooth_tvec = (SMOOTH_ALPHA * tvec
                                                 + (1 - SMOOTH_ALPHA) * self._smooth_tvec)
                        h_frame, w_frame = frame.shape[:2]
                        axis_len = int(min(h_frame, w_frame) * 0.15)
                        frame = self.p_manager.pattern.drawOrientation(
                            frame,
                            self._smooth_tvec, self._smooth_rvec,
                            self.calibrator.K,
                            self.calibrator.dist_coeff, axis_len, 3
                        )
                else:
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

        self.controls.addStretch()

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
        self.worker.status_signal.connect(
            lambda m: self.status_label.setText(f"Status: {m}")
        )
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
# MODE 3: Offline Calibration — image processing worker
# ---------------------------------------------------------------------------
class FullOfflineWorker(QThread):
    log_signal       = pyqtSignal(str)
    status_signal    = pyqtSignal(str)
    result_frame     = pyqtSignal(QPixmap)
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
            pattern_img = cv2.resize(
                pattern_img,
                (int(pw * factor), int(ph * factor)),
                interpolation=cv2.INTER_AREA
            )
            ph, pw = pattern_img.shape[:2]
            self.log_signal.emit(
                f"Pattern resized to {pw}x{ph} for offline descriptor matching."
            )

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
                all_results.append((cv2_to_qpixmap(blank), False, filename))
                continue

            self.img_h, self.img_w = img.shape[:2]
            res, matched_feat, pattern_pts, out_frame, H, corners = \
                self.p_manager.find_pattern(img)

            if res and len(matched_feat) > 3:
                self.calibrator.add_points(pattern_pts, matched_feat)
                display = self.p_manager.build_match_display(
                    img, matched_feat, pattern_found=True
                )
                pix = cv2_to_qpixmap(display)
                all_results.append((pix, True, filename))
                self.log_signal.emit(
                    f"[{i+1}/{len(self.image_paths)}] ✅ {filename}"
                    f" — {len(matched_feat)} matches"
                )
            else:
                display = self.p_manager.build_match_display(
                    img, None, pattern_found=False
                )
                pix = cv2_to_qpixmap(display)
                all_results.append((pix, False, filename))
                self.log_signal.emit(
                    f"[{i+1}/{len(self.image_paths)}] ❌ {filename}"
                    f" — pattern not found"
                )

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


# ---------------------------------------------------------------------------
# MODE 3: Live camera test worker
# ---------------------------------------------------------------------------
class OfflineCameraTestWorker(QThread):
    frame_ready    = pyqtSignal(QPixmap)
    snapshot_ready = pyqtSignal(str, object)
    debug_signal   = pyqtSignal(str)

    def __init__(self, calibrator: CameraCalibrator, pattern_path: str):
        super().__init__()
        self.calibrator        = calibrator
        self.pattern_path      = pattern_path
        self._running          = False
        self._request_snapshot = False
        self._last_raw         = None
        self._smooth_rvec      = None
        self._smooth_tvec      = None
        self.SMOOTH_ALPHA      = 0.35
        self.MIN_TRACKING_FEAT = 20

    def save_snapshot(self):
        self._request_snapshot = True

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(0)

        ret, probe = cap.read()
        if not ret:
            self.debug_signal.emit("❌ DEBUG: Cannot open camera.")
            cap.release()
            return
        cam_h, cam_w = probe.shape[:2]
        self.debug_signal.emit(f"📷 DEBUG: Camera resolution: {cam_w}x{cam_h}")

        K          = self.calibrator.K
        dist_coeff = self.calibrator.dist_coeff
        if K is None:
            self.debug_signal.emit("❌ DEBUG: calibrator.K is None — no calibration loaded!")
            cap.release()
            return
        self.debug_signal.emit(
            f"✅ DEBUG: K loaded — "
            f"fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  "
            f"cx={K[0,2]:.2f}  cy={K[1,2]:.2f}"
        )
        self.debug_signal.emit(
            f"✅ DEBUG: dist_coeff = {dist_coeff.ravel().tolist()}"
        )

        # Reload pattern at camera resolution (NOT TARGET_SIZE)
        p_manager   = PatternManager()
        pattern_img = cv2.imread(self.pattern_path)
        if pattern_img is None:
            self.debug_signal.emit(
                f"❌ DEBUG: Cannot read pattern from: {self.pattern_path}"
            )
            cap.release()
            return

        ph_orig, pw_orig = pattern_img.shape[:2]
        self.debug_signal.emit(
            f"🖼️  DEBUG: Pattern loaded from disk: {pw_orig}x{ph_orig}"
        )

        max_side = min(cam_w, cam_h)
        if max(ph_orig, pw_orig) > max_side:
            factor      = max_side / max(ph_orig, pw_orig)
            pattern_img = cv2.resize(
                pattern_img,
                (int(pw_orig * factor), int(ph_orig * factor)),
                interpolation=cv2.INTER_AREA
            )
            ph, pw = pattern_img.shape[:2]
            self.debug_signal.emit(
                f"🔄 DEBUG: Pattern resized to {pw}x{ph} (factor={factor:.3f})"
            )
        else:
            ph, pw = ph_orig, pw_orig
            self.debug_signal.emit(
                f"ℹ️  DEBUG: Pattern kept at original size {pw}x{ph}"
            )

        out = np.zeros_like(pattern_img)
        ok_create = p_manager.pattern.create(pattern_img, (pw, ph), out)
        if not ok_create:
            self.debug_signal.emit("❌ DEBUG: p_manager.pattern.create() FAILED.")
            cap.release()
            return
        p_manager.set_pattern_img(pattern_img)
        self.debug_signal.emit("✅ DEBUG: Pattern created successfully.")

        first       = True
        frame_count = 0
        axes_drawn  = 0

        while self._running:
            if first:
                frame = probe
                first = False
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            self._last_raw = frame.copy()
            frame_count   += 1

            # Snapshot
            if self._request_snapshot:
                self._request_snapshot = False
                undistorted = cv2.undistort(
                    self._last_raw, K, dist_coeff,
                    None, self.calibrator.new_camera_matrix
                )
                os.makedirs("out", exist_ok=True)
                ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join("out", f"rectified_{ts}.png")
                cv2.imwrite(path, undistorted)
                self.snapshot_ready.emit(path, undistorted)

            # find_pattern
            res, matched_feat, pattern_pts, _, _, _ = \
                p_manager.find_pattern(frame)

            display         = frame.copy()
            log_this_frame  = (frame_count % 30 == 1)

            if log_this_frame:
                if res and matched_feat is not None:
                    self.debug_signal.emit(
                        f"🔍 frame {frame_count}: "
                        f"find_pattern=True, matched={len(matched_feat)}, "
                        f"pattern_pts shape={np.array(pattern_pts).shape}"
                    )
                else:
                    self.debug_signal.emit(
                        f"🔍 frame {frame_count}: find_pattern=False"
                    )

            if (res and matched_feat is not None
                    and len(matched_feat) >= self.MIN_TRACKING_FEAT):

                mf_arr = np.array(matched_feat)
                pp_arr = np.array(pattern_pts)

                if log_this_frame:
                    self.debug_signal.emit(
                        f"   matched_feat dtype={mf_arr.dtype}, shape={mf_arr.shape}\n"
                        f"   pattern_pts  dtype={pp_arr.dtype}, shape={pp_arr.shape}"
                    )

                try:
                    ok, rvec, tvec = p_manager.pattern.findRt(
                        pattern_pts, matched_feat,
                        K, dist_coeff, None, None
                    )
                except Exception as e:
                    if log_this_frame:
                        self.debug_signal.emit(f"❌ findRt EXCEPTION: {e}")
                    ok = False

                if log_this_frame:
                    if ok:
                        self.debug_signal.emit(
                            f"   ✅ findRt OK — "
                            f"rvec={rvec.ravel().tolist()}, "
                            f"tvec={tvec.ravel().tolist()}"
                        )
                    else:
                        self.debug_signal.emit(
                            "   ❌ findRt returned False — PnP solve failed"
                        )

                if ok:
                    if self._smooth_rvec is None:
                        self._smooth_rvec = rvec.copy()
                        self._smooth_tvec = tvec.copy()
                    else:
                        self._smooth_rvec = (
                            self.SMOOTH_ALPHA * rvec
                            + (1 - self.SMOOTH_ALPHA) * self._smooth_rvec
                        )
                        self._smooth_tvec = (
                            self.SMOOTH_ALPHA * tvec
                            + (1 - self.SMOOTH_ALPHA) * self._smooth_tvec
                        )

                    h_f, w_f = display.shape[:2]
                    axis_len = int(min(h_f, w_f) * 0.15)

                    try:
                        display = p_manager.pattern.drawOrientation(
                            display,
                            self._smooth_tvec, self._smooth_rvec,
                            K, dist_coeff, axis_len, 3
                        )
                        axes_drawn += 1
                        if log_this_frame:
                            self.debug_signal.emit(
                                f"   🎯 Axes drawn (total: {axes_drawn}), "
                                f"axis_len={axis_len}"
                            )
                    except Exception as e:
                        if log_this_frame:
                            self.debug_signal.emit(
                                f"❌ drawOrientation EXCEPTION: {e}"
                            )
            else:
                self._smooth_rvec = None
                self._smooth_tvec = None

            self.frame_ready.emit(cv2_to_qpixmap(display))

        cap.release()
        self.debug_signal.emit(
            f"🏁 Worker stopped. "
            f"Frames: {frame_count}, axes drawn: {axes_drawn}"
        )

    def stop(self):
        self._running = False
        self.wait()


# ---------------------------------------------------------------------------
# MODE 3: Offline Calibration — Widget
# ---------------------------------------------------------------------------
class FullOfflineCalibrationWidget(BaseCalibrationWidget):
    def __init__(self, out_dir="out"):
        super().__init__(
            "Offline Calibration",
            "Load the pattern and a set of calibration images from disk. No camera required."
            " GA optimization starts automatically after calibration.",
            out_dir=out_dir
        )
        self.worker       = None
        self._cam_worker  = None
        self.pattern_path = None
        self.image_paths  = []
        self._all_frames  = []
        self._current_idx = 0

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

        # Navigation bar
        self.nav_widget = QWidget()
        nav_layout = QHBoxLayout(self.nav_widget)
        nav_layout.setContentsMargins(0, 6, 0, 2)

        self.btn_prev        = QPushButton("◀")
        self.nav_label       = QLabel("—")
        self.nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.nav_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.nav_status_icon = QLabel("")
        self.nav_status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.nav_status_icon.setStyleSheet("font-size: 16px;")
        self.btn_next        = QPushButton("▶")

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

        self.btn_enable_camera = QPushButton("📷 Enable Camera (Test Axes)")
        self.btn_enable_camera.setMinimumHeight(36)
        self.btn_enable_camera.setStyleSheet(
            "background-color: #1565c0; color: white; font-weight: bold;"
        )
        self.btn_enable_camera.hide()
        self.controls.addWidget(self.btn_enable_camera)

        self.btn_save_rectified = QPushButton("📸 Save Rectified Snapshot")
        self.btn_save_rectified.setMinimumHeight(36)
        self.btn_save_rectified.setEnabled(False)
        self.btn_save_rectified.hide()
        self.controls.addWidget(self.btn_save_rectified)

        self.controls.addStretch()

        self.btn_load_pattern.clicked.connect(self.load_pattern)
        self.btn_load_images.clicked.connect(self.load_images)
        self.btn_run.clicked.connect(self.run_calibration)
        self.btn_prev.clicked.connect(self._prev_frame)
        self.btn_next.clicked.connect(self._next_frame)
        self.btn_enable_camera.clicked.connect(self._toggle_camera)
        self.btn_save_rectified.clicked.connect(self._save_rectified)

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
        if self._cam_worker and self._cam_worker.isRunning():
            self._cam_worker.stop()
            self._cam_worker = None
            self.btn_enable_camera.setText("📷 Enable Camera (Test Axes)")

        self.btn_run.setEnabled(False)
        self.btn_enable_camera.hide()
        self.btn_save_rectified.hide()
        self._all_frames  = []
        self._current_idx = 0
        self.nav_widget.hide()
        self.nav_filename_label.setText("")

        self.worker = FullOfflineWorker(self.pattern_path, self.image_paths)
        self.worker.log_signal.connect(self.log_message)
        self.worker.status_signal.connect(
            lambda m: self.status_label.setText(f"Status: {m}")
        )
        self.worker.result_frame.connect(self.update_frame)
        self.worker.all_frames_ready.connect(self._on_all_frames_ready)
        self.worker.calibration_done.connect(self._on_calibration_done)
        self.worker.ga_ready.connect(self._launch_ga)
        self.worker.start()
        self.status_label.setText("Status: Processing images…")

    def _on_calibration_done(self, rms):
        self.log_message(
            f"✅ Calibration completed! RMS = {rms:.4f} — launching GA..."
        )
        self.btn_run.setEnabled(True)
        self.btn_enable_camera.show()
        self.btn_save_rectified.show()
        self.btn_save_rectified.setEnabled(False)

    def _on_all_frames_ready(self, results):
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

    def _toggle_camera(self):
        if self._cam_worker and self._cam_worker.isRunning():
            self._cam_worker.stop()
            self._cam_worker = None
            self.btn_enable_camera.setText("📷 Enable Camera (Test Axes)")
            self.btn_save_rectified.setEnabled(False)
            if self._all_frames:
                self.nav_widget.show()
                self._show_current_frame()
        else:
            if self.worker is None or self.worker.calibrator.K is None:
                self.log_message("⚠️ No calibration available yet.")
                return
            self.nav_widget.hide()
            self._cam_worker = OfflineCameraTestWorker(
                self.worker.calibrator,
                self.pattern_path
            )
            self._cam_worker.frame_ready.connect(self.update_frame)
            self._cam_worker.snapshot_ready.connect(self._on_snapshot_ready)
            # Debug log va nel pannello GA Log per visibilità immediata
            self._cam_worker.debug_signal.connect(self.ga_log.append)
            self._cam_worker.start()
            self.btn_enable_camera.setText("⏹ Stop Camera")
            self.btn_save_rectified.setEnabled(True)
            self.log_message(
                "Camera started — check GA Optimizer Log for debug output."
            )

    def _save_rectified(self):
        if self._cam_worker is None or not self._cam_worker.isRunning():
            self.log_message("⚠️ Camera is not running.")
            return
        self._cam_worker.save_snapshot()

    def _on_snapshot_ready(self, path: str, img):
        self.log_message(f"✅ Rectified snapshot saved: {path}")