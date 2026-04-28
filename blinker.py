"""
Blinker - Eye Blink Detection & Reminder
=========================================
Uses MediaPipe FaceLandmarker (Tasks API) to detect eye blinks via the
Eye Aspect Ratio (EAR) method. Tracks blink frequency and alerts with a
sound if no blink is detected for 10 seconds.

Controls:
  Q / ESC  - Quit
  R        - Reset blink counter
  +/-      - Adjust EAR sensitivity threshold
"""

import os
import sys

# Suppress MediaPipe telemetry & logging — must be set before importing mediapipe
os.environ['MEDIAPIPE_DISABLE_TELEMETRY'] = '1'

import time
import math
import threading
import winsound

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

# ─── Configuration ───────────────────────────────────────────────────────────

EAR_THRESHOLD = 0.21        # Below this EAR value, eyes are considered closed
EAR_CONSEC_FRAMES = 2       # Minimum consecutive frames below threshold for a blink
ALERT_INTERVAL_SEC = 6.0    # Seconds without blink before alarm sounds
ALERT_COOLDOWN_SEC = 3.0    # Minimum seconds between alert sounds
CAMERA_INDEX = 0            # Webcam index (0 = default)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# MediaPipe Face Mesh landmark indices for each eye (same as legacy API)
# p1=lateral corner, p2/p3=upper lid, p4=medial corner, p5/p6=lower lid
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Extended contour indices for drawing
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# ─── Color Palette (BGR) ─────────────────────────────────────────────────────

COL_PANEL     = (45, 45, 48)
COL_ACCENT    = (255, 168, 50)    # Warm amber
COL_GREEN     = (100, 220, 120)
COL_RED       = (80, 80, 255)
COL_YELLOW    = (60, 220, 255)
COL_WHITE     = (240, 240, 240)
COL_GRAY      = (160, 160, 160)
COL_DARK_GRAY = (80, 80, 80)
COL_EYE       = (255, 200, 80)


# ─── Utility Functions ───────────────────────────────────────────────────────

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    Compute the Eye Aspect Ratio (EAR) for one eye.
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    A low EAR indicates the eye is closed.
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * w, lm.y * h))

    v1 = math.dist(pts[1], pts[5])
    v2 = math.dist(pts[2], pts[4])
    h1 = math.dist(pts[0], pts[3])

    if h1 == 0:
        return 0.3
    return (v1 + v2) / (2.0 * h1)


def play_alert_sound():
    """Play an alert beep in a separate thread (non-blocking)."""
    def _beep():
        try:
            winsound.Beep(800, 150)
            winsound.Beep(1000, 150)
            winsound.Beep(1200, 300)
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True).start()


def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=0.75):
    """Draw a semi-transparent rounded rectangle."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2

    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_progress_arc(img, center, radius, progress, color, thickness=4):
    """Draw a circular progress arc (0.0 to 1.0)."""
    angle = int(progress * 360)
    if angle > 0:
        cv2.ellipse(img, center, (radius, radius), -90, 0, angle, color, thickness, cv2.LINE_AA)


def put_text(img, text, pos, scale=0.55, color=COL_WHITE, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Draw anti-aliased text with a subtle shadow."""
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# ─── HUD Drawing ─────────────────────────────────────────────────────────────

def draw_eye_contours(frame, landmarks, w, h):
    """Draw eye contours using MediaPipe landmarks."""
    for contour in [LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR]:
        pts = []
        for idx in contour:
            lm = landmarks[idx]
            pts.append([int(lm.x * w), int(lm.y * h)])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(frame, [pts], True, COL_EYE, 1, cv2.LINE_AA)


def draw_hud(frame, blink_count, ear_value, time_since_blink, ear_threshold,
             blinks_per_minute, is_alert, face_detected):
    """Draw the heads-up display overlay on the video frame."""
    h, w = frame.shape[:2]

    # ── Top-left: App title ──
    draw_rounded_rect(frame, (10, 10), (220, 50), COL_PANEL, radius=10, alpha=0.8)
    put_text(frame, "BLINKER", (24, 38), scale=0.7, color=COL_ACCENT, thickness=2)
    put_text(frame, "v1.0", (145, 38), scale=0.4, color=COL_GRAY)

    if not face_detected:
        draw_rounded_rect(frame, (w // 2 - 140, h // 2 - 25), (w // 2 + 140, h // 2 + 25),
                          COL_RED, radius=10, alpha=0.7)
        put_text(frame, "NO FACE DETECTED", (w // 2 - 115, h // 2 + 8),
                 scale=0.7, color=COL_WHITE, thickness=2)
        return

    # ── Left panel: Stats ──
    panel_y = 65
    panel_h = 190
    draw_rounded_rect(frame, (10, panel_y), (220, panel_y + panel_h), COL_PANEL, radius=10, alpha=0.8)

    put_text(frame, "BLINKS", (24, panel_y + 28), scale=0.4, color=COL_GRAY)
    put_text(frame, str(blink_count), (24, panel_y + 60), scale=0.9, color=COL_GREEN, thickness=2)

    put_text(frame, "BLINKS/MIN", (24, panel_y + 88), scale=0.4, color=COL_GRAY)
    bpm_color = COL_GREEN if blinks_per_minute >= 12 else (COL_YELLOW if blinks_per_minute >= 6 else COL_RED)
    put_text(frame, f"{blinks_per_minute:.1f}", (24, panel_y + 118), scale=0.7, color=bpm_color, thickness=2)

    put_text(frame, f"EAR", (24, panel_y + 146), scale=0.4, color=COL_GRAY)
    ear_color = COL_RED if ear_value < ear_threshold else COL_GREEN
    put_text(frame, f"{ear_value:.3f}", (24, panel_y + 172), scale=0.6, color=ear_color, thickness=1)
    put_text(frame, f"THR: {ear_threshold:.2f}", (130, panel_y + 172), scale=0.4, color=COL_DARK_GRAY)

    # ── Right panel: Timer ──
    timer_panel_w = 160
    timer_x = w - timer_panel_w - 10
    draw_rounded_rect(frame, (timer_x, 10), (w - 10, 140), COL_PANEL, radius=10, alpha=0.8)

    circle_center = (timer_x + timer_panel_w // 2, 75)
    circle_radius = 40
    progress = min(time_since_blink / ALERT_INTERVAL_SEC, 1.0)

    cv2.circle(frame, circle_center, circle_radius, COL_DARK_GRAY, 2, cv2.LINE_AA)

    if progress < 0.5:
        arc_color = COL_GREEN
    elif progress < 0.8:
        arc_color = COL_YELLOW
    else:
        arc_color = COL_RED
    draw_progress_arc(frame, circle_center, circle_radius, progress, arc_color, thickness=4)

    timer_text = f"{time_since_blink:.1f}s"
    text_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = circle_center[0] - text_size[0] // 2
    text_y = circle_center[1] + text_size[1] // 2
    put_text(frame, timer_text, (text_x, text_y), scale=0.5, color=arc_color, thickness=1)
    put_text(frame, "SINCE BLINK", (timer_x + 25, 128), scale=0.4, color=COL_GRAY)

    # ── Alert banner ──
    if is_alert:
        pulse = int(127 + 128 * math.sin(time.time() * 6))
        alert_color = (60, 60, pulse)
        draw_rounded_rect(frame, (w // 2 - 160, 10), (w // 2 + 160, 55),
                          alert_color, radius=10, alpha=0.85)
        put_text(frame, "! BLINK NOW !", (w // 2 - 80, 40),
                 scale=0.8, color=COL_WHITE, thickness=2)

    # ── Bottom bar: Controls ──
    bar_y = h - 40
    draw_rounded_rect(frame, (10, bar_y), (w - 10, h - 8), COL_PANEL, radius=8, alpha=0.7)
    controls = "Q: Quit  |  R: Reset  |  +/-: Sensitivity"
    put_text(frame, controls, (20, h - 18), scale=0.4, color=COL_DARK_GRAY)


# ─── Main Application ────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  BLINKER - Eye Blink Detection & Reminder")
    print("=" * 50)
    print(f"  Alert after:    {ALERT_INTERVAL_SEC}s without blinking")
    print(f"  EAR threshold:  {EAR_THRESHOLD}")
    print(f"  Camera index:   {CAMERA_INDEX}")
    print("=" * 50)
    print()

    # Check model file exists
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("  Download it with:")
        print('  Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "face_landmarker.task"')
        sys.exit(1)

    # Initialize MediaPipe FaceLandmarker (Tasks API, VIDEO mode)
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # Open webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check your camera connection.")
        landmarker.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera resolution: {actual_w}x{actual_h}")

    # State variables
    blink_count = 0
    frame_counter = 0
    last_blink_time = time.time()
    last_alert_time = 0.0
    ear_threshold = EAR_THRESHOLD
    ear_value = 0.3
    blink_timestamps = []
    start_time = time.time()
    frame_timestamp_ms = 0

    cv2.namedWindow("Blinker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Blinker", min(actual_w, 1280), min(actual_h, 720))

    print("[INFO] Running... Press Q or ESC to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Convert frame for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Ensure monotonically increasing timestamp
        frame_timestamp_ms += 33  # ~30 fps

        # Detect face landmarks
        try:
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        except Exception as e:
            # Timestamp issue: just increment more
            frame_timestamp_ms += 100
            try:
                result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            except Exception:
                result = None

        current_time = time.time()
        time_since_blink = current_time - last_blink_time
        face_detected = False
        is_alert = False

        if result and result.face_landmarks:
            face_detected = True
            landmarks = result.face_landmarks[0]  # First face

            # Draw eye contours
            draw_eye_contours(frame, landmarks, w, h)

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear_value = (left_ear + right_ear) / 2.0

            # Blink detection
            if ear_value < ear_threshold:
                frame_counter += 1
            else:
                if frame_counter >= EAR_CONSEC_FRAMES:
                    blink_count += 1
                    last_blink_time = current_time
                    time_since_blink = 0.0
                    blink_timestamps.append(current_time)
                    print(f"  [BLINK #{blink_count}] EAR={ear_value:.3f}")
                frame_counter = 0

        # Calculate blinks per minute (rolling 60s window)
        blink_timestamps = [t for t in blink_timestamps if current_time - t <= 60.0]
        elapsed = current_time - start_time
        if elapsed > 5:
            blinks_per_minute = len(blink_timestamps) * (60.0 / min(elapsed, 60.0))
        else:
            blinks_per_minute = 0.0

        # Alert logic
        if time_since_blink >= ALERT_INTERVAL_SEC and face_detected:
            is_alert = True
            if current_time - last_alert_time >= ALERT_COOLDOWN_SEC:
                play_alert_sound()
                last_alert_time = current_time
                print(f"  [ALERT] No blink for {time_since_blink:.1f}s!")

        # Draw HUD
        draw_hud(frame, blink_count, ear_value, time_since_blink, ear_threshold,
                 blinks_per_minute, is_alert, face_detected)

        cv2.imshow("Blinker", frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('r'), ord('R')):
            blink_count = 0
            blink_timestamps.clear()
            last_blink_time = current_time
            start_time = current_time
            print("  [RESET] Counters cleared.")
        elif key in (ord('+'), ord('=')):
            ear_threshold = min(ear_threshold + 0.01, 0.35)
            print(f"  [CONFIG] EAR threshold: {ear_threshold:.2f}")
        elif key in (ord('-'), ord('_')):
            ear_threshold = max(ear_threshold - 0.01, 0.10)
            print(f"  [CONFIG] EAR threshold: {ear_threshold:.2f}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    # Final stats
    total_time = time.time() - start_time
    print(f"\n{'=' * 50}")
    print(f"  Session Summary")
    print(f"{'=' * 50}")
    print(f"  Duration:       {total_time:.0f}s")
    print(f"  Total blinks:   {blink_count}")
    if total_time > 0:
        print(f"  Avg blinks/min: {blink_count / (total_time / 60):.1f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
