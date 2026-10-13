"""
Vietnamese License Plate Detector — Realtime
=============================================
Giao diện chuyên nghiệp 2 khung:
  • Khung trái:  Camera realtime + khung bao biển số xe
  • Khung phải:  Biển số phát hiện dạng chữ/số + lịch sử + ảnh crop

Hỗ trợ biển số Việt Nam 1 dòng & 2 dòng:
  - 2 dòng: 29A / 123.45  →  "29A-123.45"
  - 1 dòng: 29A-12345

Lưu vào folder database/:
  - plate_<timestamp>.jpg   — Ảnh crop biển số
  - plates_log.csv          — Log tổng hợp
"""

import cv2
import easyocr
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import re
import csv
from datetime import datetime
import threading
import time
from collections import deque


# ══════════════════════════ CONFIG ══════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(SCRIPT_DIR, "database")
LOG_CSV = os.path.join(DATABASE_DIR, "plates_log.csv")

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
OCR_INTERVAL = 1.0         # seconds between OCR scans
CONFIDENCE_THRESHOLD = 0.3 # min EasyOCR confidence per text line
CROP_PADDING = 15          # px padding around detected plate
GROUP_Y_THRESHOLD = 80     # max vertical gap (px) to group lines into one plate
GROUP_X_THRESHOLD = 100    # max horizontal gap (px) to group lines
DUPLICATE_COOLDOWN = 5     # seconds before same plate can be re-saved

# Vietnamese plate regex patterns
# Matches: 29A-12345, 51F-123.45, 30K1234, 29A 123 45, etc.
VN_PLATE_PATTERNS = [
    re.compile(r'^\d{2}[A-Z]\d?\s*[-.]?\s*\d{3,5}[.\s]?\d{0,3}$', re.IGNORECASE),
    re.compile(r'^\d{2}[A-Z]\d?$', re.IGNORECASE),           # first line only
    re.compile(r'^\d{3,5}[.\s]?\d{0,3}$'),                    # second line only
]
# ═══════════════════════════════════════════════════════════


def ensure_database():
    os.makedirs(DATABASE_DIR, exist_ok=True)
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Timestamp", "Plate_Text", "Confidence", "Image_File"])


def save_plate(plate_text: str, confidence: float, plate_img: np.ndarray) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"plate_{ts}.jpg"
    filepath = os.path.join(DATABASE_DIR, filename)
    cv2.imwrite(filepath, plate_img)
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            plate_text, f"{confidence:.2f}", filename,
        ])
    return filepath


def normalize_plate(text: str) -> str:
    """Normalize detected text to Vietnamese plate format."""
    t = text.strip().upper()
    # Common OCR substitutions
    t = t.replace('O', '0').replace('I', '1').replace('S', '5')
    t = t.replace('B', '8').replace('G', '6').replace('Q', '0')
    # But letters in position 2 (or 3) should stay as letters
    # Re-substitute back for the letter position
    chars = list(t.replace(' ', '').replace('-', '').replace('.', ''))
    if len(chars) >= 3:
        # Position index 2 is typically the series letter (A-Z)
        pos = 2
        if chars[pos].isdigit():
            reverse_map = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G'}
            if chars[pos] in reverse_map:
                chars[pos] = reverse_map[chars[pos]]
    result = ''.join(chars)
    # Format nicely: XX[L]-XXXXX or XX[L]-XXX.XX
    if len(result) >= 7:
        prefix = result[:3]  # e.g. 29A
        numbers = result[3:]
        if len(numbers) >= 5:
            numbers = numbers[:3] + '.' + numbers[3:]
        return f"{prefix}-{numbers}"
    return t


def is_vn_plate_part(text: str) -> bool:
    """Check if text looks like part of a Vietnamese plate."""
    cleaned = text.strip().upper()
    cleaned = re.sub(r'[^A-Z0-9.\- ]', '', cleaned)
    if len(cleaned) < 2:
        return False
    return any(p.match(cleaned) for p in VN_PLATE_PATTERNS)


def group_detections(results):
    """
    Group nearby EasyOCR detections into plate candidates.
    Vietnamese plates often have 2 lines, so we merge vertically-close boxes.
    """
    if not results:
        return []

    # Sort by vertical center position
    items = []
    for bbox, text, conf in results:
        pts = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        cx, cy = x + w // 2, y + h // 2
        items.append({
            'bbox': bbox, 'text': text, 'conf': conf,
            'rect': (x, y, w, h), 'center': (cx, cy),
        })

    items.sort(key=lambda it: it['center'][1])

    # Greedy grouping: merge items whose centers are close
    used = [False] * len(items)
    groups = []

    for i, item in enumerate(items):
        if used[i]:
            continue
        group = [item]
        used[i] = True
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            dy = abs(items[j]['center'][1] - item['center'][1])
            dx = abs(items[j]['center'][0] - item['center'][0])
            if dy < GROUP_Y_THRESHOLD and dx < GROUP_X_THRESHOLD:
                group.append(items[j])
                used[j] = True
        groups.append(group)

    # Build plate candidates from groups
    candidates = []
    for group in groups:
        # Sort group top-to-bottom
        group.sort(key=lambda it: it['center'][1])
        combined_text = ' '.join(it['text'] for it in group)
        avg_conf = np.mean([it['conf'] for it in group])

        # Compute bounding rect of entire group
        all_pts = []
        for it in group:
            for pt in it['bbox']:
                all_pts.append(pt)
        all_pts = np.array(all_pts, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(all_pts)

        candidates.append({
            'text': combined_text,
            'confidence': avg_conf,
            'rect': (x, y, w, h),
            'all_points': all_pts,
            'num_lines': len(group),
        })

    return candidates


# ═══════════════════════════════════════════════════════════
#  COLORS & STYLE
# ═══════════════════════════════════════════════════════════
class Colors:
    BG_DARK    = "#0f0f1a"
    BG_PANEL   = "#161625"
    BG_CARD    = "#1e1e35"
    BG_HEADER  = "#12122a"
    ACCENT     = "#6c5ce7"
    ACCENT2    = "#a29bfe"
    SUCCESS    = "#00b894"
    WARNING    = "#fdcb6e"
    DANGER     = "#e17055"
    TEXT       = "#dfe6e9"
    TEXT_DIM   = "#636e72"
    TEXT_BRIGHT = "#ffffff"
    BORDER     = "#2d2d50"
    PLATE_BG   = "#ffeaa7"
    PLATE_FG   = "#2d3436"


# ═══════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════
class LicensePlateApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VN License Plate Detector")
        self.root.configure(bg=Colors.BG_DARK)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        try:
            self.root.state("zoomed")
        except tk.TclError:
            self.root.geometry("1400x800")

        self.root.minsize(1000, 600)

        # State
        self.cap = None
        self.running = False
        self.reader = None
        self.last_ocr_time = 0
        self.detected_plates: deque = deque(maxlen=100)
        self.recent_plates: dict = {}  # plate_text -> last_time for cooldown
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.display_frame = None  # frame with annotations drawn
        self.plate_count = 0
        self.fps_counter = 0
        self.fps = 0
        self.fps_time = time.time()

        # Build UI
        self._build_ui()

        # Init OCR in background
        self._log_system("Đang khởi tạo EasyOCR engine...")
        threading.Thread(target=self._init_ocr, daemon=True).start()

    # ───────────────────── UI ─────────────────────
    def _build_ui(self):
        # ── Header Bar ──
        header = tk.Frame(self.root, bg=Colors.BG_HEADER, height=60)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        # Logo / Title
        title_frame = tk.Frame(header, bg=Colors.BG_HEADER)
        title_frame.pack(side=tk.LEFT, padx=20)

        tk.Label(title_frame, text="◉", font=("Segoe UI", 20), fg=Colors.ACCENT,
                 bg=Colors.BG_HEADER).pack(side=tk.LEFT)
        tk.Label(title_frame, text="  VN PLATE DETECTOR", font=("Segoe UI Semibold", 16),
                 fg=Colors.TEXT_BRIGHT, bg=Colors.BG_HEADER).pack(side=tk.LEFT)
        tk.Label(title_frame, text="  REALTIME", font=("Segoe UI", 9),
                 fg=Colors.ACCENT2, bg=Colors.BG_HEADER).pack(side=tk.LEFT, pady=(5, 0))

        # Status indicator
        self.status_dot = tk.Label(header, text="●", font=("Segoe UI", 14),
                                    fg=Colors.DANGER, bg=Colors.BG_HEADER)
        self.status_dot.pack(side=tk.RIGHT, padx=(0, 10))
        self.status_text = tk.Label(header, text="OFFLINE", font=("Segoe UI", 10),
                                     fg=Colors.TEXT_DIM, bg=Colors.BG_HEADER)
        self.status_text.pack(side=tk.RIGHT)

        # FPS counter
        self.fps_label = tk.Label(header, text="", font=("Consolas", 10),
                                   fg=Colors.TEXT_DIM, bg=Colors.BG_HEADER)
        self.fps_label.pack(side=tk.RIGHT, padx=20)

        # Control buttons
        btn_frame = tk.Frame(header, bg=Colors.BG_HEADER)
        btn_frame.pack(side=tk.RIGHT, padx=10)

        self.btn_start = self._make_button(btn_frame, "▶  BẮT ĐẦU", Colors.SUCCESS,
                                            self.start_camera)
        self.btn_start.pack(side=tk.LEFT, padx=4)

        self.btn_stop = self._make_button(btn_frame, "■  DỪNG", Colors.DANGER,
                                           self.stop_camera)
        self.btn_stop.pack(side=tk.LEFT, padx=4)
        self.btn_stop.config(state=tk.DISABLED)

        self.btn_clear = self._make_button(btn_frame, "✕  XÓA", Colors.TEXT_DIM,
                                            self.clear_log)
        self.btn_clear.pack(side=tk.LEFT, padx=4)

        # ── Main Content ──
        main = tk.Frame(self.root, bg=Colors.BG_DARK)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        # ── LEFT: Camera Panel ──
        left = tk.Frame(main, bg=Colors.BG_PANEL, relief=tk.FLAT,
                         highlightbackground=Colors.BORDER, highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        cam_header = tk.Frame(left, bg=Colors.BG_CARD, height=40)
        cam_header.pack(fill=tk.X)
        cam_header.pack_propagate(False)
        tk.Label(cam_header, text="📷  CAMERA FEED", font=("Segoe UI Semibold", 11),
                 fg=Colors.ACCENT2, bg=Colors.BG_CARD).pack(side=tk.LEFT, padx=12, pady=8)

        self.canvas = tk.Canvas(left, bg="#000000", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # Camera info bar
        cam_info = tk.Frame(left, bg=Colors.BG_CARD, height=30)
        cam_info.pack(fill=tk.X)
        cam_info.pack_propagate(False)
        self.cam_info_label = tk.Label(cam_info, text="Camera: Chưa kết nối",
                                        font=("Segoe UI", 9), fg=Colors.TEXT_DIM,
                                        bg=Colors.BG_CARD)
        self.cam_info_label.pack(side=tk.LEFT, padx=12)

        # ── RIGHT: Detection Panel ──
        right = tk.Frame(main, bg=Colors.BG_PANEL,
                          highlightbackground=Colors.BORDER, highlightthickness=1)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        det_header = tk.Frame(right, bg=Colors.BG_CARD, height=40)
        det_header.pack(fill=tk.X)
        det_header.pack_propagate(False)
        tk.Label(det_header, text="🔍  BIỂN SỐ PHÁT HIỆN", font=("Segoe UI Semibold", 11),
                 fg=Colors.ACCENT2, bg=Colors.BG_CARD).pack(side=tk.LEFT, padx=12, pady=8)
        self.count_label = tk.Label(det_header, text="0", font=("Segoe UI Bold", 12),
                                     fg=Colors.ACCENT, bg=Colors.BG_CARD)
        self.count_label.pack(side=tk.RIGHT, padx=12)

        # Latest plate card
        latest_card = tk.Frame(right, bg=Colors.BG_CARD, relief=tk.FLAT,
                                highlightbackground=Colors.ACCENT, highlightthickness=2)
        latest_card.pack(fill=tk.X, padx=8, pady=(8, 4))

        tk.Label(latest_card, text="BIỂN SỐ MỚI NHẤT", font=("Segoe UI", 9),
                 fg=Colors.TEXT_DIM, bg=Colors.BG_CARD).pack(pady=(8, 2))

        # Plate display (mimics a real plate)
        plate_frame = tk.Frame(latest_card, bg=Colors.PLATE_BG, relief=tk.RIDGE, bd=3)
        plate_frame.pack(padx=20, pady=4)

        self.latest_plate_var = tk.StringVar(value="-- --- --")
        self.plate_display = tk.Label(plate_frame, textvariable=self.latest_plate_var,
                                       font=("Consolas", 32, "bold"),
                                       bg=Colors.PLATE_BG, fg=Colors.PLATE_FG,
                                       padx=20, pady=8)
        self.plate_display.pack()

        self.conf_label = tk.Label(latest_card, text="Độ tin cậy: ---",
                                    font=("Segoe UI", 10), fg=Colors.TEXT_DIM,
                                    bg=Colors.BG_CARD)
        self.conf_label.pack(pady=(2, 4))

        # Plate image preview
        self.preview_frame = tk.Frame(latest_card, bg=Colors.BG_CARD)
        self.preview_frame.pack(pady=(0, 8))
        self.plate_preview = tk.Label(self.preview_frame, bg=Colors.BG_CARD,
                                       text="Chưa có ảnh", font=("Segoe UI", 9),
                                       fg=Colors.TEXT_DIM)
        self.plate_preview.pack()

        # ── History List ──
        hist_header = tk.Frame(right, bg=Colors.BG_PANEL)
        hist_header.pack(fill=tk.X, padx=8, pady=(8, 2))
        tk.Label(hist_header, text="📋  LỊCH SỬ", font=("Segoe UI Semibold", 10),
                 fg=Colors.TEXT, bg=Colors.BG_PANEL).pack(side=tk.LEFT)

        # Scrollable history
        hist_container = tk.Frame(right, bg=Colors.BG_DARK)
        hist_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.hist_canvas = tk.Canvas(hist_container, bg=Colors.BG_DARK,
                                      highlightthickness=0)
        scrollbar = ttk.Scrollbar(hist_container, orient=tk.VERTICAL,
                                   command=self.hist_canvas.yview)
        self.hist_inner = tk.Frame(self.hist_canvas, bg=Colors.BG_DARK)

        self.hist_inner.bind("<Configure>",
                              lambda e: self.hist_canvas.configure(
                                  scrollregion=self.hist_canvas.bbox("all")))

        self.hist_canvas.create_window((0, 0), window=self.hist_inner, anchor="nw",
                                        tags="inner")
        self.hist_canvas.configure(yscrollcommand=scrollbar.set)
        self.hist_canvas.bind("<Configure>",
                               lambda e: self.hist_canvas.itemconfig("inner", width=e.width))

        # Mouse wheel scrolling
        self.hist_canvas.bind("<MouseWheel>",
                               lambda e: self.hist_canvas.yview_scroll(
                                   int(-1 * (e.delta / 120)), "units"))

        self.hist_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # System log at bottom
        log_frame = tk.Frame(right, bg=Colors.BG_CARD, height=60)
        log_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        log_frame.pack_propagate(False)

        tk.Label(log_frame, text="SYSTEM", font=("Segoe UI", 8),
                 fg=Colors.TEXT_DIM, bg=Colors.BG_CARD).pack(anchor=tk.W, padx=8, pady=(4, 0))
        self.sys_log_var = tk.StringVar(value="Đang khởi tạo...")
        tk.Label(log_frame, textvariable=self.sys_log_var, font=("Consolas", 9),
                 fg=Colors.ACCENT2, bg=Colors.BG_CARD, anchor=tk.W,
                 wraplength=400).pack(fill=tk.X, padx=8)

    def _make_button(self, parent, text, color, command):
        btn = tk.Button(parent, text=text, font=("Segoe UI Semibold", 9),
                        fg=Colors.TEXT_BRIGHT, bg=color,
                        activebackground=color, activeforeground=Colors.TEXT_BRIGHT,
                        relief=tk.FLAT, cursor="hand2", padx=12, pady=4,
                        command=command)
        btn.bind("<Enter>", lambda e, b=btn, c=color: b.config(bg=self._lighten(c)))
        btn.bind("<Leave>", lambda e, b=btn, c=color: b.config(bg=c))
        return btn

    @staticmethod
    def _lighten(hex_color, factor=0.2):
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:], 16)
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    # ───────────────────── OCR INIT ─────────────────────
    def _init_ocr(self):
        try:
            self.reader = easyocr.Reader(['en'], gpu=True)
            self._log_system("✅ EasyOCR sẵn sàng (GPU enabled)" if self.reader else "EasyOCR ready")
        except Exception as e:
            self._log_system(f"⚠ EasyOCR init error: {e}")

    # ───────────────────── CAMERA ─────────────────────
    def start_camera(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            self._log_system("❌ Không thể mở camera!")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_dot.config(fg=Colors.SUCCESS)
        self.status_text.config(text="LIVE", fg=Colors.SUCCESS)
        self.cam_info_label.config(text=f"Camera {CAMERA_INDEX}  •  {actual_w}×{actual_h}")
        self._log_system(f"📷 Camera đang chạy ({actual_w}×{actual_h})")

        threading.Thread(target=self._camera_loop, daemon=True).start()
        threading.Thread(target=self._ocr_loop, daemon=True).start()
        self._update_canvas()

    def stop_camera(self):
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_dot.config(fg=Colors.DANGER)
        self.status_text.config(text="OFFLINE", fg=Colors.TEXT_DIM)
        self.cam_info_label.config(text="Camera: Đã dừng")
        self._log_system("⏹ Camera đã dừng")
        if self.cap:
            self.cap.release()
            self.cap = None

    def _camera_loop(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()
                self.fps_counter += 1
            time.sleep(0.025)

    # ───────────────────── OCR LOOP ─────────────────────
    def _ocr_loop(self):
        while self.running:
            if self.reader is None:
                time.sleep(0.5)
                continue

            now = time.time()
            if now - self.last_ocr_time < OCR_INTERVAL:
                time.sleep(0.1)
                continue

            with self.frame_lock:
                frame = self.current_frame.copy() if self.current_frame is not None else None

            if frame is None:
                time.sleep(0.1)
                continue

            self.last_ocr_time = now
            self._detect_plates(frame)

    def _detect_plates(self, frame: np.ndarray):
        """Detect and group plate text regions."""
        try:
            # Use detail=1 for full bbox info
            results = self.reader.readtext(frame, detail=1,
                                            allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.-',
                                            min_size=20, width_ths=0.9,
                                            paragraph=False)
        except Exception as e:
            self.root.after(0, lambda: self._log_system(f"⚠ OCR error: {e}"))
            return

        if not results:
            return

        # Filter low confidence
        filtered = [(bbox, text, conf) for bbox, text, conf in results
                     if conf >= CONFIDENCE_THRESHOLD and len(text.strip()) >= 2]

        if not filtered:
            return

        # Group nearby detections (for 2-line plates)
        candidates = group_detections(filtered)

        # Annotated frame for display
        annotated = frame.copy()

        for cand in candidates:
            text = cand['text']
            conf = cand['confidence']
            x, y, w, h = cand['rect']

            # Check if it looks like a Vietnamese plate
            cleaned = re.sub(r'[^A-Z0-9.\- ]', '', text.strip().upper())

            # Must have at least one digit and reasonable length
            has_digit = any(c.isdigit() for c in cleaned)
            if not has_digit or len(cleaned.replace(' ', '')) < 4:
                continue

            # Normalize to VN plate format
            plate_text = normalize_plate(cleaned)

            # Check cooldown (avoid saving same plate repeatedly)
            now = time.time()
            if plate_text in self.recent_plates:
                if now - self.recent_plates[plate_text] < DUPLICATE_COOLDOWN:
                    # Still draw the box but don't save
                    cv2.rectangle(annotated, (x - 2, y - 2), (x + w + 2, y + h + 2),
                                  (0, 255, 255), 2)
                    continue

            self.recent_plates[plate_text] = now

            # Crop with padding
            pad = CROP_PADDING
            y1 = max(0, y - pad)
            y2 = min(frame.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(frame.shape[1], x + w + pad)
            plate_crop = frame[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            # Draw detection on annotated frame
            cv2.rectangle(annotated, (x - 2, y - 2), (x + w + 2, y + h + 2),
                          (0, 255, 0), 2)
            # Label above box
            label = f"{plate_text} ({conf:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x - 2, y - th - 12), (x + tw + 4, y - 2),
                          (0, 255, 0), -1)
            cv2.putText(annotated, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Save
            ensure_database()
            save_plate(plate_text, conf, plate_crop)
            self.plate_count += 1

            # Update UI
            self.root.after(0, lambda t=plate_text, c=conf, img=plate_crop.copy():
                            self._on_plate_detected(t, c, img))

        # Store annotated frame for display
        with self.frame_lock:
            self.display_frame = annotated

    def _on_plate_detected(self, plate_text: str, confidence: float, plate_img: np.ndarray):
        """Update UI on plate detection (main thread)."""
        self.latest_plate_var.set(plate_text)
        self.conf_label.config(text=f"Độ tin cậy: {confidence:.0%}")
        self.count_label.config(text=str(self.plate_count))

        # Flash effect on plate display
        self.plate_display.config(bg="#55efc4", fg="#000000")
        self.root.after(300, lambda: self.plate_display.config(
            bg=Colors.PLATE_BG, fg=Colors.PLATE_FG))

        # Show plate crop preview
        try:
            img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pw = 280
            ratio = pw / max(pil_img.width, 1)
            ph = max(int(pil_img.height * ratio), 20)
            pil_img = pil_img.resize((pw, ph), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil_img)
            self.plate_preview.config(image=tk_img, text="")
            self.plate_preview.image = tk_img
        except Exception:
            pass

        # Add to history
        self._add_history_entry(plate_text, confidence, plate_img)

        self._log_system(f"✅ Phát hiện: {plate_text} ({confidence:.0%})")

    def _add_history_entry(self, plate_text: str, confidence: float, plate_img: np.ndarray):
        """Add a card to the scrollable history."""
        ts = datetime.now().strftime("%H:%M:%S")

        card = tk.Frame(self.hist_inner, bg=Colors.BG_CARD, relief=tk.FLAT,
                         highlightbackground=Colors.BORDER, highlightthickness=1)
        card.pack(fill=tk.X, pady=2, before=None if not self.hist_inner.winfo_children()
                  else self.hist_inner.winfo_children()[0])

        # Top row: time + plate text
        row = tk.Frame(card, bg=Colors.BG_CARD)
        row.pack(fill=tk.X, padx=8, pady=4)

        tk.Label(row, text=ts, font=("Consolas", 9),
                 fg=Colors.TEXT_DIM, bg=Colors.BG_CARD).pack(side=tk.LEFT)

        tk.Label(row, text=plate_text, font=("Consolas", 14, "bold"),
                 fg=Colors.WARNING, bg=Colors.BG_CARD).pack(side=tk.LEFT, padx=10)

        conf_color = Colors.SUCCESS if confidence >= 0.7 else (
            Colors.WARNING if confidence >= 0.5 else Colors.DANGER)
        tk.Label(row, text=f"{confidence:.0%}", font=("Segoe UI", 10, "bold"),
                 fg=conf_color, bg=Colors.BG_CARD).pack(side=tk.RIGHT)

        # Plate thumbnail
        try:
            img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tw = 140
            ratio = tw / max(pil_img.width, 1)
            th = max(int(pil_img.height * ratio), 15)
            pil_img = pil_img.resize((tw, th), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil_img)
            thumb = tk.Label(card, image=tk_img, bg=Colors.BG_CARD)
            thumb.image = tk_img
            thumb.pack(padx=8, pady=(0, 4), anchor=tk.W)
        except Exception:
            pass

        # Auto-scroll to top
        self.hist_canvas.yview_moveto(0)

    # ───────────────────── CANVAS UPDATE ─────────────────
    def _update_canvas(self):
        if not self.running:
            return

        # FPS calculation
        now = time.time()
        if now - self.fps_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = now
            self.fps_label.config(text=f"FPS: {self.fps}")

        with self.frame_lock:
            # Prefer annotated frame if available
            frame = (self.display_frame if self.display_frame is not None
                     else self.current_frame)
            frame = frame.copy() if frame is not None else None

        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw > 1 and ch > 1:
                # Maintain aspect ratio
                img_w, img_h = pil_img.size
                scale = min(cw / img_w, ch / img_h)
                new_w = int(img_w * scale)
                new_h = int(img_h * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

                # Center on canvas
                x_offset = (cw - new_w) // 2
                y_offset = (ch - new_h) // 2
            else:
                x_offset, y_offset = 0, 0

            tk_img = ImageTk.PhotoImage(pil_img)
            self.canvas.delete("all")
            self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=tk_img)
            self.canvas.image = tk_img

        self.root.after(33, self._update_canvas)

    # ───────────────────── HELPERS ─────────────────────
    def _log_system(self, msg: str):
        try:
            self.sys_log_var.set(msg)
        except Exception:
            pass

    def clear_log(self):
        for widget in self.hist_inner.winfo_children():
            widget.destroy()
        self.plate_count = 0
        self.detected_plates.clear()
        self.recent_plates.clear()
        self.latest_plate_var.set("-- --- --")
        self.conf_label.config(text="Độ tin cậy: ---")
        self.count_label.config(text="0")
        self.plate_preview.config(image="", text="Chưa có ảnh")
        self._log_system("🗑 Đã xóa log")

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    ensure_database()
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()
