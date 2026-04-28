import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import cv2
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────
#  ICON HELPER
# ─────────────────────────────────────────
# icon.ico sits next to main.py (and inside the PyInstaller bundle via
# sys._MEIPASS), so we resolve it at import time.
def _get_icon_path():
    import sys
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, 'icon.ico')

def _apply_icon(win):
    """Set icon.ico on any Tk or Toplevel window, silently ignoring errors."""
    try:
        win.iconbitmap(_get_icon_path())
    except Exception:
        pass

def _set_appusermodelid():
    """Pin the taskbar icon to Detectra (Windows only).
    Must be called before the first Tk() is created."""
    try:
        from ctypes import windll
        windll.shell32.SetCurrentProcessExplicitAppUserModelID('Detectra.App')
    except Exception:
        pass


# ─────────────────────────────────────────
#  SPLASH SCREEN
# ─────────────────────────────────────────
class SplashScreen:
    def __init__(self, root):
        self.root = root
        self.root.overrideredirect(True)
        self.root.configure(bg="#1E1E2E")
        self.root.attributes('-topmost', True)
        _apply_icon(self.root)

        w, h = 480, 280
        self.root.update_idletasks()   # ← add this line
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

        tk.Label(self.root, text="🎯 Detectra",
                 font=("Segoe UI", 32, "bold"),
                 bg="#1E1E2E", fg="#89B4FA").pack(pady=(40, 4))

        tk.Label(self.root, text="Object Disappearance Detection System",
                 font=("Segoe UI", 12),
                 bg="#1E1E2E", fg="#A6ADC8").pack()

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Splash.Horizontal.TProgressbar",
                         troughcolor="#313244", background="#89B4FA",
                         bordercolor="#1E1E2E", lightcolor="#89B4FA",
                         darkcolor="#89B4FA")

        self.progress = ttk.Progressbar(self.root,
                                         style="Splash.Horizontal.TProgressbar",
                                         orient="horizontal",
                                         length=380, mode="determinate")
        self.progress.pack(pady=24)

        self.status_var = tk.StringVar(value="Starting...")
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Segoe UI", 10),
                 bg="#1E1E2E", fg="#6C7086").pack()

        tk.Label(self.root, text="v3.0.0",
                 font=("Segoe UI", 8),
                 bg="#1E1E2E", fg="#45475A").place(relx=1.0, rely=1.0,
                                                    anchor="se", x=-10, y=-10)
        self.tracker = None
        threading.Thread(target=self._load, daemon=True).start()

    def _update(self, val, msg):
        self.progress['value'] = val
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _load(self):
        self._update(20, "Loading OpenCV...")
        import cv2  # noqa

        self._update(50, "Loading AI models...")
        try:
            from tracker import Tracker
            self.tracker = Tracker()
            self._update(70, "Initializing OCR...")
            self.tracker.init_ocr()
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda msg=err: self.status_var.set(f"Error: {msg}"))
            return

        self._update(90, "Preparing interface...")
        import time; time.sleep(0.3)

        self._update(100, "Ready!")
        import time; time.sleep(0.4)

        # ── Schedule the transition on the main thread ──
        self.root.after(0, self._open_main_app)

    def _open_main_app(self):
        """Runs on main thread — safe to create new Tk window."""
        self.root.destroy()                        # close splash
        _set_appusermodelid()
        root = tk.Tk()                             # open main window
        _apply_icon(root)
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        DetectraApp(root, self.tracker)
        root.mainloop()


# ─────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────
class DetectraApp:
    def __init__(self, root, tracker):
        self.root = root
        self.root.title("Detectra - Object Disappearance Detection")
        self.root.geometry("1000x700")
        _apply_icon(self.root)

        # Core state
        self.video_path = None
        self.first_frame_rgb = None
        self.detections = []
        self.selected_bbox = None
        self.tracker = tracker

        # Multi-file queue
        self.video_queue   = []   # list of file paths yet to be processed
        self.queue_index   = 0    # which file in the full batch we're on
        self.queue_total   = 0    # total files in current batch
        self._auto_advance = False  # True while auto-advancing to next file

        # Display state
        self.canvas_image = None
        self.photo = None
        self.scale_factor = 1.0
        self.x_offset = 0
        self.y_offset = 0

        # Drawing state
        self.rect_start_x = None
        self.rect_start_y = None
        self.rect_id      = None
        self.ocr_rect_id  = None          # amber OCR-region box drawn on canvas
        self._drawing_mode = 'object'     # 'object' | 'ocr'
        self.ocr_bbox     = None          # (x1,y1,x2,y2) in image coords for OCR crop
        self.resize_timer = None
        self.stop_event = None
        self.is_paused = False
        self.current_frame_idx = 0
        self.last_shown_frame_rgb = None
        self.last_results = None
        # Smooth playback: holds the newest pending frame so we never queue
        # more than one GUI update at a time.
        self._pending_frame = None     # (frame_rgb, bbox) | None
        self._frame_scheduled = False
        self._pending_progress = None   # (current, total) | None
        self._progress_scheduled = False
        self._results_win = None        # live reference to results Toplevel

        self.apply_theme()
        self.setup_ui()

    def apply_theme(self):
        self.style = ttk.Style()
        if 'clam' in self.style.theme_names():
            self.style.theme_use('clam')

        bg_color     = "#1E1E2E"
        fg_color     = "#CDD6F4"
        accent_color = "#89B4FA"
        btn_bg       = "#313244"
        btn_active   = "#45475A"

        self.root.configure(bg=bg_color)

        self.style.configure(".", background=bg_color, foreground=fg_color,
                             font=('Segoe UI', 12))
        self.style.configure("TButton",
                             padding=(20, 10), relief="flat",
                             background=btn_bg, foreground=fg_color,
                             font=('Segoe UI', 12, 'bold'))
        self.style.map("TButton",
                       background=[('active', btn_active), ('disabled', '#181825')],
                       foreground=[('disabled', '#585B70')])
        self.style.configure("Accent.TButton",
                             padding=(20, 10), relief="flat",
                             background=accent_color, foreground="#11111B",
                             font=('Segoe UI', 12, 'bold'))
        self.style.map("Accent.TButton",
                       background=[('active', "#B4BEFE"), ('disabled', '#181825')],
                       foreground=[('disabled', '#585B70')])
        self.style.configure("Stop.TButton",
                             padding=(20, 10), relief="flat",
                             background="#F38BA8", foreground="#11111B",
                             font=('Segoe UI', 12, 'bold'))
        self.style.map("Stop.TButton",
                       background=[('active', "#EBA0AC"), ('disabled', '#181825')],
                       foreground=[('disabled', '#585B70')])
        self.style.configure("Horizontal.TProgressbar",
                             background="#A6E3A1", troughcolor="#313244",
                             bordercolor=bg_color, lightcolor="#A6E3A1",
                             darkcolor="#A6E3A1")
        self.style.configure("TLabel", background=bg_color, foreground=fg_color,
                             font=('Segoe UI', 12))
        self.style.configure("TCheckbutton", background=bg_color, foreground=fg_color,
                             font=('Segoe UI', 12))
        self.style.map("TCheckbutton", background=[('active', bg_color)])
        self.style.configure("TFrame", background=bg_color)

    def setup_ui(self):
        self.control_frame = ttk.Frame(self.root, padding="15")
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        self.upload_btn = ttk.Button(self.control_frame, text="Upload Video",
                                     command=self.upload_video)
        self.upload_btn.pack(side=tk.LEFT, padx=(0, 15))

        self.status_lbl = ttk.Label(self.control_frame,
                                    text="Welcome to Detectra. Please upload a CCTV video.",
                                    font=('Segoe UI', 14, 'bold'))
        self.status_lbl.pack(side=tk.LEFT, padx=10)

        self.start_btn = ttk.Button(self.control_frame, text="Start Tracking",
                                    command=self.start_tracking,
                                    state=tk.DISABLED, style="Accent.TButton")
        self.start_btn.pack(side=tk.RIGHT, padx=(8, 0))

        self.stop_btn = ttk.Button(self.control_frame, text="Stop Tracking",
                                   command=self.true_stop_tracking,
                                   style="Stop.TButton")
        self.stop_btn.pack(side=tk.RIGHT)
        self.stop_btn.pack_forget()

        self.canvas_frame = ttk.Frame(self.root, padding=15)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#11111B",
                                highlightthickness=2, highlightbackground="#313244")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>",   self.on_press)
        self.canvas.bind("<B1-Motion>",       self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Configure>",       self.on_resize)

        self.bottom_frame = ttk.Frame(self.root, padding="15")
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.bottom_frame,
                                             variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        self.options_frame = ttk.Frame(self.bottom_frame)
        self.options_frame.pack(fill=tk.X)

        self.show_tracking_var = tk.BooleanVar(value=True)
        self.show_tracking_chk = ttk.Checkbutton(self.options_frame,
                                                  text="Show Tracking Feed",
                                                  variable=self.show_tracking_var)
        self.show_tracking_chk.pack(side=tk.LEFT)

        self.ocr_btn = ttk.Button(self.options_frame, text="📍 Set OCR Region",
                                  command=self.start_ocr_selection)
        self.ocr_btn.pack(side=tk.LEFT, padx=(15, 0))
        self.ocr_btn.pack_forget()  # hidden until video is loaded

        self.ocr_region_lbl = ttk.Label(self.options_frame, text="",
                                        font=('Segoe UI', 10))
        self.ocr_region_lbl.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(self.options_frame, text="Speed:",
                  font=('Segoe UI', 11)).pack(side=tk.LEFT, padx=(20, 4))
        self.speed_var = tk.IntVar(value=5)
        self.speed_slider = ttk.Scale(self.options_frame, from_=1, to=30,
                                      orient=tk.HORIZONTAL, variable=self.speed_var,
                                      length=140, command=self._on_speed_change)
        self.speed_slider.pack(side=tk.LEFT)
        self.speed_lbl = ttk.Label(self.options_frame, text="5x",
                                   font=('Segoe UI', 12, 'bold'), width=4)
        self.speed_lbl.pack(side=tk.LEFT, padx=(4, 0))

        self.results_btn = ttk.Button(self.options_frame, text="Show Last Results",
                                      command=self.view_last_results)
        self.results_btn.pack(side=tk.RIGHT)
        self.results_btn.pack_forget()

        self.info_lbl = ttk.Label(self.bottom_frame, text="",
                                  font=("Segoe UI", 14, "bold"))
        self.info_lbl.pack(pady=(10, 0))

    def _on_speed_change(self, _=None):
        v = int(self.speed_var.get())
        self.speed_lbl.config(text=f"{v}x")

    def start_ocr_selection(self):
        """Switch canvas to OCR-region draw mode."""
        self._drawing_mode = 'ocr'
        # Clear any previous OCR rect visual
        if self.ocr_rect_id:
            self.canvas.delete(self.ocr_rect_id)
            self.ocr_rect_id = None
        self.status_lbl.config(
            text="Draw a box around the on-screen timestamp/clock for OCR.")
        self.ocr_region_lbl.config(text="Drawing...", foreground="#FAB387")

    def on_resize(self, event):
        if self.resize_timer is not None:
            self.root.after_cancel(self.resize_timer)
        self.resize_timer = self.root.after(200, self.do_resize)

    def do_resize(self):
        if self.first_frame_rgb is not None:
            self.draw_frame()
            if self.selected_bbox:
                x1, y1, x2, y2 = self.selected_bbox
                cx1 = int(x1 * self.scale_factor) + self.x_offset
                cy1 = int(y1 * self.scale_factor) + self.y_offset
                cx2 = int(x2 * self.scale_factor) + self.x_offset
                cy2 = int(y2 * self.scale_factor) + self.y_offset
                self.rect_id = self.canvas.create_rectangle(
                    cx1, cy1, cx2, cy2, outline='#00FF00', width=2)
            if self.ocr_bbox:
                ox1, oy1, ox2, oy2 = self.ocr_bbox
                ocx1 = int(ox1 * self.scale_factor) + self.x_offset
                ocy1 = int(oy1 * self.scale_factor) + self.y_offset
                ocx2 = int(ox2 * self.scale_factor) + self.x_offset
                ocy2 = int(oy2 * self.scale_factor) + self.y_offset
                self.ocr_rect_id = self.canvas.create_rectangle(
                    ocx1, ocy1, ocx2, ocy2, outline='#FAB387', width=2)

    def upload_video(self):
        filetypes = (
            ('Video files', '*.mp4 *.avi *.mkv *.mov'),
            ('All files', '*.*')
        )
        filepaths = filedialog.askopenfilenames(title='Open video(s)', filetypes=filetypes)
        if not filepaths:
            return

        # Build queue — first file loads immediately, rest queued
        self.video_queue = list(filepaths)
        self.queue_index = 0
        self.queue_total = len(self.video_queue)
        self._load_video_from_queue(self.video_queue[0])

    def _load_video_from_queue(self, filepath):
        """Load a single video file from the queue and prepare the canvas."""
        self.video_path = filepath
        queue_info = (f" [{self.queue_index + 1}/{self.queue_total}]"
                      if self.queue_total > 1 else "")
        self.status_lbl.config(
            text=f"Loading{queue_info}: {os.path.basename(filepath)}...")
        self.root.update()

        frame, err = self.tracker.extract_first_frame(self.video_path)
        if err:
            messagebox.showerror("Error", err)
            self.status_lbl.config(text="Error loading video.")
            return

        self.first_frame_rgb       = frame
        self.last_shown_frame_rgb  = frame

        # Reset per-video tracking state
        self.selected_bbox         = None
        self.ocr_bbox              = None
        self._drawing_mode         = 'object'
        self.is_paused             = False
        self.current_frame_idx     = 0
        self.last_results          = None
        self.stop_btn.pack_forget()
        self.results_btn.pack_forget()
        self.start_btn.config(text="Start Tracking", state=tk.DISABLED)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        if self.ocr_rect_id:
            self.canvas.delete(self.ocr_rect_id)
            self.ocr_rect_id = None
        self.progress_var.set(0)
        self.info_lbl.config(text="")

        self.status_lbl.config(text="Detecting objects...")
        self.root.update()
        self.draw_frame()

        queue_info = (f" [{self.queue_index + 1}/{self.queue_total}]"
                      if self.queue_total > 1 else "")
        self.status_lbl.config(
            text=f"Video{queue_info} loaded. Draw a bounding box around the object to track.")

        # Show OCR region button now that a video is loaded
        self.ocr_btn.pack(side=tk.LEFT, padx=(15, 0))
        self.ocr_region_lbl.config(text="OCR: auto (default)")
        self._update_queue_label()

    def _update_queue_label(self):
        """Show/hide a small queue status badge next to the upload button."""
        if not hasattr(self, '_queue_lbl'):
            self._queue_lbl = ttk.Label(
                self.control_frame, text="",
                font=('Segoe UI', 10), foreground="#FAB387")
            self._queue_lbl.pack(side=tk.LEFT, padx=(0, 8))
        if self.queue_total > 1:
            remaining = self.queue_total - self.queue_index - 1
            if remaining > 0:
                self._queue_lbl.config(text=f"\U0001f4c2 {remaining} more video(s) queued")
            else:
                self._queue_lbl.config(text="")
        else:
            self._queue_lbl.config(text="")

    def draw_frame(self, frame_rgb=None, bbox=None):
        if frame_rgb is not None:
            self.last_shown_frame_rgb = frame_rgb
            img_to_draw = frame_rgb
        elif self.last_shown_frame_rgb is not None:
            img_to_draw = self.last_shown_frame_rgb
        else:
            if self.first_frame_rgb is None:
                return
            img_to_draw = self.first_frame_rgb
            self.last_shown_frame_rgb = self.first_frame_rgb

        self.canvas.delete("all")
        self.rect_id     = None
        self.ocr_rect_id = None

        canvas_width  = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1:
            canvas_width  = 800
            canvas_height = 600

        img = Image.fromarray(img_to_draw)
        img_width, img_height = img.size

        scale_w = canvas_width  / img_width
        scale_h = canvas_height / img_height
        self.scale_factor = min(scale_w, scale_h)

        new_width  = int(img_width  * self.scale_factor)
        new_height = int(img_height * self.scale_factor)

        self.x_offset = (canvas_width  - new_width)  // 2
        self.y_offset = (canvas_height - new_height) // 2

        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo  = ImageTk.PhotoImage(img_resized)
        self.canvas.create_image(self.x_offset, self.y_offset,
                                  anchor=tk.NW, image=self.photo)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cx1 = int(x1 * self.scale_factor) + self.x_offset
            cy1 = int(y1 * self.scale_factor) + self.y_offset
            cx2 = int(x2 * self.scale_factor) + self.x_offset
            cy2 = int(y2 * self.scale_factor) + self.y_offset
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="red", width=2)

    def on_press(self, event):
        if self.last_shown_frame_rgb is None:
            return
        self.rect_start_x = event.x
        self.rect_start_y = event.y
        if self._drawing_mode == 'object':
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(
                self.rect_start_x, self.rect_start_y,
                self.rect_start_x, self.rect_start_y,
                outline='#00FF00', width=2)
        else:  # 'ocr'
            if self.ocr_rect_id:
                self.canvas.delete(self.ocr_rect_id)
            self.ocr_rect_id = self.canvas.create_rectangle(
                self.rect_start_x, self.rect_start_y,
                self.rect_start_x, self.rect_start_y,
                outline='#FAB387', width=2, dash=(6, 3))

    def on_drag(self, event):
        if self.last_shown_frame_rgb is None:
            return
        if self._drawing_mode == 'object':
            if self.rect_id is None:
                return
            self.canvas.coords(self.rect_id,
                               self.rect_start_x, self.rect_start_y,
                               event.x, event.y)
        else:  # 'ocr'
            if self.ocr_rect_id is None:
                return
            self.canvas.coords(self.ocr_rect_id,
                               self.rect_start_x, self.rect_start_y,
                               event.x, event.y)

    def _coords_to_image(self, x1, y1, x2, y2):
        """Convert canvas coords to image-space coords, clamped to frame."""
        img_w = self.last_shown_frame_rgb.shape[1]
        img_h = self.last_shown_frame_rgb.shape[0]
        ix1 = max(0, min((min(x1, x2) - self.x_offset) / self.scale_factor, img_w))
        iy1 = max(0, min((min(y1, y2) - self.y_offset) / self.scale_factor, img_h))
        ix2 = max(0, min((max(x1, x2) - self.x_offset) / self.scale_factor, img_w))
        iy2 = max(0, min((max(y1, y2) - self.y_offset) / self.scale_factor, img_h))
        return int(ix1), int(iy1), int(ix2), int(iy2)

    def on_release(self, event):
        if self.last_shown_frame_rgb is None or self.rect_start_x is None:
            return
        ix1, iy1, ix2, iy2 = self._coords_to_image(
            self.rect_start_x, self.rect_start_y, event.x, event.y)

        if self._drawing_mode == 'object':
            if self.rect_id is None:
                return
            if ix2 - ix1 > 10 and iy2 - iy1 > 10:
                self.selected_bbox = (ix1, iy1, ix2, iy2)
                self.start_btn.config(state=tk.NORMAL)
                self.status_lbl.config(text="Object selected. Click 'Start Tracking'.")
            else:
                self.selected_bbox = None
                self.start_btn.config(state=tk.DISABLED)
                self.canvas.delete(self.rect_id)
                self.rect_id = None
                self.status_lbl.config(text="Box too small, draw again.")
        else:  # 'ocr'
            if self.ocr_rect_id is None:
                return
            if ix2 - ix1 > 5 and iy2 - iy1 > 5:
                self.ocr_bbox = (ix1, iy1, ix2, iy2)
                self.ocr_region_lbl.config(text="OCR region set ✓",
                                           foreground="#A6E3A1")
                self.status_lbl.config(
                    text="OCR region set! Draw the object box or click 'Start Tracking'.")
            else:
                self.ocr_bbox = None
                self.canvas.delete(self.ocr_rect_id)
                self.ocr_rect_id = None
                self.ocr_region_lbl.config(text="Too small — try again.",
                                           foreground="#F38BA8")
            # Always revert to object-draw mode after OCR selection
            self._drawing_mode = 'object'

    def update_progress(self, current, total):
        pct = (current / total) * 100
        self.progress_var.set(pct)
        self.root.update_idletasks()

    def _on_progress(self, current, total):
        """Thread-safe progress update with pending-guard.
        Only one update is ever queued; newer values overwrite the slot
        so the bar always reflects the latest position, never a stale backlog."""
        self._pending_progress = (current, total)
        if not self._progress_scheduled:
            self._progress_scheduled = True
            self.root.after(0, self._flush_pending_progress)

    def _flush_pending_progress(self):
        self._progress_scheduled = False
        if self._pending_progress is not None:
            current, total = self._pending_progress
            self._pending_progress = None
            self.update_progress(current, total)

    def live_view_callback(self, frame_rgb, bbox):
        self.draw_frame(frame_rgb, bbox)
        self.root.update_idletasks()

    # ── Smooth playback helpers ───────────────────────────────────────────

    def _on_new_frame(self, frame_rgb, bbox):
        """Called from the tracker thread. Only schedules ONE Tkinter update;
        newer frames overwrite the pending slot so the GUI always shows the
        latest frame rather than playing catch-up on a stale backlog."""
        self._pending_frame = (frame_rgb, bbox)
        if not self._frame_scheduled:
            self._frame_scheduled = True
            self.root.after(0, self._flush_pending_frame)

    def _flush_pending_frame(self):
        """Runs on the main thread. Draws whatever the latest frame is."""
        self._frame_scheduled = False
        if self._pending_frame is not None:
            frame_rgb, bbox = self._pending_frame
            self._pending_frame = None
            self.live_view_callback(frame_rgb, bbox)

    def start_tracking(self):
        if not self.video_path or not self.selected_bbox:
            return

        if self.stop_event and not self.stop_event.is_set():
            self.stop_tracking()
            return

        self.start_btn.config(text="Pause Tracking")
        self.upload_btn.config(state=tk.DISABLED)
        self.show_tracking_chk.config(state=tk.DISABLED)
        self.speed_slider.config(state=tk.DISABLED)
        self.results_btn.pack_forget()
        self.stop_btn.pack(side=tk.RIGHT)
        speed = int(self.speed_var.get())

        if self.is_paused:
            self.status_lbl.config(text=f"Resuming at {speed}x speed...")
        else:
            self.status_lbl.config(
                text=f"Processing video at {speed}x speed... Please wait.")
            self.current_frame_idx = 0

        self.progress_var.set(
            (self.current_frame_idx / float(
                self.total_frames_est
                if hasattr(self, 'total_frames_est') and self.total_frames_est
                else 100
            )) * 100
        )
        self.info_lbl.config(text="")
        self.stop_event = threading.Event()
        threading.Thread(target=self.run_tracker_thread, daemon=True).start()

    def stop_tracking(self):
        if self.stop_event is not None:
            self.stop_event.set()

    def true_stop_tracking(self):
        if self.stop_event:
            self.stop_event.set()
        self.is_paused = False
        self.current_frame_idx = 0
        self.selected_bbox = None
        self.ocr_bbox      = None
        self._drawing_mode = 'object'
        self.last_results  = None
        self.start_btn.config(text="Start Tracking")
        self.stop_btn.pack_forget()
        self.results_btn.pack_forget()
        self.status_lbl.config(text="Tracking reset.")
        self.ocr_region_lbl.config(text="OCR: auto (default)")
        self.info_lbl.config(text="")
        if self.first_frame_rgb is not None:
            self.draw_frame(self.first_frame_rgb)

    def run_tracker_thread(self):
        def progress_cb(current, total):
            self.total_frames_est = total
            self._on_progress(current, total)

        def frame_cb(frame_rgb, bbox):
            # Use the smooth pending-frame guard instead of direct after()
            # so fast speeds don't build a backlog of stale frames in the queue.
            self._on_new_frame(frame_rgb, bbox)

        def disappearance_cb(frame_before_bgr, frame_after_bgr, frame_idx,
                             frame_after_with_path=None):
            """Fired immediately when disappearance is detected, before OCR.
            Opens the results window right away with the before/after frames
            and the breadcrumb-annotated path frame (if available)."""
            from datetime import timedelta as _td
            fps_val = self.tracker._last_fps if hasattr(self.tracker, '_last_fps') else 25.0
            seconds = int(frame_idx / fps_val)
            early_results = {
                'disappeared'         : True,
                'last_frame_idx'      : frame_idx,
                'timestamp'           : str(_td(seconds=seconds)),
                'frame_before'        : frame_before_bgr,
                'frame_after'         : frame_after_bgr,
                'frame_after_with_path': frame_after_with_path,
            }
            self.root.after(0, self._on_disappearance_detected, early_results)

        callback_to_pass = frame_cb if self.show_tracking_var.get() else None
        frame_skip = int(self.speed_var.get())

        results = self.tracker.process_video(
            self.video_path, self.selected_bbox,
            progress_cb, callback_to_pass,
            stop_event=self.stop_event,
            frame_skip=frame_skip,
            start_frame=self.current_frame_idx,
            disappearance_callback=disappearance_cb,
            ocr_bbox=self.ocr_bbox
        )
        self.root.after(0, self.on_tracking_complete, results)

    def view_last_results(self):
        if self.last_results:
            self.show_results_window(self.last_results)

    def _on_disappearance_detected(self, early_results):
        self.last_results = early_results
        # ── Store disappearance frame so Resume starts from here ──
        self._disappearance_frame_idx = early_results.get('last_frame_idx', 0)
        self.results_btn.pack(side=tk.RIGHT)
        self.status_lbl.config(text="⚠️  Disappearance detected!")
        timestamp_text = early_results.get('timestamp_ocr', early_results['timestamp'])
        self.info_lbl.config(
            text=f"Object disappeared at: {timestamp_text}\n"
                 f"OCR timestamp extraction in progress...",
            foreground="red")
        if self._results_win is None or not self._results_win.winfo_exists():
            self._results_win = self.show_results_window(early_results)
        else:
            if hasattr(self._results_win, '_timestamp_var'):
                self._results_win._timestamp_var.set(f"Time of Disappearance:\n{timestamp_text}")

    def on_tracking_complete(self, results):
        self.start_btn.config(state=tk.NORMAL)
        self.upload_btn.config(state=tk.NORMAL)
        self.show_tracking_chk.config(state=tk.NORMAL)
        self.speed_slider.config(state=tk.NORMAL)

        self.current_frame_idx = results.get('last_frame_idx', self.current_frame_idx)

        if results.get('stopped'):
            self.is_paused = True
            self.start_btn.config(text="Resume Tracking")
            self.status_lbl.config(
                text=f"Tracking paused at frame {self.current_frame_idx}.")
            self.info_lbl.config(
                text="You can re-draw the box now if needed.", foreground="#89B4FA")
            return

        self.is_paused = False
        self.start_btn.config(text="Start Tracking")
        self.stop_btn.pack_forget()
        self.progress_var.set(100)

        if "error" in results:
            messagebox.showerror("Error", results["error"])
            self.status_lbl.config(text="Tracking failed.")
            return

        if results.get("disappeared"):
            self.last_results = results
            self.results_btn.pack(side=tk.RIGHT)

            # ── Resume will start from the disappearance frame ──
            self.current_frame_idx = results.get('last_frame_idx', 0)
            self.is_paused = True
            self.start_btn.config(text="Resume from Disappearance")

            timestamp_text = results.get('timestamp_ocr', results['timestamp'])
            self.info_lbl.config(
                text=f"Object disappeared at: {timestamp_text}\n"
                    f"Re-draw the box and click 'Resume from Disappearance' to continue.",
                foreground="red")
            win = self._results_win
            if win is not None and win.winfo_exists():
                if hasattr(win, '_timestamp_var'):
                    win._timestamp_var.set(f"Time of Disappearance:\n{timestamp_text}")
            else:
                self._results_win = self.show_results_window(results)
        else:
            self.is_paused = False
            self.current_frame_idx = 0

            # ── Auto-advance to next queued video if available ──
            next_idx = self.queue_index + 1
            if next_idx < self.queue_total:
                self.queue_index = next_idx
                next_file = self.video_queue[next_idx]
                self.status_lbl.config(
                    text=f"Video {self.queue_index}/{self.queue_total} done. "
                         f"Loading next: {os.path.basename(next_file)}...")
                self.info_lbl.config(text="")
                self.root.after(800, lambda f=next_file: self._advance_to_next(f))
            else:
                self.status_lbl.config(text="Tracking finished. Object never disappeared.")
                self.info_lbl.config(
                    text="Object remained in frame for the full video.", foreground="green")
                if self.queue_total > 1:
                    self.info_lbl.config(
                        text=f"All {self.queue_total} videos processed. "
                             "Object never disappeared in any file.", foreground="green")

    def _advance_to_next(self, filepath):
        """Called after a short delay to load the next queued video and
        auto-start tracking with the same bbox and settings as before."""
        prev_bbox   = self.selected_bbox  # remember user's drawn box
        prev_ocr    = self.ocr_bbox
        prev_speed  = int(self.speed_var.get())
        prev_show   = self.show_tracking_var.get()

        self._load_video_from_queue(filepath)

        # Re-apply bbox from previous video so user doesn't have to redraw
        if prev_bbox:
            self.selected_bbox = prev_bbox
            self.ocr_bbox      = prev_ocr
            # Draw the box visually on the new first frame
            x1, y1, x2, y2 = prev_bbox
            cx1 = int(x1 * self.scale_factor) + self.x_offset
            cy1 = int(y1 * self.scale_factor) + self.y_offset
            cx2 = int(x2 * self.scale_factor) + self.x_offset
            cy2 = int(y2 * self.scale_factor) + self.y_offset
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(
                cx1, cy1, cx2, cy2, outline='#00FF00', width=2)
            if prev_ocr:
                ox1, oy1, ox2, oy2 = prev_ocr
                ocx1 = int(ox1 * self.scale_factor) + self.x_offset
                ocy1 = int(oy1 * self.scale_factor) + self.y_offset
                ocx2 = int(ox2 * self.scale_factor) + self.x_offset
                ocy2 = int(oy2 * self.scale_factor) + self.y_offset
                if self.ocr_rect_id:
                    self.canvas.delete(self.ocr_rect_id)
                self.ocr_rect_id = self.canvas.create_rectangle(
                    ocx1, ocy1, ocx2, ocy2, outline='#FAB387', width=2, dash=(6, 3))
                self.ocr_region_lbl.config(text="OCR region set ✓", foreground="#A6E3A1")

            self.start_btn.config(state=tk.NORMAL)
            queue_info = f" [{self.queue_index + 1}/{self.queue_total}]"
            self.status_lbl.config(
                text=f"Video{queue_info} ready. Auto-starting tracking...")
            # Auto-start tracking after a brief pause so canvas renders
            self.root.after(600, self.start_tracking)

    def show_results_window(self, results):
        res_win = tk.Toplevel(self.root)
        res_win.title("Results - Disappearance Detected")
        res_win.geometry("800x450")
        res_win.minsize(600, 350)
        res_win.configure(bg="#1E1E2E")
        _apply_icon(res_win)

        timestamp_text = results.get('timestamp_ocr', results['timestamp'])

        # Store as StringVar on the window so on_tracking_complete can update
        # it later when OCR finishes (without opening a second window).
        res_win._timestamp_var = tk.StringVar(
            value=f"Time of Disappearance:\n{timestamp_text}")
        ttk.Label(res_win,
                  textvariable=res_win._timestamp_var,
                  font=("Segoe UI", 22, "bold"),
                  justify="center", anchor="center").pack(pady=15)

        frames_frame = ttk.Frame(res_win)
        frames_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        frames_frame.columnconfigure(0, weight=1)
        frames_frame.columnconfigure(1, weight=1)
        frames_frame.rowconfigure(1, weight=1)

        lbl_b_text = ttk.Label(frames_frame, text="Before Disappearance",
                               font=("Segoe UI", 16, "bold"), anchor="center")
        lbl_b_text.grid(row=0, column=0, pady=(0, 5))
        lbl_b_img = ttk.Label(frames_frame, anchor="center")
        lbl_b_img.grid(row=1, column=0, sticky="nsew", padx=10)

        lbl_a_text = ttk.Label(frames_frame, text="After Disappearance",
                               font=("Segoe UI", 16, "bold"), anchor="center")
        lbl_a_text.grid(row=0, column=1, pady=(0, 5))
        lbl_a_img = ttk.Label(frames_frame, anchor="center")
        lbl_a_img.grid(row=1, column=1, sticky="nsew", padx=10)

        orig_img_b = Image.fromarray(cv2.cvtColor(results['frame_before'], cv2.COLOR_BGR2RGB))
        # Use the breadcrumb-annotated frame for the 'After' panel when available
        _path_frame = results.get('frame_after_with_path')
        _after_src  = _path_frame if _path_frame is not None else results['frame_after']
        orig_img_a = Image.fromarray(cv2.cvtColor(_after_src, cv2.COLOR_BGR2RGB))

        ttk.Button(res_win, text="Export Results...", style="Accent.TButton",
                   command=lambda: self.export_results(results)).pack(pady=10)

        res_win.resize_timer = None

        def resize_images(event=None):
            if event and event.widget != frames_frame:
                return
            w = frames_frame.winfo_width() // 2 - 20
            h = frames_frame.winfo_height() - lbl_b_text.winfo_height() - 10
            if w <= 10 or h <= 10:
                return
            img_b = orig_img_b.copy()
            img_b.thumbnail((w, h), Image.Resampling.LANCZOS)
            res_win.photo_b = ImageTk.PhotoImage(img_b)
            lbl_b_img.config(image=res_win.photo_b)

            img_a = orig_img_a.copy()
            img_a.thumbnail((w, h), Image.Resampling.LANCZOS)
            res_win.photo_a = ImageTk.PhotoImage(img_a)
            lbl_a_img.config(image=res_win.photo_a)

        def on_resize(event):
            if res_win.resize_timer is not None:
                res_win.after_cancel(res_win.resize_timer)
            res_win.resize_timer = res_win.after(100, lambda: resize_images(event))

        frames_frame.bind("<Configure>", on_resize)
        return res_win

    def export_results(self, results):
        initial_dir = str(Path.home() / "Documents")
        export_base = filedialog.askdirectory(title="Select Export Directory",
                                              initialdir=initial_dir)
        if not export_base:
            return

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = Path(export_base) / f"DetectraResults_{timestamp}"

        try:
            os.makedirs(export_dir, exist_ok=True)
            cv2.imwrite(str(export_dir / "before_disappearance.jpg"), results['frame_before'])
            cv2.imwrite(str(export_dir / "after_disappearance.jpg"),  results['frame_after'])
            if 'frame_after_with_path' in results and results['frame_after_with_path'] is not None:
                cv2.imwrite(str(export_dir / "after_disappearance_path.jpg"),
                            results['frame_after_with_path'])
            messagebox.showinfo("Export Successful", f"Results exported to:\n{export_dir}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {e}")


# ─────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    from setup import FLAG_FILE, run_setup

    if not FLAG_FILE.exists():
        # First launch — run download setup
        run_setup()
    else:
        # Already set up — show splash on main thread
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

        _set_appusermodelid()
        root = tk.Tk()
        _apply_icon(root)
        SplashScreen(root)
        root.mainloop()