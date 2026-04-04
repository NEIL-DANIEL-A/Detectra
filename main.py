import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import cv2
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────
#  SPLASH SCREEN
# ─────────────────────────────────────────
class SplashScreen:
    def __init__(self, root):
        self.root = root
        self.root.overrideredirect(True)
        self.root.configure(bg="#1E1E2E")
        self.root.attributes('-topmost', True)

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

        tk.Label(self.root, text="v2.0.0",
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
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {e}"))
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
        root = tk.Tk()                             # open main window
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

        # Core state
        self.video_path = None
        self.first_frame_rgb = None
        self.detections = []
        self.selected_bbox = None
        self.tracker = tracker

        # Display state
        self.canvas_image = None
        self.photo = None
        self.scale_factor = 1.0
        self.x_offset = 0
        self.y_offset = 0

        # Drawing state
        self.rect_start_x = None
        self.rect_start_y = None
        self.rect_id = None
        self.resize_timer = None
        self.stop_event = None
        self.is_paused = False
        self.current_frame_idx = 0
        self.last_shown_frame_rgb = None
        self.last_results = None

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

        ttk.Label(self.options_frame, text="Speed:",
                  font=('Segoe UI', 11)).pack(side=tk.LEFT, padx=(20, 4))
        self.speed_var = tk.IntVar(value=2)
        self.speed_slider = ttk.Scale(self.options_frame, from_=1, to=10,
                                      orient=tk.HORIZONTAL, variable=self.speed_var,
                                      length=140, command=self._on_speed_change)
        self.speed_slider.pack(side=tk.LEFT)
        self.speed_lbl = ttk.Label(self.options_frame, text="2x",
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

    def upload_video(self):
        filetypes = (
            ('Video files', '*.mp4 *.avi *.mkv *.mov'),
            ('All files', '*.*')
        )
        filepath = filedialog.askopenfilename(title='Open a video', filetypes=filetypes)
        if not filepath:
            return

        self.video_path = filepath
        self.status_lbl.config(text=f"Loading: {os.path.basename(filepath)}...")
        self.root.update()

        frame, err = self.tracker.extract_first_frame(self.video_path)
        if err:
            messagebox.showerror("Error", err)
            self.status_lbl.config(text="Error loading video.")
            return

        self.first_frame_rgb = frame
        self.last_shown_frame_rgb = frame
        self.status_lbl.config(text="Detecting objects...")
        self.root.update()

        self.draw_frame()
        self.status_lbl.config(
            text="Click and drag to draw a bounding box around the object to track.")

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
        self.rect_id = None

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
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.rect_start_x, self.rect_start_y,
            self.rect_start_x, self.rect_start_y,
            outline='#00FF00', width=2)

    def on_drag(self, event):
        if self.last_shown_frame_rgb is None or self.rect_id is None:
            return
        self.canvas.coords(self.rect_id,
                           self.rect_start_x, self.rect_start_y,
                           event.x, event.y)

    def on_release(self, event):
        if self.last_shown_frame_rgb is None or self.rect_id is None:
            return

        end_x, end_y = event.x, event.y

        ix1 = (min(self.rect_start_x, end_x) - self.x_offset) / self.scale_factor
        iy1 = (min(self.rect_start_y, end_y) - self.y_offset) / self.scale_factor
        ix2 = (max(self.rect_start_x, end_x) - self.x_offset) / self.scale_factor
        iy2 = (max(self.rect_start_y, end_y) - self.y_offset) / self.scale_factor

        img_width  = self.last_shown_frame_rgb.shape[1]
        img_height = self.last_shown_frame_rgb.shape[0]

        ix1 = max(0, min(ix1, img_width))
        iy1 = max(0, min(iy1, img_height))
        ix2 = max(0, min(ix2, img_width))
        iy2 = max(0, min(iy2, img_height))

        if ix2 - ix1 > 10 and iy2 - iy1 > 10:
            self.selected_bbox = (int(ix1), int(iy1), int(ix2), int(iy2))
            self.start_btn.config(state=tk.NORMAL)
            self.status_lbl.config(text="Object selected. Click 'Start Tracking'.")
        else:
            self.selected_bbox = None
            self.start_btn.config(state=tk.DISABLED)
            self.canvas.delete(self.rect_id)
            self.rect_id = None
            self.status_lbl.config(text="Box too small, draw again.")

    def update_progress(self, current, total):
        pct = (current / total) * 100
        self.progress_var.set(pct)
        self.root.update_idletasks()

    def live_view_callback(self, frame_rgb, bbox):
        self.draw_frame(frame_rgb, bbox)
        self.root.update_idletasks()

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
        self.last_results = None
        self.start_btn.config(text="Start Tracking")
        self.stop_btn.pack_forget()
        self.results_btn.pack_forget()
        self.status_lbl.config(text="Tracking reset.")
        self.info_lbl.config(text="")
        if self.first_frame_rgb is not None:
            self.draw_frame(self.first_frame_rgb)

    def run_tracker_thread(self):
        def progress_cb(current, total):
            self.total_frames_est = total
            self.root.after(0, self.update_progress, current, total)

        def frame_cb(frame_rgb, bbox):
            self.root.after(0, self.live_view_callback, frame_rgb, bbox)

        callback_to_pass = frame_cb if self.show_tracking_var.get() else None
        frame_skip = int(self.speed_var.get())

        results = self.tracker.process_video(
            self.video_path, self.selected_bbox,
            progress_cb, callback_to_pass,
            stop_event=self.stop_event,
            frame_skip=frame_skip,
            start_frame=self.current_frame_idx
        )
        self.root.after(0, self.on_tracking_complete, results)

    def view_last_results(self):
        if self.last_results:
            self.show_results_window(self.last_results)

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
        self.current_frame_idx = 0
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
            self.status_lbl.config(text="Disappearance detected!")
            timestamp_text = results.get('timestamp_ocr', results['timestamp'])
            self.info_lbl.config(
                text=f"Object disappeared at: {timestamp_text}\n"
                     f"Click 'Show Last Results' to export snapshots.",
                foreground="red")
            self.show_results_window(results)
        else:
            self.status_lbl.config(text="Tracking finished. Object never disappeared.")
            self.info_lbl.config(
                text="Object remained in frame for the full video.", foreground="green")

    def show_results_window(self, results):
        res_win = tk.Toplevel(self.root)
        res_win.title("Results - Disappearance Detected")
        res_win.geometry("800x450")
        res_win.minsize(600, 350)
        res_win.configure(bg="#1E1E2E")

        timestamp_text = results.get('timestamp_ocr', results['timestamp'])

        ttk.Label(res_win,
                  text=f"Time of Disappearance: \n{timestamp_text}",
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
        orig_img_a = Image.fromarray(cv2.cvtColor(results['frame_after'],  cv2.COLOR_BGR2RGB))

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

        root = tk.Tk()
        SplashScreen(root)
        root.mainloop()