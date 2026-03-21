import tkinter as tk
from tkinter import ttk
import threading
import requests
import os
import zipfile
from pathlib import Path

# ─────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────
APP_DATA_DIR = Path(os.getenv("LOCALAPPDATA")) / "Detectra"
MODELS_DIR   = APP_DATA_DIR / "models"
EASYOCR_DIR  = APP_DATA_DIR / "easyocr"
FLAG_FILE    = APP_DATA_DIR / "setup_complete.flag"

DOWNLOADS = [
    {
        "name"  : "YOLOv8 Nano Model",
        "url"   : "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "dest"  : MODELS_DIR / "yolov8n.pt",
        "size"  : "6.2 MB",
        "zip"   : False
    },
    {
        "name"  : "EasyOCR Detection Model",
        "url"   : "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
        "dest"  : EASYOCR_DIR / "craft_mlt_25k.zip",
        "size"  : "80 MB",
        "zip"   : True,
        "extract_to": EASYOCR_DIR
    },
    {
        "name"  : "EasyOCR Recognition Model",
        "url"   : "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip",
        "dest"  : EASYOCR_DIR / "english_g2.zip",
        "size"  : "40 MB",
        "zip"   : True,
        "extract_to": EASYOCR_DIR
    }
]


# ─────────────────────────────────────────
#  SETUP SCREEN
# ─────────────────────────────────────────
class SetupScreen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Detectra — First Time Setup")
        self.root.geometry("560x520")
        self.root.resizable(False, False)
        self.root.configure(bg="#1E1E2E")
        self.root.attributes('-topmost', True)

        # Center window
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"560x520+{(sw-560)//2}+{(sh-520)//2}")

        self._build_ui()

        threading.Thread(target=self._run_setup, daemon=True).start()
        self.root.mainloop()

    def _build_ui(self):
        # ── Header ──
        tk.Label(self.root, text="🎯 Detectra",
                 font=("Segoe UI", 26, "bold"),
                 bg="#1E1E2E", fg="#89B4FA").pack(pady=(30, 4))

        tk.Label(self.root,
                 text="First-time setup — Downloading required AI models",
                 font=("Segoe UI", 10),
                 bg="#1E1E2E", fg="#A6ADC8").pack()

        tk.Label(self.root,
                 text="This will only happen once. Please stay connected to the internet.",
                 font=("Segoe UI", 9),
                 bg="#1E1E2E", fg="#6C7086").pack(pady=(2, 0))

        # ── Progress bars container ──
        self.container = tk.Frame(self.root, bg="#1E1E2E")
        self.container.pack(pady=20, padx=40, fill=tk.X)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Done.Horizontal.TProgressbar",
                         troughcolor="#313244", background="#A6E3A1")  # green when done
        style.configure("Active.Horizontal.TProgressbar",
                         troughcolor="#313244", background="#89B4FA")  # blue when active
        style.configure("Wait.Horizontal.TProgressbar",
                         troughcolor="#313244", background="#45475A")  # grey when waiting

        self.bars       = []
        self.pct_labels = []
        self.ico_labels = []

        for i, item in enumerate(DOWNLOADS):
            # Row: icon + name + size
            row = tk.Frame(self.container, bg="#1E1E2E")
            row.pack(fill=tk.X, pady=(12, 2))

            ico = tk.Label(row, text="⏳",
                           font=("Segoe UI", 11),
                           bg="#1E1E2E", fg="#6C7086")
            ico.pack(side=tk.LEFT, padx=(0, 6))
            self.ico_labels.append(ico)

            tk.Label(row, text=item["name"],
                     font=("Segoe UI", 10, "bold"),
                     bg="#1E1E2E", fg="#CDD6F4").pack(side=tk.LEFT)

            tk.Label(row, text=item["size"],
                     font=("Segoe UI", 9),
                     bg="#1E1E2E", fg="#6C7086").pack(side=tk.RIGHT)

            # Progress bar
            bar = ttk.Progressbar(self.container,
                                   style="Wait.Horizontal.TProgressbar",
                                   orient="horizontal",
                                   length=480, maximum=100,
                                   mode="determinate")
            bar.pack(fill=tk.X)
            self.bars.append(bar)

            # Percent label
            pct = tk.Label(self.container, text="Waiting...",
                           font=("Segoe UI", 9),
                           bg="#1E1E2E", fg="#6C7086", anchor="w")
            pct.pack(fill=tk.X, pady=(1, 0))
            self.pct_labels.append(pct)

        # ── Overall status ──
        sep = tk.Frame(self.root, bg="#313244", height=1)
        sep.pack(fill=tk.X, padx=40, pady=(10, 0))

        self.status_var = tk.StringVar(value="Preparing...")
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Segoe UI", 10),
                 bg="#1E1E2E", fg="#A6ADC8").pack(pady=10)

        # Overall progress bar
        self.overall_bar = ttk.Progressbar(self.root,
                                            style="Active.Horizontal.TProgressbar",
                                            orient="horizontal",
                                            length=480, maximum=len(DOWNLOADS),
                                            mode="determinate")
        self.overall_bar.pack(padx=40)

        tk.Label(self.root,
                 text="Do not close this window during setup.",
                 font=("Segoe UI", 8, "italic"),
                 bg="#1E1E2E", fg="#45475A").pack(pady=(8, 0))

    # ── Helpers ──
    def _set_bar(self, index, pct, text, style="Active.Horizontal.TProgressbar"):
        self.bars[index].config(style=style)
        self.bars[index]['value'] = pct
        self.pct_labels[index].config(text=text)
        self.root.update_idletasks()

    def _set_icon(self, index, icon, color="#6C7086"):
        self.ico_labels[index].config(text=icon, fg=color)
        self.root.update_idletasks()

    # ── Core setup logic ──
    def _run_setup(self):
        # Create all directories
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        EASYOCR_DIR.mkdir(parents=True, exist_ok=True)

        all_ok = True

        for i, item in enumerate(DOWNLOADS):
            self.status_var.set(f"Downloading {item['name']}...")
            self._set_icon(i, "⬇️", "#89B4FA")
            success = self._download(i, item)

            if success:
                self._set_icon(i, "✅", "#A6E3A1")
                self.overall_bar['value'] = i + 1
                self.root.update_idletasks()
            else:
                self._set_icon(i, "❌", "#F38BA8")
                all_ok = False
        # ✅ Fixed — import and call correctly
        if all_ok:
            FLAG_FILE.write_text("setup_complete")
            self.status_var.set("✅ All done! Launching Detectra...")
            import time; time.sleep(1.2)
            self.root.destroy()
            # Import and launch the main app from main.py
            from main import _launch_app
            _launch_app()
        else:
            self.status_var.set("❌ Some downloads failed. Check your internet and restart.")

    def _download(self, index, item):
        dest = Path(item["dest"])

        # Skip if already downloaded
        if dest.exists():
            self._set_bar(index, 100, "✅ Already downloaded", "Done.Horizontal.TProgressbar")
            return True

        try:
            resp  = requests.get(item["url"], stream=True, timeout=60)
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = (downloaded / total) * 100
                            mb  = downloaded / 1_000_000
                            tot_mb = total / 1_000_000
                            self._set_bar(index, pct,
                                          f"{pct:.1f}%   {mb:.1f} MB / {tot_mb:.1f} MB")

            # Extract zip if needed
            if item.get("zip") and dest.suffix == ".zip":
                self._set_bar(index, 100, "📦 Extracting...")
                with zipfile.ZipFile(dest, "r") as z:
                    z.extractall(item["extract_to"])
                dest.unlink()  # delete zip after extracting

            self._set_bar(index, 100, "✅ Complete", "Done.Horizontal.TProgressbar")
            return True

        except Exception as e:
            self._set_bar(index, 0, f"❌ Error: {e}")
            return False


def run_setup():
    SetupScreen()
    