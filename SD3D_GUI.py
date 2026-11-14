
"""
Stable Diffusion 3D Converter - SD3D Edition
- Deleted: Clarity, Clarity Blend, Enhance, Anti-Alias, Resize Mode, Batch, Gamma A, Gamma B, Text Fix
  (tiling gives real detail, so upscalers / clarity blends are no longer needed;
   Gamma A/B are fixed in inference_SD3D.py: GAMMA_A=1.0, GAMMA_B=1.0;
   Resize is fixed to stretch, Batch is fixed to 1)
- Kept: Mode, 3D Strength (was Baseline), Face Restore (None/Both Eyes), Seed, Output Type, Resolution, Steps
- Single-process: GUI starts ONE inference subprocess for the whole folder/file.
  StereoSpace loads ONCE, then inference loops all images sequentially inside that process.
  Old V75.6 started one subprocess PER image, so the model was reloaded for every image.
- Filename: ..._SD3D_S<steps>_R<resolution>.3D_SBS_... e.g. _SD3D_S20_R1344
- Layout: Conversion Options = 4 options per row; Facebook + About moved to top, left of Load, adjacent
- About: Easy3D layout, window narrowed to 600x400, image 260 wide = text column width, Close fully visible
- Buttons: Empty on the right of Cancel (clears Input + File List + Preview, does not delete files)
- Dead code removed V75.6: SCHEDULER_OPTIONS, GUI_SETTING_OPTIONS, apply_gui_setting/on_gui_setting_change, batch_val, stale save_config defaults
"""

import os
import sys
import json
import threading
import subprocess
import time
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

__VERSION__ = "V80.8 2026-11-14 "
APP_TITLE = "SD3D: Create high-resolution 3D SBS images with StereoSpace "
CONFIG_FILE = "preset/sd3d_gui_config.json"
PRESET_FILE = "preset/sd3d_presets.json"
LOG_FILE = "sd3d_debug.log"
SCRIPT_NAME = "inference_SD3D.py"
CONFIG_NAME = "stereospace_3070_8GB.yaml"
ABOUT_FACEBOOK_URL = "https://www.facebook.com/share/g/1KokTEPmCs/"  # 3D Stereo Lover group
ABOUT_GITHUB_URL = "https://github.com/KolaKater/SD3D/"

MODE_OPTIONS = {
    "Orig_Right (Right AI - Default)": "orig_right",
    "Orig_Left (Left AI)": "orig_left",
    "Both/Symmetric (Both AI)": "symmetric"
}
BASELINE_OPTIONS = {
    "3": 0.03,
    "4": 0.04,
    "5": 0.05,
    "6": 0.06,
    "7": 0.07,
    "8": 0.08,
    "9": 0.09,
    "10": 0.10,
    "11": 0.11,
    "12": 0.12,
    "13": 0.13,
    "14": 0.14,
    "15": 0.15,
}


# === V80.3 HD 3070 Options (added, original options kept) ===
HIGH_RES_OPTIONS = {
    "768- 1x1 Steps (Fast)": 768,
    "832- 2x2 Steps": 832,
    "896- 2x2 Steps": 896,
    "960- 2x2 Steps": 960,
    "1024- 2x2 Steps": 1024,
    "1152- 2x2 Steps": 1152,
    "1280- 2x2 Steps": 1280,
    "1344- 2x2 Steps (Recommend)": 1344,
    "1536- 3x3 Steps": 1536,
    "1792- 3x3 Steps (Slow)": 1792,
    "2048- 3x3 Steps (Slowest)": 2048,
}
STEPS_OPTIONS = {
    "8 - 3070 Fast (Default)": 8,
    "12 - Quality": 12,
    "16 - Quality+": 16,
    "20 - Quality++": 20,
    "25 - Original (Recommend)": 25,
    "30 - High Quality": 30,
    "35 - High Quality+": 35,
    "40 - Max Quality (Slow)": 40,
    "50 - Ultra Quality (Very Slow)": 50,
}


FACE_OPTIONS = {
    "1. None - Original": "none",
    "2. Gfpgan_Both - Both Eyes": "gfpgan_both"
}
# NOTE: old presets "2. Gfpgan - AI Eye Only" / "3. Gfpgan_Half" no longer exist; _migrate_display() falls back to None.


def load_seed_options():
    from pathlib import Path
    possible_paths = [
        Path(__file__).parent / "preset" / "seed_config.txt",
        Path(__file__).parent / "configs" / "seed_config.txt",
        Path(__file__).parent / "seed_config.txt",
    ]
    config_path = possible_paths[0]
    for p in possible_paths:
        if p.exists():
            config_path = p
            break
    default = {
        "Random (Default) - Different Each Time": "",
        "42 - Reproducible, Testing": "42",
        "123 - Reproducible, Stable": "123",
        "999 - Reproducible, High Variation": "999",
        "2025 - Reproducible, Current Year": "2025",
        "0 - Reproducible, Minimal": "0",
        "777 - Reproducible, Lucky": "777",
        "1234 - Reproducible, Sequence": "1234",
        "2024 - Reproducible, Last Year": "2024",
        "1655838429 - Your Fixed Seed": "1655838429"
    }
    try:
        if config_path.exists():
            opts = {}
            with open(config_path, "r", encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line or line.startswith("#"): continue
                    if "=" in line:
                        display, val = line.split("=", 1)
                        opts[display.strip()] = val.strip()
                    else:
                        opts[line] = line
            if opts:
                return opts
    except Exception as e:
        print(f"[Seed Config] Failed {e}")
    return default

SEED_OPTIONS = load_seed_options()

OUTPUT_TYPE_OPTIONS = {
    "1. Parallel (Default)": "parallel",
    "2. Cross-Eyed (XD Swap)": "xd",
    "3. Auto-Align Parallel": "align_parallel",
    "4. Auto-Align Cross-Eyed (XD)": "align_xd",
    "5. Auto-Align Parallel+XD (Both - 2 Files)": "align_both"
}


def _migrate_display(saved_value, options_dict):
    if not saved_value:
        return None
    if saved_value in options_dict:
        return saved_value
    low = saved_value.lower().strip()
    for k in options_dict.keys():
        if k.lower() == low:
            return k
    import re
    _m = re.match(r'^(\d+)', low)
    if _m:
        try:
            _v = int(_m.group(1))
            for k, v in options_dict.items():
                if isinstance(v, int) and v == _v:
                    return k
        except:
            pass
    s_stripped = re.sub(r'^\d+\.\s*', '', low)
    for k in options_dict.keys():
        k_low = k.lower()
        k_stripped = re.sub(r'^\d+\.\s*', '', k_low)
        if k_stripped == s_stripped:
            return k
    for k, v in options_dict.items():
        if isinstance(v, str) and v.lower() in low:
            return k
        if isinstance(v, (int,float)):
            try:
                if str(v) in low:
                    # for baseline etc
                    if k == str(v) or k.lower() == low:
                        return k
            except:
                pass
    # fuzzy: if saved contains internal value
    return None

class SD3DGUI:
    def __init__(self, root, use_dnd=False):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1280x820")
        # Icon
        try:
            base_dir = Path(__file__).parent
            for icon_name in ["SD3D.ico", "icon.ico", "app.ico", "3D.ico", "SD3D_GUI.ico"]:
                icon_path = base_dir / icon_name
                if icon_path.exists():
                    self.root.iconbitmap(str(icon_path))
                    break
        except Exception as e:
            print(f"[Icon] Could not load icon: {e}")
# V58.3 Gold Title Bar - User requested gold instead of teal
        try:
            import ctypes
            from ctypes import wintypes
            # Win11 DWMWA_CAPTION_COLOR = 35, DWMWA_TEXT_COLOR = 36, DWMWA_BORDER_COLOR = 34
            # Gold #FFD700 = RGB(255,215,0) -> COLORREF BGR 0x00D7FF
            DWMWA_CAPTION_COLOR = 35
            DWMWA_TEXT_COLOR = 36
            DWMWA_BORDER_COLOR = 34
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            if hwnd == 0:
                hwnd = self.root.winfo_id()
            # Try to set gold caption
            gold_ref = ctypes.c_int(55295)  # Gold
            black_ref = ctypes.c_int(0)  # Black text
            border_ref = ctypes.c_int(55295)
            try:
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_CAPTION_COLOR, ctypes.byref(gold_ref), ctypes.sizeof(gold_ref))
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_TEXT_COLOR, ctypes.byref(black_ref), ctypes.sizeof(black_ref))
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_BORDER_COLOR, ctypes.byref(border_ref), ctypes.sizeof(border_ref))
                print(f"[TitleBar] Gold #FFD700 set")
            except Exception as e:
                print(f"[TitleBar] Gold failed {e}, trying teal removal")
        except Exception as e:
            print(f"[TitleBar] {e}")

        self.root.minsize(1120, 750)
        self.root.resizable(True, True)
        self.dnd_available = use_dnd

        self.process = None
        self.pause_flag = False
        self.stop_flag = False
        self.config = self.load_config()

        self.gui_always_on_top_init = self.config.get("gui_always_on_top", False)
        try:
            if self.gui_always_on_top_init:
                self.root.after(500, lambda: self.root.attributes("-topmost", True))
        except:
            pass

        style = ttk.Style()
        # Use system-following theme: vista > winnative > clam
        try:
            style.theme_use("vista")
        except:
            try:
                style.theme_use("winnative")
            except:
                style.theme_use("clam")
        # Header follows system - no custom background, only font
        style.configure("TLabel", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 9))
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Small.TLabel", font=("Segoe UI", 8), foreground="#555555")

        main_frame = ttk.Frame(root, padding=(10,6))
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0,6))
        ttk.Label(title_frame, text=APP_TITLE, style="Header.TLabel").pack()

        io_frame = ttk.LabelFrame(main_frame, text="Input / Output", padding=(10,6))
        io_frame.pack(fill=tk.X, pady=(0,6))
        # Entry/combo columns expand and uniform like Conversion Options
        io_frame.columnconfigure(1, weight=1, uniform="io")
        io_frame.columnconfigure(3, weight=1, uniform="io")
        io_frame.columnconfigure(5, weight=1, uniform="io")
        # button columns weight 0 fixed

        # Row 0: Input: [entry] | Browse File | Browse Folder | Load | Save | Delete - all fixed width=13 like Browse File
        ttk.Label(io_frame, text="Input:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)
        self.input_var = tk.StringVar(value=self.config.get("input_path", ""))
        self.input_entry = ttk.Entry(io_frame, textvariable=self.input_var)
        self.input_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=4)
        self.input_entry.bind("<Button-3>", self.show_input_context)
        if self.dnd_available:
            try:
                self.input_entry.drop_target_register('DND_Files')
                self.input_entry.dnd_bind('<<Drop>>', self.on_drop_input)
            except:
                self.dnd_available = False

        ttk.Button(io_frame, text="Browse File", width=13, command=self.browse_input_file).grid(row=0, column=2, padx=5, pady=4)
        ttk.Button(io_frame, text="Browse Folder", width=13, command=self.browse_input_folder).grid(row=0, column=3, padx=5, pady=4, sticky=tk.W)

        # Preset buttons fixed same size as Browse File - aligned to RIGHT like Preset row
        self.preset_btn_frame = ttk.Frame(io_frame)
        self.preset_btn_frame.grid(row=0, column=4, columnspan=2, sticky=tk.E, padx=5, pady=2)
        self.preset_delete_btn = ttk.Button(self.preset_btn_frame, text="Delete", width=10, command=self.on_preset_delete)
        self.preset_delete_btn.pack(side=tk.RIGHT, padx=2)
        self.preset_save_btn = ttk.Button(self.preset_btn_frame, text="Save", width=10, command=self.on_preset_save)
        self.preset_save_btn.pack(side=tk.RIGHT, padx=2)
        self.preset_load_btn = ttk.Button(self.preset_btn_frame, text="Load", width=10, command=self.on_preset_load)
        self.preset_load_btn.pack(side=tk.RIGHT, padx=2)
        # Facebook + About: top row, left of Load, adjacent (pack RIGHT order -> visual: Facebook | About | Load)
        self.about_btn = ttk.Button(self.preset_btn_frame, text="About", width=10, command=self._show_about)
        self.about_btn.pack(side=tk.RIGHT, padx=2)
        self.facebook_btn = ttk.Button(self.preset_btn_frame, text="Facebook", width=10, command=self._open_facebook)
        self.facebook_btn.pack(side=tk.RIGHT, padx=2)

        # Row 1: Output: Default/Custom | Choose Folder fixed | [Custom path entry] | Preset: | [combo]
        ttk.Label(io_frame, text="Output:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=4)
        self.output_mode_var = tk.StringVar(value=self.config.get("output_mode", "default_new"))
        self.output_path_var = tk.StringVar(value=self.config.get("output_path", ""))
        
        self.rf = ttk.Frame(io_frame)
        self.rf.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(self.rf, text="Default (InputName_3D)", variable=self.output_mode_var, value="default_new", command=self.on_output_mode_change).pack(side=tk.LEFT)
        ttk.Radiobutton(self.rf, text="Custom", variable=self.output_mode_var, value="custom", command=self.on_output_mode_change).pack(side=tk.LEFT, padx=8)

        ttk.Button(io_frame, text="Choose Folder", width=13, command=self.browse_output_folder).grid(row=1, column=2, padx=5, pady=4)
        self.output_entry = ttk.Entry(io_frame, textvariable=self.output_path_var)
        self.output_entry.grid(row=1, column=3, sticky=tk.EW, padx=5, pady=4)

        ttk.Label(io_frame, text="Preset:", font=("Consolas", 9, "bold")).grid(row=1, column=4, sticky=tk.W, padx=5, pady=4)
        self.preset_var = tk.StringVar(value="")
        self.preset_combo = ttk.Combobox(io_frame, textvariable=self.preset_var, state="normal")
        self.preset_combo.grid(row=1, column=5, sticky=tk.EW, padx=5, pady=4)


        opts_frame = ttk.LabelFrame(main_frame, text="Conversion Options", padding=(10,8))

        opts_frame.pack(fill=tk.X, pady=(0,6))
        for c in [1,3,5,7]:
            opts_frame.columnconfigure(c, weight=1, uniform="opts")

        # Row 0, col 0-1: Mode
        mig = _migrate_display(self.config.get("mode_display", ""), MODE_OPTIONS)
        self.mode_var = tk.StringVar(value=mig if mig else "Orig_Right (Right AI - Default)")
        ttk.Label(opts_frame, text="Mode:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)
        self.mode_combo = ttk.Combobox(opts_frame, textvariable=self.mode_var, values=list(MODE_OPTIONS.keys()), state="readonly")
        self.mode_combo.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=4)
        self.mode_combo.bind("<<ComboboxSelected>>", self.save_config_event)

        ttk.Label(opts_frame, text="3D Strength:").grid(row=0, column=2, sticky=tk.W, padx=12, pady=4)
        self.baseline_var = tk.StringVar(value=self._fix_baseline_display(self.config.get("baseline_display", "6")))
        self.baseline_combo = ttk.Combobox(opts_frame, textvariable=self.baseline_var, values=list(BASELINE_OPTIONS.keys()), state="readonly")
        self.baseline_combo.grid(row=0, column=3, sticky=tk.EW, padx=5, pady=4)
        self.baseline_combo.bind("<<ComboboxSelected>>", self.save_config_event)


        # Row 0, col 4-5: Face Restore
        mig_face = _migrate_display(self.config.get("face_display", ""), FACE_OPTIONS)
        self.face_var = tk.StringVar(value=mig_face if mig_face else "1. None - Original")
        ttk.Label(opts_frame, text="Face Restore:").grid(row=0, column=4, sticky=tk.W, padx=12, pady=4)
        self.face_combo = ttk.Combobox(opts_frame, textvariable=self.face_var, values=list(FACE_OPTIONS.keys()), state="readonly")
        self.face_combo.grid(row=0, column=5, sticky=tk.EW, padx=5, pady=4)
        self.face_combo.bind("<<ComboboxSelected>>", self.save_config_event)

        # Row 0, col 6-7: Resolution (merged in from HD Fast Options)
        self.high_res_var = tk.StringVar(value=_migrate_display(self.config.get("high_res_display",""), HIGH_RES_OPTIONS) or "1344- 2x2 Steps (Recommend)")
        ttk.Label(opts_frame, text="Resolution:").grid(row=0, column=6, sticky=tk.W, padx=12, pady=4)
        self.high_res_combo = ttk.Combobox(opts_frame, textvariable=self.high_res_var, values=list(HIGH_RES_OPTIONS.keys()), state="readonly")
        self.high_res_combo.grid(row=0, column=7, sticky=tk.EW, padx=5, pady=4)
        self.high_res_combo.bind("<<ComboboxSelected>>", self.save_config_event)

        # Row 1, col 0-1: Steps (merged in from HD Fast Options)
        self.steps_var = tk.StringVar(value=_migrate_display(self.config.get("steps_display",""), STEPS_OPTIONS) or "8 - 3070 Fast (Default)")
        ttk.Label(opts_frame, text="Steps:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=4)
        self.steps_combo = ttk.Combobox(opts_frame, textvariable=self.steps_var, values=list(STEPS_OPTIONS.keys()), state="readonly")
        self.steps_combo.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=4)
        self.steps_combo.bind("<<ComboboxSelected>>", self.save_config_event)


        # Row 1, col 2-3: Seed
        mig_seed = _migrate_display(self.config.get("seed_display", ""), SEED_OPTIONS)
        self.seed_var = tk.StringVar(value=mig_seed if mig_seed else "Random (Default) - Different Each Time")
        ttk.Label(opts_frame, text="Seed:").grid(row=1, column=2, sticky=tk.W, padx=12, pady=4)
        self.seed_combo = ttk.Combobox(opts_frame, textvariable=self.seed_var, values=list(SEED_OPTIONS.keys()))
        self.seed_combo.grid(row=1, column=3, sticky=tk.EW, padx=5, pady=4)
        self.seed_combo.bind("<<ComboboxSelected>>", self.save_config_event)
        self.seed_combo.bind("<FocusOut>", self.save_config_event)
        self.seed_combo.bind("<Return>", self.save_config_event)
        

        # Row 1, col 4-5: Output Type
        mig_ot = _migrate_display(self.config.get("output_type_display", ""), OUTPUT_TYPE_OPTIONS)
        self.output_type_var = tk.StringVar(value=mig_ot if mig_ot else "1. Parallel (Default)")
        ttk.Label(opts_frame, text="Output Type:").grid(row=1, column=4, sticky=tk.W, padx=12, pady=4)
        self.output_type_combo = ttk.Combobox(opts_frame, textvariable=self.output_type_var, values=list(OUTPUT_TYPE_OPTIONS.keys()), state="readonly")
        self.output_type_combo.grid(row=1, column=5, sticky=tk.EW, padx=5, pady=4)
        self.output_type_combo.bind("<<ComboboxSelected>>", self.save_config_event)


        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(8,4))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=0)

        ttk.Label(btn_frame, text="Parallel=normal | XD=Left/Right Swap | Auto-Align fixes vertical misalignment | Align suffix added", style="Small.TLabel").grid(row=0, column=0, sticky=tk.W, padx=5)

        right_btn_frame = ttk.Frame(btn_frame)
        right_btn_frame.grid(row=0, column=1, sticky=tk.E)

        self.empty_btn = ttk.Button(right_btn_frame, text="Empty", width=10, command=self.clear_input_and_filelist)
        self.empty_btn.pack(side=tk.RIGHT, padx=(5,0))
        self.cancel_btn = ttk.Button(right_btn_frame, text="Cancel", width=10, command=self.cancel_conversion, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.RIGHT, padx=(5,0))

        self.suspend_btn = ttk.Button(right_btn_frame, text="Suspend", width=10, command=self.toggle_suspend, state=tk.DISABLED)
        self.suspend_btn.pack(side=tk.RIGHT, padx=5)

        self.start_btn = ttk.Button(right_btn_frame, text="Start", width=10, command=self.start_conversion)
        self.start_btn.pack(side=tk.RIGHT, padx=5)

        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=2)
        progress_frame.columnconfigure(0, weight=0)
        progress_frame.columnconfigure(1, weight=1)
        progress_frame.columnconfigure(2, weight=0)

        self.status_var = tk.StringVar(value="Progress: 0/0 (0%)")
        # Easy3D-style: Running.... animated dots, Done = clickable to open folder
        self._running_anim = False
        self._running_dots = 0
        self._progress_text = "Progress: 0/0 (0%)"
        self._last_output_dir = ""
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var, style="Small.TLabel", cursor="arrow")
        self.status_label.grid(row=0, column=0, sticky=tk.W, padx=5)
        self.status_label.bind("<Button-1>", lambda e: self._on_status_click())

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=(10,0))

        # Toggle panel
        toggle_frame = ttk.Frame(main_frame)
        toggle_frame.pack(fill=tk.X, pady=(4,2))
        ttk.Label(toggle_frame, text="View:", style="Small.TLabel").pack(side=tk.LEFT, padx=(5,10))

        self.show_log_var = tk.BooleanVar(value=self.config.get("show_log", True))
        self.show_preview_var = tk.BooleanVar(value=self.config.get("show_preview", True))
        self.show_filelist_var = tk.BooleanVar(value=self.config.get("show_filelist", True))
        self.gui_always_on_top_var = tk.BooleanVar(value=self.config.get("gui_always_on_top", False))

        ttk.Checkbutton(toggle_frame, text="Log", variable=self.show_log_var, command=self.toggle_panels).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(toggle_frame, text="Preview", variable=self.show_preview_var, command=self.toggle_panels).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(toggle_frame, text="File List", variable=self.show_filelist_var, command=self.toggle_panels).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(toggle_frame, text="GUI Always On Top", variable=self.gui_always_on_top_var, command=self.on_gui_always_on_top_change).pack(side=tk.LEFT, padx=15)

        # Bottom 4 panels
        self.bottom_frame = ttk.Frame(main_frame)
        self.bottom_frame.pack(fill=tk.BOTH, expand=True, pady=(2,0))
        self.bottom_frame.columnconfigure(0, weight=1)
        self.bottom_frame.columnconfigure(1, weight=1)
        self.bottom_frame.columnconfigure(2, weight=1)
        self.bottom_frame.rowconfigure(0, weight=1)

        self.log_frame = ttk.LabelFrame(self.bottom_frame, text="Log (sd3d_debug.log)", padding=5)
        self.log_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0,5))
        self.log_frame.columnconfigure(0, weight=1)
        self.log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(self.log_frame, wrap=tk.WORD, font=("Consolas", 8), bg="#1e1e1e", fg="#d4d4d4", insertbackground="white", height=12)
        scrollbar = ttk.Scrollbar(self.log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)

        self.preview_frame = ttk.LabelFrame(self.bottom_frame, text="Preview (Original)", padding=5)
        self.preview_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5)
        self.preview_frame.columnconfigure(0, weight=1)
        self.preview_frame.rowconfigure(0, weight=1)

        self.preview_label = ttk.Label(self.preview_frame, text="No Image", anchor=tk.CENTER)
        self.preview_label.grid(row=0, column=0, sticky=tk.NSEW)
        self.preview_label.bind("<Configure>", lambda e: self._resize_preview_if_needed())

        self.preview_pil_current = None
        self.preview_tk = None
        self._preview_pending_resize = False


        self.file_frame = ttk.LabelFrame(self.bottom_frame, text="File List", padding=5)
        self.file_frame.grid(row=0, column=2, sticky=tk.NSEW, padx=(5,0))
        self.file_frame.columnconfigure(0, weight=1)
        self.file_frame.rowconfigure(0, weight=1)

        self.file_listbox = tk.Listbox(self.file_frame, font=("Consolas", 8), bg="#1e1e1e", fg="#d4d4d4", selectbackground="#007acc", height=12)
        file_scroll = ttk.Scrollbar(self.file_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=file_scroll.set)
        self.file_listbox.grid(row=0, column=0, sticky=tk.NSEW)
        file_scroll.grid(row=0, column=1, sticky=tk.NS)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_list_select)
        self.file_listbox.bind("<Double-Button-1>", self.on_file_list_select)

        self.root.bind("<Button-1>", lambda e: self.root.focus())
        self.on_output_mode_change()
        self.log("GUI started - " + APP_TITLE)
        self.log(f"Settings loaded from {CONFIG_FILE}")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(400, self.refresh_file_list_and_preview)
        self.root.after(600, self.toggle_panels)
        self.root.after(200, self.refresh_preset_combo)

    def _fix_baseline_display(self, v):
        m = {"0.03":"3","0.04":"4","0.05":"5","0.06":"6","0.07":"7","0.08":"8","0.09":"9","0.10":"10","0.11":"11","0.12":"12","0.13":"13","0.14":"14","0.15":"15"}
        s = str(v).strip()
        if s in m:
            return m[s]
        return s if s in BASELINE_OPTIONS else "6"

    def _open_facebook(self):
        try:
            import webbrowser
            webbrowser.open(ABOUT_FACEBOOK_URL)
        except Exception as e:
            self.log(f"[Facebook] Open failed: {e}")

    def _show_about(self):
        """About - Easy3D layout (left text + links, right image, footer Close).
        FIX V75.5: window narrowed 680x360 -> 600x400 (was too wide, Close cut off).
        Image width = text column width (both 260): thumbnail (260, 300)."""
        try:
            _win = tk.Toplevel(self.root)
            _win.title("About SD3D")
            _win.geometry("600x400")
            _win.resizable(False, False)
            try:
                _win.transient(self.root)
                _win.grab_set()
            except Exception:
                pass
            try:
                _icon = None
                for _cand in (Path(__file__).parent / "images" / "SD3D.ico", Path(__file__).parent / "SD3D.ico"):
                    if _cand.exists():
                        _icon = _cand; break
                if _icon is not None:
                    _win.iconbitmap(str(_icon))
            except Exception as _ie:
                print(f"[About Icon] {_ie}")
            _BG = "#23272E"; _TEXT = "#E8EAED"; _MUTED = "#9AA0A8"; _LINK = "#6FB3FF"
            _win.configure(bg=_BG)
            _body = tk.Frame(_win, bg=_BG); _body.pack(fill=tk.BOTH, expand=True, padx=20, pady=(16, 8))
            _left = tk.Frame(_body, bg=_BG, width=260); _left.pack(side=tk.LEFT, fill=tk.Y, expand=False)
            _left.pack_propagate(False)
            tk.Label(_left, text="SD3D", font=("Segoe UI", 20, "bold"), bg=_BG, fg=_TEXT).pack(anchor=tk.W, pady=(0, 10))
            tk.Label(_left, text="It is not only a 3D converter,", font=("Segoe UI", 11), bg=_BG, fg=_TEXT).pack(anchor=tk.W)
            tk.Label(_left, text="but also a 3D creator.", font=("Segoe UI", 11), bg=_BG, fg=_TEXT).pack(anchor=tk.W, pady=(0, 14))
            tk.Label(_left, text="Written by Tony Lin.", font=("Segoe UI", 10), bg=_BG, fg=_MUTED).pack(anchor=tk.W, pady=(0, 16))
            tk.Frame(_left, bg="#3A3F47", height=1).pack(fill=tk.X, pady=(0, 16))
            _links = tk.Frame(_left, bg=_BG); _links.pack(anchor=tk.W, pady=(0, 16))
            def _open(url):
                try:
                    import webbrowser; webbrowser.open(url)
                except Exception as _e:
                    self.log(f"[About] Open failed: {_e}")
            _gh = tk.Label(_links, text="GitHub", fg=_LINK, bg=_BG, cursor="hand2", font=("Segoe UI", 10, "underline"))
            _gh.pack(side=tk.LEFT, padx=(0, 24)); _gh.bind("<Button-1>", lambda e: _open(ABOUT_GITHUB_URL))
            _fb = tk.Label(_links, text="Facebook", fg=_LINK, bg=_BG, cursor="hand2", font=("Segoe UI", 10, "underline"))
            _fb.pack(side=tk.LEFT); _fb.bind("<Button-1>", lambda e: _open(ABOUT_FACEBOOK_URL))
            tk.Label(_left, text=__VERSION__, font=("Segoe UI", 8), bg=_BG, fg=_MUTED).pack(anchor=tk.W)
            try:
                _bg_path = None
                for _s in (Path(__file__).parent / "images" / "SD3D-v2.png", Path(__file__).parent / "SD3D-v2.png"):
                    if _s.exists():
                        _bg_path = _s; break
                if _bg_path is not None:
                    from PIL import Image as _PILImg2, ImageTk as _PILImgTk2
                    _img = _PILImg2.open(_bg_path)
                    if _img.mode in ("RGBA", "LA") or (_img.mode == "P" and "transparency" in getattr(_img, "info", {})):
                        _rgba = _img.convert("RGBA"); _bgw = _PILImg2.new("RGB", _rgba.size, (35, 39, 46))
                        _bgw.paste(_rgba, mask=_rgba.split()[3]); _img = _bgw
                    else:
                        _img = _img.convert("RGB")
                    _img.thumbnail((260, 300), _PILImg2.LANCZOS)
                    self._about_bg_tk = _PILImgTk2.PhotoImage(_img)
                    tk.Label(_body, image=self._about_bg_tk, bg=_BG).pack(side=tk.RIGHT, padx=(20, 0), anchor=tk.N)
            except Exception as _be:
                print(f"[About BG] {_be}")
            _foot = tk.Frame(_win, bg=_BG); _foot.pack(fill=tk.X, padx=20, pady=(0, 12))
            ttk.Button(_foot, text="Close", width=12, command=_win.destroy).pack(anchor=tk.CENTER)
        except Exception:
            try:
                messagebox.showinfo("About SD3D", "SD3D\nIt is not only a 3D converter, but also a 3D creator.\nWritten by Tony Lin.")
            except Exception:
                pass

    def clear_input_and_filelist(self):
        """Empty: clear Input + File List + Preview. Does NOT delete files, does NOT clear params."""
        try: self.input_var.set("")
        except Exception: pass
        try: self.file_listbox.delete(0, tk.END)
        except Exception: pass
        try:
            self.preview_pil_current = None; self.preview_tk = None
            self.preview_label.configure(image="", text="No Image")
        except Exception: pass
        try: self.refresh_file_list_and_preview()
        except Exception: pass
        try: self.save_config()
        except Exception: pass
        try: self.log("[Empty] Input + File List cleared")
        except Exception: pass

    def on_gui_always_on_top_change(self):
        try:
            if self.gui_always_on_top_var.get():
                self.root.attributes("-topmost", True)
                self.root.lift()
                self.log("GUI Always On Top: ON")
            else:
                self.root.attributes("-topmost", False)
                self.log("GUI Always On Top: OFF")
            self.save_config()
        except Exception as e:
            print(f"always on top failed {e}")

    def toggle_panels(self):
        try:
            if self.show_log_var.get():
                self.log_frame.grid()
            else:
                self.log_frame.grid_remove()
            if self.show_preview_var.get():
                self.preview_frame.grid()
            else:
                self.preview_frame.grid_remove()
            if self.show_filelist_var.get():
                self.file_frame.grid()
            else:
                self.file_frame.grid_remove()
            self.save_config()
        except Exception as e:
            print(f"toggle panels failed {e}")

    def _resize_preview_if_needed(self):
        if self.preview_pil_current:
            if self._preview_pending_resize:
                return
            self._preview_pending_resize = True
            self.root.after(100, self._do_resize_preview)

    def _do_resize_preview(self):
        self._preview_pending_resize = False
        try:
            if not self.preview_pil_current:
                return
            w = self.preview_label.winfo_width()
            h = self.preview_label.winfo_height()
            if w < 20 or h < 20:
                w, h = 350, 300
            pil = self.preview_pil_current.copy()
            pil.thumbnail((max(100,w-10), max(100,h-10)), 3)
            from PIL import ImageTk
            self.preview_tk = ImageTk.PhotoImage(pil)
            self.preview_label.configure(image=self.preview_tk, text="")
        except Exception as e:
            print(f"preview resize failed {e}")

    def refresh_file_list_and_preview(self):
        try:
            inp = self.input_var.get().strip().strip('"')
            self.file_listbox.delete(0, tk.END)
            if not inp or not os.path.exists(inp):
                self.preview_label.configure(text="No Image", image="")
                return
            p = Path(inp)
            files = []
            if p.is_file():
                try:
                    parent_files = [f for f in p.parent.iterdir() if f.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}]
                    files = sorted(parent_files)[:200]
                except:
                    files = [p]
            else:
                exts = {'.jpg','.jpeg','.png','.bmp','.webp','.JPG','.JPEG','.PNG'}
                files = [f for f in p.iterdir() if f.is_file() and f.suffix in exts]
                files = sorted(files)
            for f in files[:500]:
                self.file_listbox.insert(tk.END, f.name)
            self._file_list_paths = files
            if files:
                if p.is_file():
                    try:
                        idx = [x.name for x in files].index(p.name)
                        self.file_listbox.selection_set(idx)
                        self.file_listbox.see(idx)
                        self.load_preview_image(p)
                    except:
                        self.load_preview_image(files[0])
                        self.file_listbox.selection_set(0)
                else:
                    self.file_listbox.selection_set(0)
                    self.load_preview_image(files[0])
        except Exception as e:
            print(f"refresh file list failed {e}")

    def load_preview_image(self, path):
        try:
            from PIL import Image
            pil = Image.open(path).convert("RGB")
            self.preview_pil_current = pil.copy()
            self._do_resize_preview()
            self.log(f"Preview: {Path(path).name} {pil.size}")
        except Exception as e:
            self.preview_label.configure(text=f"Failed: {e}")

    def on_file_list_select(self, event):
        try:
            sel = self.file_listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            if hasattr(self, '_file_list_paths') and idx < len(self._file_list_paths):
                self.load_preview_image(self._file_list_paths[idx])
        except Exception as e:
            print(f"file select failed {e}")

    def load_config(self):
        try:
            from pathlib import Path
            try:
                base = Path(__file__).parent
                cfg_path = base / CONFIG_FILE
            except:
                cfg_path = Path(CONFIG_FILE)
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except:
            pass
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except:
            pass
        return {}


    def load_presets(self):
        try:
            from pathlib import Path
            try:
                base = Path(__file__).parent
                p = base / PRESET_FILE
            except:
                p = Path(PRESET_FILE)
            if p.exists():
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
        except Exception as e:
            print(f"[Preset] load failed {e}")
        return {}

    def save_presets(self, data):
        try:
            from pathlib import Path
            try:
                base = Path(__file__).parent
                p = base / PRESET_FILE
            except:
                p = Path(PRESET_FILE)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.log(f"[Preset] Saved {len(data)} presets to {p}")
            return True
        except Exception as e:
            self.log(f"[Preset] save failed {e}")
            import traceback; traceback.print_exc()
            return False

    def refresh_preset_combo(self):
        try:
            presets = self.load_presets()
            names = sorted(presets.keys())
            if hasattr(self, 'preset_combo'):
                self.preset_combo['values'] = names
            if names:
                self.log(f"[Preset] {len(names)} presets: {', '.join(names)}")
        except Exception as e:
            self.log(f"[Preset] refresh failed {e}")

    def on_preset_save(self):
        current_name = self.preset_var.get().strip()
        import tkinter.simpledialog as simpledialog
        name = simpledialog.askstring("Save Preset", "Preset name (e.g. Face-Both,S1655):", initialvalue=current_name if current_name else "")
        if not name:
            return
        name = name.strip()
        if not name:
            from tkinter import messagebox
            messagebox.showwarning("Save Preset", "Name cannot be empty")
            return
        presets = self.load_presets()
        if name not in presets and len(presets) >= 20:
            from tkinter import messagebox
            messagebox.showwarning("Save Preset", f"Preset limit 20 reached!\nYou have {len(presets)} presets.\nDelete one first.")
            self.log(f"[Preset] Limit 20 reached")
            return
        try:
            opts = {
                "mode": self.mode_var.get(),
                "baseline": self.baseline_var.get(),
                "face_restore": self.face_var.get(),
                "seed": self.seed_var.get(),
                "output_type": self.output_type_var.get(),
                "high_res": self.high_res_var.get() if hasattr(self, 'high_res_var') else "",
                "steps": self.steps_var.get() if hasattr(self, 'steps_var') else "",
            }
            presets[name] = opts
            if self.save_presets(presets):
                self.preset_var.set(name)
                self.refresh_preset_combo()
                self.log(f"[Preset] Saved '{name}'")
                from tkinter import messagebox
                messagebox.showinfo("Save Preset", f"Preset '{name}' saved!")
        except Exception as e:
            self.log(f"[Preset] save error {e}")
            import traceback; traceback.print_exc()
            from tkinter import messagebox
            messagebox.showerror("Save Preset", f"Failed: {e}")

    def on_preset_load(self):
        name = self.preset_var.get().strip()
        if not name:
            from tkinter import messagebox
            messagebox.showwarning("Load Preset", "Select a preset")
            return
        presets = self.load_presets()
        if name not in presets:
            from tkinter import messagebox
            messagebox.showerror("Load Preset", f"Preset '{name}' not found")
            return
        opts = presets[name]
        try:
            if "mode" in opts: self.mode_var.set(opts["mode"])
            if "baseline" in opts: self.baseline_var.set(self._fix_baseline_display(opts["baseline"]))
            if "face_restore" in opts: self.face_var.set(opts["face_restore"])
            if "seed" in opts: self.seed_var.set(opts["seed"])
            if "output_type" in opts: self.output_type_var.set(opts["output_type"])
            if "high_res" in opts and opts["high_res"]: self.high_res_var.set(opts["high_res"])
            if "steps" in opts and opts["steps"]: self.steps_var.set(opts["steps"])
            self.log(f"[Preset] Loaded '{name}'")
            from tkinter import messagebox
            messagebox.showinfo("Load Preset", f"Preset '{name}' loaded!")
            self.save_config_event()
        except Exception as e:
            self.log(f"[Preset] load error {e}")
            import traceback; traceback.print_exc()
            from tkinter import messagebox
            messagebox.showerror("Load Preset", f"Failed: {e}")

    def on_preset_delete(self):
        name = self.preset_var.get().strip()
        if not name:
            from tkinter import messagebox
            messagebox.showwarning("Delete Preset", "Select a preset")
            return
        presets = self.load_presets()
        if name not in presets:
            from tkinter import messagebox
            messagebox.showerror("Delete Preset", f"Preset '{name}' not found")
            return
        from tkinter import messagebox
        if not messagebox.askyesno("Delete Preset", f"Delete preset '{name}'?"):
            return
        try:
            del presets[name]
            if self.save_presets(presets):
                self.preset_var.set("")
                self.refresh_preset_combo()
                self.log(f"[Preset] Deleted '{name}'")
                messagebox.showinfo("Delete Preset", f"Deleted '{name}'")
        except Exception as e:
            self.log(f"[Preset] delete error {e}")
            messagebox.showerror("Delete Preset", f"Failed: {e}")


    def save_config(self):
        try:
            from pathlib import Path
            try:
                base = Path(__file__).parent
                cfg_path = base / CONFIG_FILE
            except:
                cfg_path = Path(CONFIG_FILE)
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg = {
                "input_path": self.input_var.get() if hasattr(self, 'input_var') else "",
                "output_mode": self.output_mode_var.get() if hasattr(self, 'output_mode_var') else "default_new",
                "output_path": self.output_path_var.get() if hasattr(self, 'output_path_var') else "",
                "mode_display": self.mode_var.get() if hasattr(self, 'mode_var') else "Orig_Right (Right AI - Default)",
                "baseline_display": self.baseline_var.get() if hasattr(self, 'baseline_var') else "6",
                "face_display": self.face_var.get() if hasattr(self, 'face_var') else "2. Gfpgan_Both - Both Eyes",
                "seed_display": self.seed_var.get() if hasattr(self, 'seed_var') else "Random (Default) - different each time",
                "output_type_display": self.output_type_var.get() if hasattr(self, 'output_type_var') else "1. Parallel (Default)",
                "high_res_display": self.high_res_var.get() if hasattr(self, 'high_res_var') else "1344- 2x2 Steps (Recommend)",
                "steps_display": self.steps_var.get() if hasattr(self, 'steps_var') else "8 - 3070 Fast (Default)",
                "gui_always_on_top": self.gui_always_on_top_var.get() if hasattr(self, 'gui_always_on_top_var') else False,
                "show_log": self.show_log_var.get() if hasattr(self, 'show_log_var') else True,
                "show_preview": self.show_preview_var.get() if hasattr(self, 'show_preview_var') else True,
                "show_filelist": self.show_filelist_var.get() if hasattr(self, 'show_filelist_var') else True
            }
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Config Save Failed] {e}")

    def save_config_event(self, *args):
        if not hasattr(self, 'mode_var'):
            return
        self.save_config()

    def on_output_mode_change(self):
        try:
            if hasattr(self, 'output_entry'):
                if self.output_mode_var.get() == "custom":
                    self.output_entry.grid()
                    self.output_entry.configure(state="normal")
                else:
                    self.output_entry.grid_remove()
        except:
            pass
        self.save_config_event()

    def browse_input_file(self):
        p = filedialog.askopenfilename(title="Select Image File", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")])
        if p:
            self.input_var.set(p)
            self.save_config_event()
            self.root.after(100, self.refresh_file_list_and_preview)

    def browse_input_folder(self):
        p = filedialog.askdirectory(title="Select Input Folder")
        if p:
            self.input_var.set(p)
            self.save_config_event()
            self.root.after(100, self.refresh_file_list_and_preview)

    def browse_output_folder(self):
        p = filedialog.askdirectory(title="Select Output Folder")
        if p:
            self.output_path_var.set(p)
            self.output_mode_var.set("custom")
            self.on_output_mode_change()
            self.save_config_event()

    def on_drop_input(self, event):
        data = event.data
        files = self.root.tk.splitlist(data)
        if files:
            path = files[0].strip("{}")
            self.input_var.set(path)
            self.log(f"Dropped: {path}")
            self.save_config_event()
            self.root.after(100, self.refresh_file_list_and_preview)

    def show_input_context(self, event):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Paste Path", command=lambda: self.input_var.set(self.root.clipboard_get()))
        menu.tk_popup(event.x_root, event.y_root)

    def log(self, msg):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}\n"
        try:
            self.log_text.insert(tk.END, line)
            self.log_text.see(tk.END)
        except:
            pass
        try:
            with open(LOG_FILE, "a", encoding="utf-8", errors="ignore") as f:
                f.write(line)
        except:
            pass
        print(line, end="")

    def clear_log_file(self):
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(f"=== {APP_TITLE} Log - {time.strftime('%Y-%m-%d %H:%M:%S')} ===\nCleared on new run\n")
        except:
            pass
        try:
            self.log_text.delete(1.0, tk.END)
        except:
            pass

    def get_output_dir(self, input_path):
        input_path = Path(input_path)
        if self.output_mode_var.get() == "custom" and self.output_path_var.get().strip():
            out = Path(self.output_path_var.get().strip())
            if input_path.is_dir():
                if out.resolve() == input_path.resolve():
                    messagebox.showerror("Output Error", "If input is a folder, output must be different folder.")
                    return None
            return out
        else:
            if input_path.is_file():
                return input_path.parent / f"{input_path.stem}_3D"
            else:
                return input_path.parent / f"{input_path.name}_3D"

    def process_sbs_file(self, sbs_path, output_type):
        try:
            from PIL import Image
            sbs_path = Path(sbs_path)
            if not sbs_path.exists():
                return None
            img = Image.open(sbs_path)
            w, h = img.size
            half = w // 2
            left = img.crop((0, 0, half, h))
            right = img.crop((half, 0, w, h))

            needs_align = "align" in output_type
            is_xd = "xd" in output_type
            is_both = "both" in output_type

            base_name = sbs_path.stem
            suffix = sbs_path.suffix

            if needs_align:
                base_dir = Path(__file__).parent
                possible_exe = [
                    base_dir / "ffmpeg" / "bin" / "stereoautoalign030" / "StereoAutoAlign.exe",
                    base_dir / "ffmpeg" / "bin" / "StereoAutoAlign.exe",
                    Path("ffmpeg/bin/stereoautoalign030/StereoAutoAlign.exe"),
                ]
                align_exe = next((p for p in possible_exe if p.exists()), None)

                if align_exe:
                    self.log(f"[Auto-Align] Running {align_exe.name}...")
                    temp_dir = sbs_path.parent / "_temp_align"
                    temp_dir.mkdir(exist_ok=True)
                    left_temp = temp_dir / f"{base_name}_L_raw.png"
                    right_temp = temp_dir / f"{base_name}_R_raw.png"
                    left.save(left_temp)
                    right.save(right_temp)

                    try:
                        cmd = [str(align_exe), str(left_temp), str(right_temp), "10"]
                        # Hide CMD window that StereoAutoAlign would pop each image
                        si = None
                        creationflags = 0
                        if sys.platform == "win32":
                            try:
                                si = subprocess.STARTUPINFO()
                                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                                si.wShowWindow = subprocess.SW_HIDE
                                creationflags = subprocess.CREATE_NO_WINDOW
                            except:
                                si = None
                                creationflags = 0
                        subprocess.run(cmd, cwd=str(temp_dir), timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, startupinfo=si, creationflags=creationflags)
                        aligned_l = None
                        aligned_r = None
                        for f in temp_dir.glob("*Align_l.*"):
                            aligned_l = f
                            break
                        for f in temp_dir.glob("*Align_r.*"):
                            aligned_r = f
                            break
                        if not aligned_l:
                            for f in sbs_path.parent.glob("*_L_raw_Align_l.*"):
                                aligned_l = f
                                break
                        if not aligned_r:
                            for f in sbs_path.parent.glob("*_L_raw_Align_r.*"):
                                aligned_r = f
                                break
                        if aligned_l and aligned_r:
                            left_aligned = Image.open(aligned_l)
                            right_aligned = Image.open(aligned_r)
                            lw, lh = left_aligned.size
                            left_cropped = left_aligned.crop((1, 0, lw-1, lh))
                            rw, rh = right_aligned.size
                            right_cropped = right_aligned.crop((1, 0, rw-1, rh))
                            left = left_cropped
                            right = right_cropped
                            if left.size != right.size:
                                right = right.resize(left.size, Image.LANCZOS)
                        else:
                            self.log(f"[Auto-Align] WARNING: Aligned files not found")
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        for f in Path.cwd().glob("*_L_raw_Align_*"):
                            try:
                                f.unlink()
                            except:
                                pass
                    except Exception as e:
                        self.log(f"[Auto-Align] Failed: {e}")
                else:
                    self.log(f"[Auto-Align] StereoAutoAlign.exe not found, adding Align suffix only")

            def make_sbs(l, r):
                out = Image.new("RGB", (l.width + r.width, l.height))
                out.paste(l, (0,0))
                out.paste(r, (l.width,0))
                return out

            # FIX: User reported 1=Parallel was actually XD, 2=XD was Parallel - swap logic
            # Base SBS from inference is ORIG|GEN which is actually Cross-Eyed in VR, so:
            # Parallel (correct for VR) needs Swap, XD needs No-Swap
            if is_both:
                final_parallel = make_sbs(left, right)  # No swap = True Parallel V58.2
                name_parallel = f"{base_name}_Align{suffix}"
                path_parallel = sbs_path.parent / name_parallel
                final_parallel.save(path_parallel, "JPEG", quality=95, subsampling=0)
                self.log(f"[OutputType] Saved {output_type} Parallel (fixed): {name_parallel}")

                final_xd = make_sbs(right, left)  # Swap = True XD V58.2
                name_xd = f"{base_name}_Align_XD{suffix}"
                path_xd = sbs_path.parent / name_xd
                final_xd.save(path_xd, "JPEG", quality=95, subsampling=0)
                self.log(f"[OutputType] Saved {output_type} XD (fixed): {name_xd}")

                try:
                    if sbs_path.exists():
                        sbs_path.unlink()
                        self.log(f"[OutputType] Removed original: {sbs_path.name}")
                except:
                    pass
                return [path_parallel, path_xd]

            elif is_xd and needs_align:
                final = make_sbs(right, left)  # XD = swap V58.2
                new_name = f"{base_name}_Align_XD{suffix}"
            elif needs_align:
                final = make_sbs(left, right)  # Parallel = no swap V58.2
                new_name = f"{base_name}_Align{suffix}"
            elif is_xd:
                final = make_sbs(right, left)  # XD = swap V58.2
                new_name = f"{base_name}_XD{suffix}"
            else:
                final = make_sbs(left, right)  # Parallel = no swap V58.2
                new_name = f"{base_name}{suffix}"  # Keep original name, but content swapped to true Parallel
                # To avoid overwriting confusion, save over original with swapped content
                new_path = sbs_path.parent / new_name
                if new_path != sbs_path:
                    final.save(new_path, "JPEG", quality=95, subsampling=0)
                    try:
                        sbs_path.unlink()
                    except:
                        pass
                    self.log(f"[OutputType] Fixed Parallel (swapped): {new_name}")
                    return new_path
                else:
                    # Same path, need to overwrite with swapped version
                    # Save to temp then replace
                    temp_path = sbs_path.parent / f"_{base_name}_temp{suffix}"
                    final.save(temp_path, "JPEG", quality=95, subsampling=0)
                    try:
                        sbs_path.unlink()
                        temp_path.rename(sbs_path)
                        self.log(f"[OutputType] Fixed Parallel in place: {sbs_path.name}")
                    except Exception as e:
                        self.log(f"[OutputType] Failed to overwrite: {e}")
                        return temp_path
                    return sbs_path

            new_path = sbs_path.parent / new_name
            final.save(new_path, "JPEG", quality=95, subsampling=0)
            self.log(f"[OutputType] Saved {output_type}: {new_name}")
            try:
                if new_path != sbs_path and sbs_path.exists():
                    sbs_path.unlink()
                    self.log(f"[OutputType] Removed original: {sbs_path.name}")
            except:
                pass
            return new_path

        except Exception as e:
            self.log(f"[OutputType] Failed: {e}")
            import traceback
            self.log(traceback.format_exc())
            return None

    def _start_running_anim(self):
        self._running_anim = True
        self._running_dots = 0
        self.status_label.configure(cursor="arrow")
        self._tick_running_anim()
    def _tick_running_anim(self):
        if not getattr(self, "_running_anim", False):
            return
        self._running_dots = (self._running_dots % 4) + 1
        dots = "." * self._running_dots
        self.status_var.set(f"Running{dots}  {self._progress_text}")
        self.root.after(400, self._tick_running_anim)
    def _stop_running_anim(self):
        self._running_anim = False
    def _set_progress(self, p, cur, tot):
        self._progress_text = f"Progress: {cur}/{tot} ({p:.0f}%)"
        self.progress_var.set(p)
    def _set_status_done(self, output_dir, done_count, total_images):
        self._stop_running_anim()
        self._last_output_dir = str(output_dir)
        self.status_var.set(f"Done - Click to open folder ({done_count}/{total_images})")
        self.progress_var.set(100)
        self.status_label.configure(cursor="hand2")
    def _on_status_click(self):
        if getattr(self, "_last_output_dir", "") and self.status_var.get().startswith("Done"):
            self._open_output_folder()
    def _open_output_folder(self):
        import os as _os, subprocess as _sp, sys as _sys
        _d = getattr(self, "_last_output_dir", "")
        if not _d:
            return
        try:
            if _sys.platform.startswith("win"):
                _os.startfile(_d)
            elif _sys.platform == "darwin":
                _sp.Popen(["open", _d])
            else:
                _sp.Popen(["xdg-open", _d])
        except Exception as _e:
            self.log(f"[Open Folder] Failed: {_e}")
    def start_conversion(self):
        input_path = self.input_var.get().strip().strip('"')
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Input Error", "Please select valid input file or folder")
            return

        output_dir = self.get_output_dir(input_path)
        if not output_dir:
            return

        self.clear_log_file()
        self.log(f"=== Starting Conversion ===")
        self.log(f"Input: {input_path}")
        self.log(f"Output: {output_dir}")
        self.log(f"Mode: {self.mode_var.get()}")
        self.log(f"3D Strength: {self.baseline_var.get()}")
        self.log(f"[SD3D] Resolution {HIGH_RES_OPTIONS.get(self.high_res_var.get(), 1024) if hasattr(self, 'high_res_var') else 1024}")
        self.log(f"Output Type: {self.output_type_var.get()}")

        self.save_config()

        # FIX WinError 1114 shm.dll: use ABSOLUTE venv path based on this file, not relative cwd.
        # Relative "venv_..." + GUI's inherited PYTHONHOME/PATH can make venv python load system Python DLLs.
        _gui_dir = Path(__file__).resolve().parent
        venv_python = _gui_dir / "venv_SD3D" / "Scripts" / "python.exe"
        python_exe = str(venv_python) if venv_python.exists() else sys.executable

        mode_val = MODE_OPTIONS.get(self.mode_var.get(), "orig_right")
        baseline_val = BASELINE_OPTIONS.get(str(self.baseline_var.get()).strip(), 0.06)
        face_val = FACE_OPTIONS.get(self.face_var.get(), "none")
        # FIXED (removed from GUI): Gamma A/B=1.0 in inference_SD3D.py, Resize=stretch, Batch=1,
        # Clarity/Clarity Blend/Enhance/Anti-Alias/Text Fix/Line Fix=off. Not passed on cmdline.
        seed_display = self.seed_var.get().strip()
        seed_val = SEED_OPTIONS.get(seed_display, "")
        # Custom number typed directly
        if seed_display not in SEED_OPTIONS and seed_display:
            import re
            # If user typed "847291" or "Custom: 847291"
            m = re.search(r'-?\d+', seed_display)
            if m and "random" not in seed_display.lower():
                # Only use number if string is mostly numeric or contains custom
                if seed_display.replace(' ','').isdigit() or 'custom' in seed_display.lower() or len(seed_display) < 12:
                    seed_val = m.group(0)
            if "random" in seed_display.lower():
                seed_val = ""

        self.last_random_seed = None
        if not seed_val:
            # Random mode - generate a real seed so we can log and reproduce it
            import random
            self.last_random_seed = random.randint(0, 2147483647)
            seed_val = str(self.last_random_seed)
            # We will pass this seed to backend, but log it as random-generated
            # Backend will use this seed, so result is reproducible if you reuse it
        else:
            self.last_random_seed = None

        output_type_display = self.output_type_var.get()
        output_type_val = OUTPUT_TYPE_OPTIONS.get(output_type_display, "parallel")

        high_res_val = HIGH_RES_OPTIONS.get(self.high_res_var.get(), 1024) if hasattr(self, 'high_res_var') else 1024
        steps_val = STEPS_OPTIONS.get(self.steps_var.get(), 8) if hasattr(self, 'steps_var') else 8

        # Update yaml steps for 3070 fix
        try:
            from omegaconf import OmegaConf
            cfg_path = Path(CONFIG_NAME)
            if not cfg_path.exists():
                cfg_path = Path("configs") / CONFIG_NAME
            if cfg_path.exists():
                cfg = OmegaConf.load(str(cfg_path))
                cfg.num_inference_steps = steps_val
                # TILING: model always runs at 768x768 per tile, high_res is only the stitched output size.
                # Do NOT set train_width/train_height to high_res, or StereoSpace will try to run 2048 directly and OOM.
                cfg.data.train_width = 768
                cfg.data.train_height = 768
                try:
                    cfg.data.target_high_res = high_res_val
                except Exception:
                    pass
                OmegaConf.save(cfg, str(cfg_path))
                self.log(f"[SD3D] Resolution {high_res_val} | Steps {steps_val}")
        except Exception as e:
            self.log(f"[HD Config] Update failed {e}")

        cfg_path_for_cmd = None
        try:
            from pathlib import Path as _P
            for _cand in [_P(CONFIG_NAME), _P("configs") / CONFIG_NAME, _P(__file__).parent / CONFIG_NAME]:
                if _cand.exists():
                    cfg_path_for_cmd = _cand
                    break
            if cfg_path_for_cmd is None:
                cfg_path_for_cmd = _P(CONFIG_NAME)
        except:
            cfg_path_for_cmd = None
        # SEQUENTIAL: build the full file list first. One image at a time, in order, until ALL are done.
        _inp = Path(input_path)
        if _inp.is_file():
            file_list = [_inp]
        else:
            file_list = sorted([f for f in _inp.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}])
        if not file_list:
            messagebox.showerror("Input Error", "No image found")
            return
        self.log(f"[Single-process] {len(file_list)} image(s) in ONE inference run, StereoSpace loads once: {[f.name for f in file_list]}")
        base_cmd = [
            python_exe, "-u", SCRIPT_NAME,
            "--output", str(output_dir),
            "--config", str(cfg_path_for_cmd) if cfg_path_for_cmd else CONFIG_NAME,
            "--baseline", str(baseline_val),
            "--mode", mode_val,
            "--high_res", str(high_res_val),
            "--steps", str(steps_val),
            "--face_restore", face_val,
        ]
        if seed_val:
            base_cmd += ["--seed", seed_val]

        if self.last_random_seed is not None:
            self.log(f"[SEED] >>> Random mode - Generated seed: {self.last_random_seed}")
            self.log(f"[SEED] >>> If you like this result, copy this number and set Seed to: {self.last_random_seed}")
            self.log(f"[SEED] >>> Then future runs with Seed {self.last_random_seed} will give IDENTICAL result")
        else:
            self.log(f"[SEED] Fixed seed mode: {seed_val} - result will be reproducible")

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Output Error", f"Cannot create output dir: {e}")
            return

        self.stop_flag = False
        self.pause_flag = False
        self.start_btn.configure(state=tk.DISABLED)
        self.suspend_btn.configure(state=tk.NORMAL, text="Suspend")
        self.cancel_btn.configure(state=tk.NORMAL)
        self._last_output_dir = str(output_dir)
        self._progress_text = "Progress: 0/0 (0%)"
        self.progress_var.set(0)
        self._start_running_anim()

        def run_thread():
            # V75.7: ONE subprocess for the whole input (file or folder).
            # inference_SD3D.py: StereoSpace(config) once (line ~269), then
            # `for start in range(0, len(image_paths), bs)` loops all images.
            # Per-image Popen (V75.6) reloaded the model for EVERY image - that is fixed here.
            try:
                creationflags = 0
                if sys.platform == "win32":
                    creationflags = subprocess.CREATE_NO_WINDOW
                import os
                # FIX WinError 1114: clean env for venv subprocess.
                # GUI's own PYTHONHOME/PYTHONPATH/VIRTUAL_ENV/PATH (system Python) breaks venv torch DLL load.
                # Manual cmd works because its env is clean; GUI must rebuild it.
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                for _k in ["PYTHONHOME", "PYTHONPATH", "__PYVENV_LAUNCHER__", "PYTHON_EXECUTABLE"]:
                    env.pop(_k, None)
                _venv_dir = Path(__file__).resolve().parent / "venv_SD3D"
                env["VIRTUAL_ENV"] = str(_venv_dir)
                # PATH: venv Scripts first, drop other Python dirs that could shadow DLLs
                _old_path = env.get("PATH", "")
                _parts = [p for p in _old_path.split(";") if p and "python" not in p.lower() or str(_venv_dir).lower() in p.lower()]
                env["PATH"] = str(_venv_dir / "Scripts") + ";" + str(_venv_dir) + ";" + ";".join(_parts)
                total_images = len(file_list)
                # Pass the ORIGINAL input (file or folder) once, not one file at a time.
                # collect_image_paths() in inference handles both cases and returns the same list.
                cmd = base_cmd + ["--input", str(input_path)]
                self.log(f"Command: {' '.join(cmd)}")
                self.log(f"[Single-process] StereoSpace will load ONCE, then process {total_images} image(s) in order")
                self._saved_count = 0
                self.process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="ignore", bufsize=1,
                    creationflags=creationflags, cwd=os.getcwd(), env=env
                )
                def is_model_import_line(l):
                    if "Processed" in l or "-> Saved" in l or "Loading local weights" in l or "StereoSpace loaded OK" in l or "PROCESS" in l:
                        return False
                    return any(k in l for k in ["UNEXPECTED", "UserWarning", "pretrained", "warnings.warn", "huggingface"])
                for line in self.process.stdout:
                    if self.stop_flag:
                        break
                    while self.pause_flag and not self.stop_flag:
                        time.sleep(0.2)
                    self.log(line.rstrip())
                    clean = line.strip()
                    if not clean or is_model_import_line(clean):
                        continue
                    # FIX: inference never prints "Processed", it prints "[SD3D] Saved:" per image.
                    # Count Saved -> progress, so bar moves per image, not only at end.
                    if "[SD3D] Saved:" in clean:
                        try:
                            self._saved_count = getattr(self, "_saved_count", 0) + 1
                            _cur = min(self._saved_count, total_images)
                            _p = _cur / max(1, total_images) * 100
                            self.root.after(0, lambda p=_p, cur=_cur, tot=total_images: self._set_progress(p, cur, tot))
                        except Exception:
                            pass
                        continue
                    if "[SD3D] Stitching" in clean:
                        try:
                            self.root.after(0, lambda: setattr(self, "_progress_text", f"Stitching... ({getattr(self, '_saved_count', 0)}/{total_images})"))
                        except Exception:
                            pass
                        continue
                    if "Processed" in clean and "/" in clean:
                        import re
                        m = re.search(r'Processed\s+(\d+)/(\d+)', clean)
                        if m:
                            _p = int(m.group(1)); _tt = int(m.group(2)) or 1
                            _global = _p / _tt * 100 if _tt else 0
                            self.root.after(0, lambda p=_global, cur=_p, tot=_tt: self._set_progress(p, cur, tot))
                            continue
                    if "-> Saved" in clean:
                        fname = clean.split("-> Saved:")[-1].strip()
                        if len(fname) > 55:
                            fname = "..." + fname[-52:]
                        self.root.after(0, lambda f=fname: setattr(self, "_progress_text", f"Saving: {f}"))
                        continue
                self.process.wait()
                last_returncode = self.process.returncode
                if self.stop_flag:
                    self.root.after(0, lambda: (self._stop_running_anim(), self.status_var.set("Cancelled"), self.progress_var.set(0), self.status_label.configure(cursor="arrow")))
                    return
                if last_returncode != 0:
                    self.root.after(0, lambda: (self._stop_running_anim(), self.status_var.set(f"Finished code {last_returncode}"), self.status_label.configure(cursor="arrow")))
                    return
                # Post-processing (Output Type) ONCE, for ALL SBS files from this run.
                # inference already did GFPGAN Phase 2 on the SBS files; this only does XD/Align.
                done_count = total_images
                if output_type_val != "parallel":
                    self.log(f"=== Post-processing Output Type: {output_type_val} (all {total_images} file(s)) ===")
                    self.root.after(0, lambda: setattr(self, "_progress_text", f"Post-processing {output_type_val}..."))
                    _stems = {f.stem for f in file_list}
                    sbs_files = [f for f in output_dir.glob("*.jpg") if "3D_SBS" in f.name and "_XD" not in f.name and "_Align" not in f.name and any(f.name.startswith(_s) for _s in _stems)]
                    self.log(f"Found {len(sbs_files)} SBS file(s) for {output_type_val}")
                    for idx2, sbs_file in enumerate(sorted(sbs_files)):
                        if self.stop_flag:
                            break
                        self.process_sbs_file(sbs_file, output_type_val)
                self.root.after(0, lambda: self._set_status_done(output_dir, done_count, total_images))
            except Exception as e:
                self.log(f"[Exception] {e}")
                import traceback
                self.log(traceback.format_exc())
            finally:
                self.process = None
                self.root.after(0, lambda: self.start_btn.configure(state=tk.NORMAL))
                self.root.after(0, lambda: self.suspend_btn.configure(state=tk.DISABLED, text="Suspend"))
                self.root.after(0, lambda: self.cancel_btn.configure(state=tk.DISABLED))

        threading.Thread(target=run_thread, daemon=True).start()

    def toggle_suspend(self):
        if not self.process:
            return
        if not self.pause_flag:
            self.pause_flag = True
            self.suspend_btn.configure(text="Resume")
            self._stop_running_anim()
            self.status_var.set("Suspended")
            self.status_label.configure(cursor="arrow")
            self.log("Suspended")
        else:
            self.pause_flag = False
            self.suspend_btn.configure(text="Suspend")
            self._start_running_anim()
            self.log("Resumed")

    def cancel_conversion(self):
        if not self.process:
            return
        self.stop_flag = True
        self.pause_flag = False
        self.log("Cancelling...")
        try:
            self.process.terminate()
            time.sleep(0.5)
            if self.process.poll() is None:
                self.process.kill()
        except Exception as e:
            self.log(f"[Cancel Failed] {e}")
        self._stop_running_anim()
        self.status_var.set("Cancelling...")

    def on_close(self):
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("Quit", "Conversion running, quit and cancel?"):
                return
            self.cancel_conversion()
            time.sleep(0.5)
        self.save_config()
        self.root.destroy()

def main():
    dnd = False
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
        dnd = True
    except:
        root = tk.Tk()
    app = SD3DGUI(root, use_dnd=dnd)
    root.mainloop()

if __name__ == "__main__":
    main()
