import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import subprocess
import threading
import queue
import time
import pathlib
import json # For potential future config saving, not used actively now
import logging # Added for custom logging handler
import re # Added for parsing segment messages, if needed

# Attempt to import necessary libraries for Whisper and PyTorch
try:
    import whisper
    import torch
    from imageio_ffmpeg import get_ffmpeg_exe
except ImportError as e:
    messagebox.showerror("Missing Dependencies",
                         f"Critical libraries are missing: {e}. "
                         "Please ensure openai-whisper, torch, and imageio-ffmpeg are installed. "
                         "You might need to install PyTorch with CUDA support separately if you want GPU acceleration. "
                         "Run: pip install openai-whisper torch torchaudio torchvision imageio-ffmpeg")
    sys.exit(1)

# --- Constants ---
APP_NAME = "VerbaScript" # UPDATED APP NAME
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm")
DEFAULT_MODEL = "base"
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]
AVAILABLE_LANGUAGES = ["auto", "en", "es", "fr", "de", "it", "ja", "ko", "pt", "ru", "zh"]


# --- Helper Functions ---

def get_asset_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

FFMPEG_EXE = None
try:
    FFMPEG_EXE = get_ffmpeg_exe()
except Exception as e:
    print(f"Could not automatically find or download ffmpeg via imageio-ffmpeg: {e}")


BUNDLED_MODEL_PATH_ROOT = get_asset_path("assets/whisper_models")
os.makedirs(BUNDLED_MODEL_PATH_ROOT, exist_ok=True)


def format_timestamp_srt(seconds):
    assert seconds >= 0, "non-negative timestamp"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    seconds_val = milliseconds // 1_000
    milliseconds %= 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds_val:02d},{milliseconds:03d}"


def segments_to_srt(segments):
    srt_content = []
    for i, seg in enumerate(segments):
        start_time = format_timestamp_srt(seg['start'])
        end_time = format_timestamp_srt(seg['end'])
        text = seg['text'].strip()
        srt_content.append(f"{i + 1}\n{start_time} --> {end_time}\n{text}\n")
    return "\n".join(srt_content)


# --- Custom Logging Handler for Real-time Transcription ---
class GuiLogger(logging.Handler):
    def __init__(self, log_queue_ui):
        super().__init__()
        self.log_queue_ui = log_queue_ui
        # This pattern helps ensure we only capture segment lines.
        # Whisper logs segments like: "[00:00.000 --> 00:02.500] Hello world."
        self.segment_pattern = re.compile(r"^\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s+.*")

    def emit(self, record):
        # We are interested in INFO messages from the 'whisper' logger that match segment format
        if record.name == 'whisper' and record.levelno == logging.INFO:
            raw_msg = record.getMessage()
            if self.segment_pattern.match(raw_msg):
                self.log_queue_ui.put((raw_msg, "segment"))
            # else: # Optionally log other whisper INFO messages to the main log
            #     self.log_queue_ui.put((f"WHISPER: {raw_msg}", "debug"))


class TranscriberApp:
    def __init__(self, root_tk):
        self.root = root_tk
        self.root.title(APP_NAME) # Uses the updated APP_NAME
        self.root.geometry("700x750") 
        self.root.minsize(600, 650)

        self.input_path_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.language_var = tk.StringVar(value="auto")
        self.device_var = tk.StringVar(value="cpu")

        self.is_processing = False
        self.processing_thread = None
        self.log_queue = queue.Queue()
        
        self.gui_log_handler = GuiLogger(self.log_queue) 

        if FFMPEG_EXE is None:
             self.log_message("ERROR: ffmpeg executable not found. Ensure ffmpeg is installed or imageio-ffmpeg can download it.", "error")
        else:
            self.log_message(f"Found ffmpeg at: {FFMPEG_EXE}", "info")

        self.setup_ui()
        self.root.after(100, self.process_log_queue)

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        input_frame = ttk.LabelFrame(main_frame, text="Input Video(s)", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(input_frame, textvariable=self.input_path_var, width=60).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        ttk.Button(input_frame, text="Browse File", command=self.browse_input_file).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(input_frame, text="Browse Folder", command=self.browse_input_folder).pack(side=tk.LEFT)

        output_frame = ttk.LabelFrame(main_frame, text="Output SRT Folder", padding="10")
        output_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(output_frame, textvariable=self.output_folder_var, width=60).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        ttk.Button(output_frame, text="Browse", command=self.browse_output_folder).pack(side=tk.LEFT)

        settings_frame = ttk.Frame(main_frame, padding="5")
        settings_frame.pack(fill=tk.X, pady=5)

        model_frame = ttk.LabelFrame(settings_frame, text="Whisper Model", padding="5")
        model_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        model_combobox = ttk.Combobox(model_frame, textvariable=self.model_var, values=AVAILABLE_MODELS, state="readonly", width=15)
        model_combobox.pack(pady=5, padx=5)

        lang_frame = ttk.LabelFrame(settings_frame, text="Language", padding="5")
        lang_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        lang_combobox = ttk.Combobox(lang_frame, textvariable=self.language_var, values=AVAILABLE_LANGUAGES, width=15)
        lang_combobox.pack(pady=5, padx=5)
        lang_combobox.set("auto")

        device_frame = ttk.LabelFrame(settings_frame, text="Processing Device", padding="5")
        device_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Radiobutton(device_frame, text="CPU", variable=self.device_var, value="cpu").pack(anchor=tk.W)
        
        self.has_cuda = torch.cuda.is_available()
        gpu_radio_state = tk.NORMAL if self.has_cuda else tk.DISABLED
        gpu_radio_text = "GPU (CUDA)" if self.has_cuda else "GPU (CUDA not available)"
        ttk.Radiobutton(device_frame, text=gpu_radio_text, variable=self.device_var, value="cuda", state=gpu_radio_state).pack(anchor=tk.W)
        if self.has_cuda:
            self.device_var.set("cuda")
        else:
            self.device_var.set("cpu")

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=10)

        realtime_frame = ttk.LabelFrame(main_frame, text="Real-time Transcription Output", padding="10")
        realtime_frame.pack(expand=True, fill=tk.BOTH, pady=5)
        self.realtime_text = tk.Text(realtime_frame, height=7, wrap=tk.WORD, state=tk.DISABLED, relief=tk.SOLID, borderwidth=1)
        realtime_scrollbar = ttk.Scrollbar(realtime_frame, command=self.realtime_text.yview)
        self.realtime_text.config(yscrollcommand=realtime_scrollbar.set)
        realtime_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.realtime_text.pack(expand=True, fill=tk.BOTH)

        log_frame = ttk.LabelFrame(main_frame, text="Log & Status", padding="10")
        log_frame.pack(expand=True, fill=tk.BOTH, pady=5)
        self.log_text = tk.Text(log_frame, height=7, wrap=tk.WORD, state=tk.DISABLED, relief=tk.SOLID, borderwidth=1)
        log_scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(expand=True, fill=tk.BOTH)

        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.pack(fill=tk.X, pady=10)
        self.start_button = ttk.Button(control_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.stop_button = ttk.Button(control_frame, text="Stop Processing", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

    def browse_input_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("Video files", "*.mp4 *.mkv *.avi *.mov *.flv *.wmv *.webm"), ("All files", "*.*"))
        )
        if filepath:
            self.input_path_var.set(filepath)

    def browse_input_folder(self):
        folderpath = filedialog.askdirectory(title="Select Folder Containing Videos")
        if folderpath:
            self.input_path_var.set(folderpath)

    def browse_output_folder(self):
        folderpath = filedialog.askdirectory(title="Select Output Folder for SRT Files")
        if folderpath:
            self.output_folder_var.set(folderpath)

    def log_message(self, message, level="info"):
        self.log_queue.put((message, level))

    def clear_realtime_display(self):
        self.realtime_text.config(state=tk.NORMAL)
        self.realtime_text.delete(1.0, tk.END)
        self.realtime_text.config(state=tk.DISABLED)

    def _update_log_text(self, message, level):
        if level == "segment":
            self.realtime_text.config(state=tk.NORMAL)
            self.realtime_text.insert(tk.END, message + "\n")
            self.realtime_text.see(tk.END)
            self.realtime_text.config(state=tk.DISABLED)
        else:
            self.log_text.config(state=tk.NORMAL)
            tag_name = f"log_{level}"
            color = "black"
            if level == "error": color = "red"
            elif level == "warning": color = "orange"
            elif level == "success": color = "green"
            elif level == "debug": color = "gray"
            
            self.log_text.tag_configure(tag_name, foreground=color)
            self.log_text.insert(tk.END, f"[{level.upper()}] {message}\n", tag_name)
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)

    def process_log_queue(self):
        try:
            while True:
                message, level = self.log_queue.get_nowait()
                self._update_log_text(message, level)
        except queue.Empty:
            pass
        self.root.after(100, self.process_log_queue)

    def set_ui_state(self, processing):
        self.is_processing = processing
        self.start_button.config(state=tk.DISABLED if processing else tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL if processing else tk.DISABLED)
        
        widgets_to_toggle = []
        for frame_name_tuple in self.root.winfo_children(): # Iterate main_frame and its children
            if isinstance(frame_name_tuple, ttk.Frame): # This is main_frame
                for sub_child in frame_name_tuple.winfo_children():
                    # Check LabelFrames by their text attribute if they have one
                    try:
                        frame_text = sub_child.cget("text")
                        if frame_text in ["Input Video(s)", "Output SRT Folder", "Whisper Model", "Language", "Processing Device"]:
                            for widget in sub_child.winfo_children():
                                if isinstance(widget, (ttk.Entry, ttk.Button, ttk.Combobox, ttk.Radiobutton)):
                                    widgets_to_toggle.append(widget)
                    except tk.TclError: # Not all children are LabelFrames or have 'text'
                        pass
                    
                    # Handle settings_frame children directly as it's a simple Frame
                    if isinstance(sub_child, ttk.Frame): # e.g. settings_frame
                            for setting_widget_container in sub_child.winfo_children(): # e.g. model_frame (which is a LabelFrame)
                                if isinstance(setting_widget_container, ttk.LabelFrame):
                                    for widget in setting_widget_container.winfo_children():
                                        if isinstance(widget, (ttk.Entry, ttk.Button, ttk.Combobox, ttk.Radiobutton)):
                                            widgets_to_toggle.append(widget)


        for widget in widgets_to_toggle:
            try:
                if processing:
                    widget.config(state=tk.DISABLED)
                else:
                    if isinstance(widget, ttk.Radiobutton) and widget.cget("value") == "cuda" and not self.has_cuda:
                        widget.config(state=tk.DISABLED)
                    else:
                        widget.config(state=tk.NORMAL)
            except tk.TclError:
                pass


    def start_processing(self):
        if FFMPEG_EXE is None:
            messagebox.showerror("ffmpeg Error", "ffmpeg executable not found. Cannot proceed.")
            return

        input_path_str = self.input_path_var.get()
        output_folder_str = self.output_folder_var.get()

        if not input_path_str or not output_folder_str or not os.path.exists(output_folder_str):
            messagebox.showerror("Input Error", "Please check input/output paths.")
            return

        input_path = pathlib.Path(input_path_str)
        video_files = []
        if input_path.is_file():
            if input_path.suffix.lower() in VIDEO_EXTENSIONS: video_files.append(input_path)
            else: messagebox.showerror("Input Error", "Selected file is not a recognized video format."); return
        elif input_path.is_dir():
            video_files = [item for item in input_path.iterdir() if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS]
            if not video_files: messagebox.showerror("Input Error", "No video files found in folder."); return
        else: messagebox.showerror("Input Error", "Invalid input path."); return

        self.set_ui_state(True)
        self.progress_var.set(0)
        self.log_message("Starting processing...", "info")
        self.clear_realtime_display() 

        self.processing_thread = threading.Thread(
            target=self._process_videos_thread,
            args=(video_files, output_folder_str, self.model_var.get(), self.language_var.get(), self.device_var.get()),
            daemon=True
        )
        self.processing_thread.start()

    def stop_processing(self):
        if self.is_processing and self.processing_thread:
            self.log_message("Attempting to stop processing...", "warning")
            self.is_processing = False 
            self.stop_button.config(state=tk.DISABLED)

    def _process_videos_thread(self, video_files, output_folder_str, model_name, language, device_choice):
        start_time_total = time.time()
        files_processed, files_succeeded, files_failed = 0, 0, 0
        
        actual_device, use_fp16_for_whisper = "cpu", False
        if device_choice == "cuda":
            if self.has_cuda:
                try:
                    _ = torch.tensor([1.0, 2.0]).cuda() 
                    actual_device, use_fp16_for_whisper = "cuda", True
                    self.log_message("Using GPU (CUDA) for processing.", "info")
                except Exception as e:
                    self.log_message(f"CUDA selected, but error during test: {e}. Defaulting to CPU.", "warning")
            else:
                self.log_message("CUDA selected, but not available. Defaulting to CPU.", "warning")
        else:
            self.log_message("Using CPU for processing.", "info")
        
        whisper_logger = logging.getLogger("whisper")
        whisper_logger.setLevel(logging.INFO) 
        for handler in whisper_logger.handlers[:]:
            whisper_logger.removeHandler(handler)
        whisper_logger.addHandler(self.gui_log_handler) 
        whisper_logger.propagate = False


        try:
            self.log_message(f"Loading Whisper model '{model_name}' on '{actual_device}'. This may take time...", "info")
            model = whisper.load_model(model_name, device=actual_device, download_root=BUNDLED_MODEL_PATH_ROOT)
            self.log_message(f"Whisper model '{model_name}' loaded.", "success")
        except Exception as e:
            self.log_message(f"Error loading Whisper model '{model_name}': {e}", "error")
            self.is_processing = False
            self.root.after(0, lambda: self.set_ui_state(False))
            self.root.after(0, lambda: messagebox.showerror("Model Error", f"Failed to load Whisper model: {e}"))
            whisper_logger.removeHandler(self.gui_log_handler) 
            whisper_logger.propagate = True 
            return

        num_total_files = len(video_files)
        for i, video_path in enumerate(video_files):
            if not self.is_processing: self.log_message("Processing stopped by user.", "warning"); break
            
            self.root.after(0, self.clear_realtime_display) 
            self.log_message(f"Processing file {i+1}/{num_total_files}: {video_path.name}", "info")
            self.root.after(0, lambda p=i: self.progress_var.set((p / num_total_files) * 100))

            temp_audio_path = None
            try:
                self.log_message(f"Extracting audio from '{video_path.name}'...", "debug")
                temp_audio_path = pathlib.Path(output_folder_str) / f"_temp_audio_{video_path.stem}.wav"
                temp_audio_path.parent.mkdir(parents=True, exist_ok=True)
                ffmpeg_command = [FFMPEG_EXE, "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", str(temp_audio_path)]
                
                process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
                _, stderr = process.communicate()
                if process.returncode != 0:
                    self.log_message(f"ffmpeg error for {video_path.name}: {stderr.decode(errors='ignore')}", "error")
                    files_failed += 1; continue

                self.log_message(f"Audio extracted. Transcribing '{video_path.name}' (lang: {language})...", "info")
                lang_to_whisper = language if language != "auto" else None
                transcribe_options = {"fp16": use_fp16_for_whisper} 
                if lang_to_whisper: transcribe_options["language"] = lang_to_whisper
                
                result = model.transcribe(str(temp_audio_path), **transcribe_options) 
                
                self.log_message(f"Transcription complete for '{video_path.name}'.", "debug")
                srt_content = segments_to_srt(result["segments"])
                srt_filepath = pathlib.Path(output_folder_str) / (video_path.stem + ".srt")
                with open(srt_filepath, "w", encoding="utf-8") as f: f.write(srt_content)
                self.log_message(f"SRT file saved: {srt_filepath}", "success")
                files_succeeded +=1
            except Exception as e:
                self.log_message(f"Error processing {video_path.name}: {e}", "error")
                files_failed += 1
            finally:
                if temp_audio_path and temp_audio_path.exists():
                    try: os.remove(temp_audio_path)
                    except OSError as e_del: self.log_message(f"Could not delete temp audio '{temp_audio_path.name}': {e_del}", "warning")
                files_processed +=1
                self.root.after(0, lambda p=files_processed: self.progress_var.set((p / num_total_files) * 100 if num_total_files > 0 else 0))
        
        whisper_logger.removeHandler(self.gui_log_handler)
        whisper_logger.propagate = True

        total_time_taken = time.time() - start_time_total
        self.log_message("--- Processing Finished ---", "info")
        self.log_message(f"Total files: {num_total_files}, Succeeded: {files_succeeded}, Failed: {files_failed}", "info")
        self.log_message(f"Total time: {total_time_taken:.2f}s.", "info")
        
        self.is_processing = False
        self.root.after(0, lambda: self.set_ui_state(False))
        self.root.after(0, lambda: self.progress_var.set(100 if num_total_files > 0 else 0))
        
        summary_message = f"Finished: {files_succeeded}/{num_total_files} files.\nTime: {total_time_taken:.2f}s."
        if num_total_files == 0: summary_message = "No files were processed."
        
        if files_failed > 0 and files_succeeded > 0 :
            self.root.after(0, lambda: messagebox.showwarning("Processing Complete with Issues", summary_message + f"\n{files_failed} file(s) failed."))
        elif files_failed > 0 and files_succeeded == 0 and num_total_files > 0:
             self.root.after(0, lambda: messagebox.showerror("Processing Failed", summary_message + f"\nAll {files_failed} processed file(s) failed."))
        else: 
             self.root.after(0, lambda: messagebox.showinfo("Processing Complete", summary_message))


def main():
    if sys.platform == "win32":
        try: from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
        except Exception: pass
    root = tk.Tk()
    app = TranscriberApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
