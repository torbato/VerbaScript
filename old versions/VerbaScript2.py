import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import subprocess
import threading
import queue
import time
import pathlib
import json # For potential future config saving
import logging 
import re 
import multiprocessing # Added for better process control

# Attempt to import necessary libraries
try:
    import whisper # Keep for type hinting if needed, actual import in child process
    import torch
    from imageio_ffmpeg import get_ffmpeg_exe
except ImportError as e:
    messagebox.showerror("Missing Dependencies",
                         f"Critical libraries are missing: {e}. "
                         "Please ensure openai-whisper, torch, and imageio-ffmpeg are installed. "
                         "You might need to install PyTorch with CUDA support separately for GPU acceleration. "
                         "Run: pip install openai-whisper torch torchaudio torchvision imageio-ffmpeg")
    sys.exit(1)

# --- Constants ---
APP_VERSION = "v0.7"
APP_NAME_BASE = "VerbaScript"
APP_NAME = f"{APP_NAME_BASE} {APP_VERSION}" 
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm")
DEFAULT_MODEL = "base"
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]
AVAILABLE_LANGUAGES_WHISPER = ["auto", "en", "es", "fr", "de", "it", "ja", "ko", "pt", "ru", "zh"] 

# --- Helper Functions ---
def get_asset_path(relative_path):
    try: base_path = sys._MEIPASS
    except AttributeError: base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

FFMPEG_EXE = None
try: FFMPEG_EXE = get_ffmpeg_exe()
except Exception as e: print(f"Could not automatically find or download ffmpeg: {e}")

BUNDLED_MODEL_PATH_ROOT = get_asset_path("assets/whisper_models")
os.makedirs(BUNDLED_MODEL_PATH_ROOT, exist_ok=True)

def format_timestamp_srt(seconds):
    assert seconds >= 0, "non-negative timestamp"
    ms = round(seconds * 1000.0)
    hr = ms // 3_600_000; ms %= 3_600_000
    m = ms // 60_000; ms %= 60_000
    s = ms // 1_000; ms %= 1_000
    return f"{hr:02d}:{m:02d}:{s:02d},{ms:03d}"

def segments_to_srt(segments):
    return "\n".join(
        f"{i+1}\n{format_timestamp_srt(s['start'])} --> {format_timestamp_srt(s['end'])}\n{s['text'].strip()}\n"
        for i, s in enumerate(segments)
    )

# --- Target function for multiprocessing ---
def run_whisper_transcription_process(temp_audio_path_str, output_folder_str, video_stem, model_name, lang_whisper, device_choice, use_fp16, result_queue):
    """
    This function runs in a separate process to perform Whisper transcription.
    It loads the model, transcribes, and saves the SRT.
    Puts True/False into result_queue indicating success/failure.
    """
    try:
        # Import whisper here as it's needed in this separate process
        import whisper 
        
        print(f"[Whisper Process] Starting for {video_stem}, Model: {model_name}, Device: {device_choice}")

        model = whisper.load_model(model_name, device=device_choice, download_root=BUNDLED_MODEL_PATH_ROOT)
        
        transcribe_options = {"fp16": use_fp16, "verbose": False} 
        if lang_whisper != "auto": 
            transcribe_options["language"] = lang_whisper
        
        result = model.transcribe(temp_audio_path_str, **transcribe_options)
        
        srt_content = segments_to_srt(result["segments"])
        srt_filepath = pathlib.Path(output_folder_str) / (video_stem + ".srt")
        with open(srt_filepath, "w", encoding="utf-8") as f:
            f.write(srt_content)
        
        print(f"[Whisper Process] SRT file saved: {srt_filepath}")
        result_queue.put(True) 
        return True
    except Exception as e:
        print(f"[Whisper Process] Error during transcription for {video_stem}: {e}")
        import traceback
        traceback.print_exc() 
        result_queue.put(False) 
        return False


class TranscriberApp:
    def __init__(self, root_tk):
        self.root = root_tk
        self.root.title(APP_NAME) 
        self.root.geometry("700x650") # Adjusted height after progress bar removal
        self.root.minsize(600, 550) # Adjusted min height

        self.input_path_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.language_whisper_var = tk.StringVar(value="auto") 
        self.device_var = tk.StringVar(value="cpu")

        self.is_processing = False 
        self.processing_thread = None
        self.ffmpeg_process = None 
        self.whisper_mp_process = None 
        self.log_queue = queue.Queue()

        if FFMPEG_EXE is None:
             self.log_message("ffmpeg executable not found. Cannot proceed. Please install ffmpeg or check imageio-ffmpeg installation.", "error") 
        else:
            self.log_message(f"Found ffmpeg at: {FFMPEG_EXE}", "info")

        self.setup_menu()
        self.setup_ui()
        self.root.after(100, self.process_log_queue)

    def setup_menu(self):
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        self.input_frame = ttk.LabelFrame(self.main_frame, text="Input Video(s)", padding="10")
        self.input_frame.pack(fill=tk.X, pady=5)
        self.input_entry = ttk.Entry(self.input_frame, textvariable=self.input_path_var, width=60)
        self.input_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        self.browse_file_button = ttk.Button(self.input_frame, text="Browse File", command=self.browse_input_file)
        self.browse_file_button.pack(side=tk.LEFT, padx=(0,5))
        self.browse_folder_button = ttk.Button(self.input_frame, text="Browse Folder", command=self.browse_input_folder)
        self.browse_folder_button.pack(side=tk.LEFT)

        self.output_frame = ttk.LabelFrame(self.main_frame, text="Output SRT Folder", padding="10")
        self.output_frame.pack(fill=tk.X, pady=5)
        self.output_entry = ttk.Entry(self.output_frame, textvariable=self.output_folder_var, width=60)
        self.output_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        self.browse_output_button = ttk.Button(self.output_frame, text="Browse", command=self.browse_output_folder)
        self.browse_output_button.pack(side=tk.LEFT)

        self.settings_frame = ttk.Frame(self.main_frame, padding="5")
        self.settings_frame.pack(fill=tk.X, pady=5)

        self.model_frame = ttk.LabelFrame(self.settings_frame, text="Whisper Model", padding="5")
        self.model_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.model_combobox = ttk.Combobox(self.model_frame, textvariable=self.model_var, values=AVAILABLE_MODELS, state="readonly", width=15)
        self.model_combobox.pack(pady=5, padx=5)

        self.lang_whisper_frame = ttk.LabelFrame(self.settings_frame, text="Transcription Language", padding="5")
        self.lang_whisper_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.lang_whisper_combobox = ttk.Combobox(self.lang_whisper_frame, textvariable=self.language_whisper_var, values=AVAILABLE_LANGUAGES_WHISPER, width=15)
        self.lang_whisper_combobox.pack(pady=5, padx=5)
        self.lang_whisper_combobox.set("auto")

        self.device_frame = ttk.LabelFrame(self.settings_frame, text="Processing Device", padding="5")
        self.device_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.cpu_radio_button = ttk.Radiobutton(self.device_frame, text="CPU", variable=self.device_var, value="cpu")
        self.cpu_radio_button.pack(anchor=tk.W)
        
        self.has_cuda = torch.cuda.is_available()
        gpu_radio_state = tk.NORMAL if self.has_cuda else tk.DISABLED
        gpu_radio_text = "GPU (CUDA)" if self.has_cuda else "GPU (CUDA not available)"
        self.gpu_radio_button = ttk.Radiobutton(self.device_frame, text=gpu_radio_text, variable=self.device_var, value="cuda", state=gpu_radio_state)
        self.gpu_radio_button.pack(anchor=tk.W)
        if self.has_cuda: self.device_var.set("cuda")
        else: self.device_var.set("cpu")

        # Progress bar and its variable removed
        # self.progress_var = tk.DoubleVar()
        # self.progress_bar = ttk.Progressbar(self.main_frame, variable=self.progress_var, maximum=100)
        # self.progress_bar.pack(fill=tk.X, pady=10)

        self.log_frame = ttk.LabelFrame(self.main_frame, text="Log & Status", padding="10") 
        self.log_frame.pack(expand=True, fill=tk.BOTH, pady=10) # Added pady for spacing
        self.log_text = tk.Text(self.log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED, relief=tk.SOLID, borderwidth=1)
        log_scrollbar = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(expand=True, fill=tk.BOTH)

        control_frame = ttk.Frame(self.main_frame, padding="5")
        control_frame.pack(fill=tk.X, pady=10)
        self.start_button = ttk.Button(control_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.stop_button = ttk.Button(control_frame, text="Stop Processing", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
    def browse_input_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("Video files", "*.mp4 *.mkv *.avi *.mov *.flv *.wmv *.webm"), 
                       ("All files", "*.*"))
        )
        if filepath: self.input_path_var.set(filepath)

    def browse_input_folder(self):
        folderpath = filedialog.askdirectory(title="Select Folder Containing Videos")
        if folderpath: self.input_path_var.set(folderpath)

    def browse_output_folder(self):
        folderpath = filedialog.askdirectory(title="Select Output Folder for SRT Files")
        if folderpath: self.output_folder_var.set(folderpath)

    def log_message(self, message, level="info"):
        self.log_queue.put((message, level))

    def _update_log_text(self, message, level):
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
        except queue.Empty: pass
        self.root.after(100, self.process_log_queue)

    def set_ui_state(self, processing):
        self.is_processing = processing 
        self.start_button.config(state=tk.DISABLED if processing else tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL if processing else tk.DISABLED)
        
        if hasattr(self, 'main_frame') and self.main_frame.winfo_exists():
            for child in self.main_frame.winfo_children():
                if isinstance(child, (ttk.LabelFrame, ttk.Frame)):
                    for widget in child.winfo_children():
                        if isinstance(widget, (ttk.Entry, ttk.Button, ttk.Combobox, ttk.Radiobutton)):
                            if widget not in [self.start_button, self.stop_button]:
                                try:
                                    new_state = tk.DISABLED if processing else tk.NORMAL
                                    if isinstance(widget, ttk.Radiobutton) and widget.cget("value") == "cuda" and not self.has_cuda and not processing:
                                        new_state = tk.DISABLED
                                    widget.config(state=new_state)
                                except tk.TclError: pass
                elif isinstance(child, (ttk.Entry, ttk.Button, ttk.Combobox, ttk.Radiobutton)):
                     # Exclude progress_bar as it's removed
                     if child not in [self.start_button, self.stop_button]: 
                        try: child.config(state=tk.DISABLED if processing else tk.NORMAL)
                        except tk.TclError: pass

    def start_processing(self):
        if FFMPEG_EXE is None:
            messagebox.showerror("ffmpeg Error", "ffmpeg executable not found. Cannot proceed.")
            return
        input_path_str = self.input_path_var.get()
        output_folder_str = self.output_folder_var.get()
        if not input_path_str: messagebox.showerror("Input Error", "Please select an input video file or folder."); return
        if not output_folder_str: messagebox.showerror("Input Error", "Please select an output folder."); return
        if not os.path.exists(output_folder_str): messagebox.showerror("Input Error", "Output folder does not exist."); return

        input_path = pathlib.Path(input_path_str)
        video_files = []
        if input_path.is_file():
            if input_path.suffix.lower() in VIDEO_EXTENSIONS: video_files.append(input_path)
            else: messagebox.showerror("Input Error", f"Selected file '{input_path.name}' is not a recognized video format."); return
        elif input_path.is_dir():
            video_files = [item for item in input_path.iterdir() if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS]
            if not video_files: messagebox.showerror("Input Error", "No video files found in the selected folder."); return
        else: messagebox.showerror("Input Error", "Invalid input path."); return

        self.set_ui_state(True) 
        # self.progress_var.set(0.0) # Removed
        self.log_message("Starting processing...", "info")

        self.processing_thread = threading.Thread(
            target=self._process_videos_thread,
            args=(video_files, output_folder_str, self.model_var.get(), self.language_whisper_var.get(), self.device_var.get()),
            daemon=True
        )
        self.processing_thread.start()

    def stop_processing(self):
        self.log_message("Stop request received. Attempting to halt current operations.", "warning")
        self.is_processing = False 
        
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None: 
            self.log_message("Terminating active ffmpeg process...", "debug")
            try: self.ffmpeg_process.terminate(); self.ffmpeg_process.wait(timeout=0.5)
            except subprocess.TimeoutExpired: self.ffmpeg_process.kill()
            except Exception as e: self.log_message(f"Error terminating ffmpeg: {e}", "error")
        self.ffmpeg_process = None 
        
        if self.whisper_mp_process and self.whisper_mp_process.is_alive():
            self.log_message("Terminating active Whisper transcription process...", "debug")
            try: self.whisper_mp_process.terminate(); self.whisper_mp_process.join(timeout=1) 
            except Exception as e: self.log_message(f"Error terminating Whisper process: {e}", "error")
        self.whisper_mp_process = None
        
        self.stop_button.config(state=tk.DISABLED)

    def _process_videos_thread(self, video_files, output_folder_str, model_name, lang_whisper, device_choice):
        start_time_total = time.time()
        files_processed_count, files_succeeded_count, files_failed_count = 0, 0, 0
        
        actual_device_for_whisper, use_fp16_for_whisper = "cpu", False
        if device_choice == "cuda":
            if self.has_cuda:
                actual_device_for_whisper, use_fp16_for_whisper = "cuda", True
                self.log_message("Using GPU (CUDA) for Whisper process.", "info")
            else: self.log_message("CUDA selected, but not available. Whisper process will use CPU.", "warning")
        else:
            self.log_message("Using CPU for Whisper process.", "info")
        
        try:
            num_total_files = len(video_files)
            if num_total_files == 0: 
                self.log_message("No video files to process in thread.", "warning")
                return

            for i, video_path in enumerate(video_files):
                if not self.is_processing: 
                    self.log_message("Processing stopped by user before starting next file.", "warning")
                    break 
                
                files_processed_count += 1
                self.log_message(f"Processing file {files_processed_count}/{num_total_files}: {video_path.name}", "info")
                
                # Progress bar update logic removed
                # progress_percent = (float(files_processed_count) / num_total_files) * 100
                # self.log_message(f"DEBUG: Progress bar update: {files_processed_count}/{num_total_files} = {progress_percent:.2f}%", "debug")
                # self.root.after(0, lambda val=progress_percent: self.progress_var.set(val))


                temp_audio_path = None
                try:
                    if not self.is_processing: break 

                    self.log_message(f"Extracting audio from '{video_path.name}'...", "debug")
                    temp_audio_path = pathlib.Path(output_folder_str) / f"_temp_audio_{video_path.stem}.wav"
                    temp_audio_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    ffmpeg_command = [FFMPEG_EXE, "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", str(temp_audio_path)]
                    
                    if not self.is_processing: break 
                    self.ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
                    stdout, stderr = self.ffmpeg_process.communicate() 
                    ffmpeg_rc = self.ffmpeg_process.returncode
                    self.ffmpeg_process = None 

                    if not self.is_processing: break 
                    if ffmpeg_rc != 0:
                        self.log_message(f"ffmpeg error for {video_path.name}: {stderr.decode(errors='ignore')}", "error")
                        files_failed_count += 1; continue 
                    
                    self.log_message(f"Audio extracted to '{temp_audio_path.name}'.", "debug")
                    self.log_message(f"Starting Whisper transcription process for '{video_path.name}'...", "info")
                    
                    if not self.is_processing: break
                    
                    result_queue_mp = multiprocessing.Queue()
                    self.whisper_mp_process = multiprocessing.Process(
                        target=run_whisper_transcription_process,
                        args=(
                            str(temp_audio_path), output_folder_str, video_path.stem,
                            model_name, lang_whisper, actual_device_for_whisper, use_fp16_for_whisper,
                            result_queue_mp
                        )
                    )
                    self.whisper_mp_process.start()
                    
                    while self.whisper_mp_process.is_alive():
                        if not self.is_processing:
                            self.log_message(f"Stop requested. Terminating Whisper process for {video_path.name}...", "warning")
                            self.whisper_mp_process.terminate()
                            self.whisper_mp_process.join(timeout=2) 
                            break 
                        time.sleep(0.2) 
                    
                    if not self.is_processing: 
                        self.log_message(f"Whisper process for {video_path.name} handled after stop request.", "debug")
                        try: result_queue_mp.get_nowait() 
                        except queue.Empty: pass
                        break 

                    transcription_succeeded = False
                    try:
                        transcription_succeeded = result_queue_mp.get(timeout=5) 
                    except queue.Empty:
                        self.log_message(f"Whisper process for {video_path.name} did not return a result in time.", "warning")
                    
                    self.whisper_mp_process = None 

                    if transcription_succeeded:
                        self.log_message(f"SRT file for '{video_path.name}' generated successfully by child process.", "success")
                        files_succeeded_count +=1
                    else:
                        self.log_message(f"Whisper transcription failed or was interrupted for '{video_path.name}'.", "error")
                        files_failed_count += 1
                        
                except Exception as e_file:
                    if self.is_processing: 
                        self.log_message(f"Error processing {video_path.name}: {e_file}", "error")
                        files_failed_count += 1
                finally:
                    if self.ffmpeg_process: self.ffmpeg_process = None
                    if self.whisper_mp_process and not self.whisper_mp_process.is_alive(): self.whisper_mp_process = None

                    if temp_audio_path and temp_audio_path.exists():
                        try: os.remove(temp_audio_path)
                        except OSError as e_del: self.log_message(f"Could not delete temporary audio file '{temp_audio_path.name}': {e_del}", "warning")
                    
        except Exception as e_thread: 
            self.log_message(f"Major error in processing thread: {e_thread}", "error")
            import traceback; self.log_message(f"Traceback: {traceback.format_exc()}", "debug")
        finally:
            total_time_taken = time.time() - start_time_total
            self.log_message("--- Processing Finished ---", "info")
            self.log_message(f"Total files processed/attempted: {files_processed_count}", "info") 
            self.log_message(f"Successfully processed: {files_succeeded_count}", "success")
            non_success_count = files_processed_count - files_succeeded_count
            self.log_message(f"Failed or stopped early: {non_success_count}", "error" if non_success_count > 0 else "info")
            self.log_message(f"Total time taken: {total_time_taken:.2f} seconds.", "info")
            
            self.is_processing = False 
            self.root.after(0, lambda: self.set_ui_state(False)) 
            
            # Final progress bar update logic removed
            # if num_total_files > 0:
            #     final_progress_val = (float(files_processed_count) / num_total_files) * 100
            #     self.log_message(f"DEBUG: Final progress bar update: {files_processed_count}/{num_total_files} = {final_progress_val:.2f}%", "debug")
            #     self.root.after(0, lambda val=final_progress_val: self.progress_var.set(val))
            # elif num_total_files == 0 : 
            #      self.log_message(f"DEBUG: Final progress bar update: No files, setting to 0%", "debug")
            #      self.root.after(0, lambda: self.progress_var.set(0.0))
            
            num_input_files = len(video_files)
            summary_msg = ""
            summary_title = "Processing Complete"
            if num_input_files == 0: summary_msg = "No files were processed."
            else: summary_msg = f"Finished: {files_succeeded_count}/{num_input_files} files.\nTotal time: {total_time_taken:.2f}s."

            if non_success_count > 0 and num_input_files > 0: 
                summary_title = "Processing Issues"
                summary_msg += f"\nFailed or stopped early: {non_success_count}" 
                self.root.after(0, lambda sm=summary_msg, st=summary_title: messagebox.showwarning(st, sm))
            elif num_input_files > 0: self.root.after(0, lambda sm=summary_msg, st=summary_title: messagebox.showinfo(st, sm))
            elif num_input_files == 0: self.root.after(0, lambda sm=summary_msg, st=summary_title: messagebox.showinfo(st, sm))

def main():
    if sys.platform in ['win32', 'darwin']: 
        multiprocessing.set_start_method('spawn', force=True)
    
    if sys.platform == "win32":
        try: from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
        except Exception: pass
        
    root = tk.Tk()
    app = TranscriberApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
