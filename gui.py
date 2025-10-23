import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import os
import sys
import gc
import numpy as np
import subprocess
from pathlib import Path
import cv2
from PIL import Image, ImageTk

# Import the new functions from our backend
from pipeline import PLAYER_CLASS_IDS, get_initial_detections, process_video_and_get_masks, finalize_and_save

class VideoPlayer:
    def __init__(self, label: ttk.Label, root: tk.Tk):
        self.label = label
        self.root = root
        self.cap = None
        self.paused = False
        self.video_path = None

    def load(self, path):
        if self.cap: self.cap.release()
        self.video_path = path
        self.cap = cv2.VideoCapture(self.video_path)
        self.paused = False
        self._update_frame()

    def replay(self):
        if self.video_path: self.load(self.video_path)

    def _update_frame(self):
        if self.paused or not self.cap or not self.cap.isOpened(): return
        ret, frame = self.cap.read()
        if ret:
            frame_height, frame_width, _ = frame.shape
            label_width, label_height = self.label.winfo_width(), self.label.winfo_height()
            if label_width < 2 or label_height < 2:
                self.root.after(20, self._update_frame)
                return
            scale = min(label_width / frame_width, label_height / frame_height)
            resized_frame = cv2.resize(frame, (int(frame_width * scale), int(frame_height * scale)))
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)))
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
            delay = int(1000 / self.cap.get(cv2.CAP_PROP_FPS))
            self.root.after(delay, self._update_frame)
        else:
            self.cap.release()
            self.cap = None

class SegmentationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Minimal Segmentation Tool")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- State Variables ---
        self.default_input_dir = "/fs/scratch/PAS3184"
        self.default_output_dir = "/fs/scratch/PAS3184"
        self.input_path, self.output_path = tk.StringVar(value=self.default_input_dir), tk.StringVar(value=self.default_output_dir)
        self.player_name, self.motion_class = tk.StringVar(), tk.StringVar()
        self.status = tk.StringVar(value="Status: Ready")
        self.first_frame, self.initial_detections = None, None
        self.temp_masks_json_path, self.temp_video_path = None, None
        self.selected_player_idx = -1

        # --- Layout ---
        top_frame = ttk.Frame(root, padding="10")
        top_frame.pack(fill=tk.X)
        video_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        video_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        original_frame = ttk.Frame(video_paned_window, relief="groove")
        segmented_frame = ttk.Frame(video_paned_window, relief="groove")
        video_paned_window.add(original_frame, weight=1)
        video_paned_window.add(segmented_frame, weight=1)
        self.original_video_player = VideoPlayer(ttk.Label(original_frame), root)
        self.original_video_player.label.pack(fill=tk.BOTH, expand=True)
        self.segmented_video_player = VideoPlayer(ttk.Label(segmented_frame), root)
        self.segmented_video_player.label.pack(fill=tk.BOTH, expand=True)
        status_frame = ttk.Frame(root, padding="10")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # --- Controls ---
        ttk.Button(top_frame, text="Browse Input...", command=self.select_input).grid(row=0, column=0, sticky='w')
        self.input_entry = ttk.Entry(top_frame, textvariable=self.input_path)
        self.input_entry.grid(row=0, column=1, sticky='we')
        self.input_entry.bind("<Return>", lambda e: self.on_input_entry_enter())
        ttk.Button(top_frame, text="Browse Output...", command=self.select_output).grid(row=1, column=0, sticky='w')
        self.output_entry = ttk.Entry(top_frame, textvariable=self.output_path)
        self.output_entry.grid(row=1, column=1, sticky='we')
        self.output_entry.bind("<Return>", lambda e: self.on_output_entry_enter())
        ttk.Label(top_frame, text="Player Name:").grid(row=0, column=2, sticky='w', padx=(10,0))
        ttk.Entry(top_frame, textvariable=self.player_name).grid(row=0, column=3, sticky='w')
        ttk.Label(top_frame, text="Motion Class:").grid(row=1, column=2, sticky='w', padx=(10,0))
        ttk.Entry(top_frame, textvariable=self.motion_class).grid(row=1, column=3, sticky='w')
        self.start_button = ttk.Button(top_frame, text="Start Segmentation", command=self.start_full_process)
        self.start_button.grid(row=0, column=4, rowspan=2, padx=20, ipady=10)
        self.reset_button = ttk.Button(top_frame, text="Reset For New Video", command=self.reset_state)
        self.reset_button.grid(row=0, column=5, rowspan=2, padx=5)
        top_frame.columnconfigure(1, weight=1)

        # Make entries stretch across available space
        top_frame.columnconfigure(3, weight=0)

        # --- Status Bar ---
        ttk.Label(status_frame, textvariable=self.status).pack(side=tk.LEFT)
        self.save_button = ttk.Button(status_frame, text="Confirm & Save", state=tk.DISABLED, command=self.confirm_and_save)
        self.save_button.pack(side=tk.RIGHT, padx=5)
        self.replay_seg_button = ttk.Button(status_frame, text="Replay Segmented", state=tk.DISABLED, command=self.segmented_video_player.replay)
        self.replay_seg_button.pack(side=tk.RIGHT, padx=5)
        self.replay_og_button = ttk.Button(status_frame, text="Replay Original", state=tk.DISABLED, command=self.original_video_player.replay)
        self.replay_og_button.pack(side=tk.RIGHT, padx=5)

    # REPLACE the entire 'reset_models' method with this:
    def reset_state(self):
        """Clears the application state to prepare for a new video."""
        self.update_status("Status: Resetting. Ready for a new video.")

        # Clean up any leftover temp video file
        if self.temp_video_path and os.path.exists(self.temp_video_path):
            try:
                os.remove(self.temp_video_path)
                print(f"Cleaned up temporary file: {self.temp_video_path}")
            except OSError as e:
                print(f"Error removing temp file during reset: {e}")

        # Clear all state-holding variables
        self.input_path.set("")
        self.first_frame = None
        self.initial_detections = None
        self.temp_masks_json_path = None
        self.temp_video_path = None
        self.selected_player_idx = -1

        # Stop video players and release resources
        if self.original_video_player.cap:
            self.original_video_player.cap.release()
            self.original_video_player.cap = None
        if self.segmented_video_player.cap:
            self.segmented_video_player.cap.release()
            self.segmented_video_player.cap = None

        # Create a blank image to clear the video display labels
        blank_img = ImageTk.PhotoImage(Image.new('RGB', (1, 1)))
        self.original_video_player.label.configure(image=blank_img)
        self.original_video_player.label.imgtk = blank_img
        self.segmented_video_player.label.configure(image=blank_img)
        self.segmented_video_player.label.imgtk = blank_img

        # Reset button states
        self.save_button.config(state=tk.DISABLED)
        self.replay_seg_button.config(state=tk.DISABLED)
        self.replay_og_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)

        # Force Python's garbage collector to run
        gc.collect()

    def update_status(self, message): self.root.after(0, lambda: self.status.set(message))

    def select_input(self):
        initial_dir = self.input_path.get().strip()
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")], initialdir=initial_dir)
        if not path: return
        self.input_path.set(path)
        self.replay_og_button.config(state=tk.NORMAL)
        self.original_video_player.load(path)
        self.update_status("Status: Video loaded. Ready to start segmentation.")

    def select_output(self):
        initial_dir = self.output_path.get().strip()
        path = filedialog.askdirectory(initialdir=initial_dir)
        if path: self.output_path.set(path)

    def on_input_entry_enter(self):
        """Called when user types a path into the input entry and presses Enter."""
        path = self.input_path.get().strip()
        if not path:
            messagebox.showwarning("Input Path", "Please enter a path to an input video.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Input Path", f"The path does not exist: {path}")
            return
        # Try to load the video into the player
        try:
            self.replay_og_button.config(state=tk.NORMAL)
            self.original_video_player.load(path)
            self.update_status("Status: Video loaded from typed path. Ready to start segmentation.")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load video: {e}")

    def on_output_entry_enter(self):
        """Called when user types a path into the output entry and presses Enter."""
        path = self.output_path.get().strip()
        if not path:
            messagebox.showwarning("Output Path", "Please enter a path to an output folder.")
            return
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
                self.update_status(f"Status: Created output folder: {path}")
            except Exception as e:
                messagebox.showerror("Output Path", f"Unable to create folder: {e}")
                return
        else:
            self.update_status(f"Status: Using output folder: {path}")

    def start_full_process(self):
        if not self.input_path.get() or not self.output_path.get():
            self.update_status("Status: Please select input and output paths first.")
            return
        self.update_status("Status: Finding players in the first frame...")
        self.root.update()
        try:
            self.first_frame, self.initial_detections = get_initial_detections(self.input_path.get())
            if len(self.initial_detections) == 0:
                self.update_status("ERROR: No players detected. Please select a different video.")
                return
        except Exception as e:
            self.update_status(f"ERROR during detection: {e}")
            return
        self.select_player_with_cv2()
        if self.selected_player_idx != -1:
            self.start_background_thread()


    def select_player_with_cv2(self):
        selection_result = []
        player_detections = self.initial_detections[np.isin(self.initial_detections.class_id, PLAYER_CLASS_IDS)]
        annotated_frame = self.first_frame.copy()
        window_name = "Click Player to Track (then press any key)"
        for i, xyxy in enumerate(player_detections.xyxy):
            cv2.rectangle(annotated_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(annotated_frame, str(i), (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check clicks against the players-only list
                for i, xyxy in enumerate(player_detections.xyxy):
                    if xyxy[0] < x < xyxy[2] and xyxy[1] < y < xyxy[3]:
                        selection_result.append(i)
                        temp_frame = self.first_frame.copy()
                        # Highlight the selected player from the players-only list
                        for j, box in enumerate(player_detections.xyxy):
                            color = (0, 255, 0) if i == j else (0, 0, 255)
                            cv2.rectangle(temp_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 4 if i == j else 2)
                        cv2.imshow(window_name, temp_frame)
                        break
        cv2.imshow(window_name, annotated_frame)
        cv2.setMouseCallback(window_name, mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if selection_result:
            self.selected_player_idx = selection_result[-1]
            self.update_status(f"Status: Player {self.selected_player_idx} selected. Starting processing...")
        else:
            self.selected_player_idx = -1
            self.update_status("Status: No player selected. Aborting.")

    def start_background_thread(self):
            """Starts the main segmentation process in a separate thread to avoid freezing the GUI."""
            self.start_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
            self.replay_seg_button.config(state=tk.DISABLED)
            
            # The target is our existing method that already has all the info it needs
            thread = threading.Thread(target=self.run_segmentation_in_background, daemon=True)
            thread.start()
    
    def run_segmentation_in_background(self):
        """This is the function that the background thread actually runs."""
        # This calls the main processing function from pipeline.py
        self.temp_masks_json_path, self.temp_video_path = process_video_and_get_masks(  
            source_video_path_str=self.input_path.get(),
            output_folder_str=self.output_path.get(),
            selected_player_idx=self.selected_player_idx,
            first_frame=self.first_frame,
            initial_detections=self.initial_detections,
            status_callback=self.update_status
        )
        
        # When the processing is done, schedule the on_processing_finished method to run on the main GUI thread
        self.root.after(0, self.on_processing_finished)

    def check_thread(self, thread, on_finish_func):
        if thread.is_alive():
            self.root.after(100, self.check_thread, thread, on_finish_func)
        else:
            on_finish_func()

    def on_processing_finished(self):
        self.start_button.config(state=tk.NORMAL)
        if self.temp_video_path:
            self.save_button.config(state=tk.NORMAL)
            self.segmented_video_player.load(self.temp_video_path)
            self.replay_seg_button.config(state=tk.NORMAL)
        else:
            self.update_status("Status: Processing failed. Check console for details.")

    def confirm_and_save(self):
        self.update_status("Status: Saving...")
        self.save_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        # We can run save in a thread too, though it's much faster
        thread = threading.Thread(target=self.save_in_background, daemon=True)
        thread.start()

    def save_in_background(self):
        final_path = finalize_and_save(
            temp_video_path_str=self.temp_video_path, 
            temp_masks_json_str=self.temp_masks_json_path,  # 传递JSON文件路径
            output_folder_str=self.output_path.get(), 
            source_video_path_str=self.input_path.get(),
            player_name=self.player_name.get(), 
            motion_class=self.motion_class.get(),
            status_callback=self.update_status
        )
        def on_save_finish():
            self.start_button.config(state=tk.NORMAL)
        self.root.after(0, on_save_finish)

    def on_closing(self):
        if self.temp_video_path and os.path.exists(self.temp_video_path):
            try:
                os.remove(self.temp_video_path)
                print(f"Cleaned up temporary file: {self.temp_video_path}")
            except OSError as e: print(f"Error removing temporary file: {e}")
        self.root.destroy()
            
    def open_output_folder(self):
        if self.output_path.get():
            folder = self.output_path.get()
            if sys.platform == "win32": os.startfile(folder)
            elif sys.platform == "darwin": subprocess.run(["open", folder])
            else: subprocess.run(["xdg-open", folder])

if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
        os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
        os.environ["ROBOFLOW_API_KEY"] = os.getenv("ROBOFLOW_API_KEY")
        os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"
        os.environ["MODEL_CACHE_DIR"] = "./cache"
    except ImportError:
        print("dotenv library not found, skipping .env file loading.")
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()