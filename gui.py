# gui.py
import tkinter as tk
from tkinter import filedialog, ttk
import threading
import os
import sys
import subprocess
from pathlib import Path
import cv2
from PIL import Image, ImageTk

# Import the functions from our backend
from pipeline import get_initial_detections, run_segmentation_pipeline

class VideoPlayer:
    """A simple video player to display video frames in a Tkinter Label."""
    def __init__(self, label: ttk.Label, root: tk.Tk):
        self.label = label
        self.root = root
        self.cap = None
        self.paused = False
        self.video_path = None

    def load(self, path):
        if self.cap:
            self.cap.release()
        self.video_path = path
        self.cap = cv2.VideoCapture(self.video_path)
        self.paused = False
        self._update_frame()

    def _update_frame(self):
        if self.paused or not self.cap:
            return

        ret, frame = self.cap.read()
        if ret:
            # Resize frame to fit the label while maintaining aspect ratio
            frame_height, frame_width, _ = frame.shape
            label_width = self.label.winfo_width()
            label_height = self.label.winfo_height()
            
            # Avoid division by zero if window is not yet drawn
            if label_width < 2 or label_height < 2: 
                self.root.after(20, self._update_frame) # Try again shortly
                return

            scale = min(label_width / frame_width, label_height / frame_height)
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert frame for Tkinter
            cv2image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

            # Repeat after a delay matching the video's FPS
            delay = int(1000 / self.cap.get(cv2.CAP_PROP_FPS))
            self.root.after(delay, self._update_frame)
        else:
            # End of video, release the capture
            self.cap.release()
            self.cap = None

    def toggle_pause(self):
        self.paused = not self.paused
        if not self.paused:
            self._update_frame()

# gui.py

# ... (imports and VideoPlayer class are unchanged) ...

class SegmentationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Minimal Segmentation Tool")
        self.root.geometry("1200x800")

        # --- State Variables ---
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.player_name = tk.StringVar(value="Bam Adebayo")
        self.motion_class = tk.StringVar(value="freethrow")
        self.status = tk.StringVar(value="Status: Ready")
        
        self.first_frame = None
        self.initial_detections = None
        self.selected_player_idx = -1
        self.final_video_path = None

        # --- Layout ---
        top_frame = ttk.Frame(root, padding="10")
        top_frame.pack(fill=tk.X, side=tk.TOP)
        video_frame = ttk.Frame(root, padding="10")
        video_frame.pack(fill=tk.BOTH, expand=True)
        status_frame = ttk.Frame(root, padding="10")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.original_video_player = VideoPlayer(ttk.Label(video_frame, text="Original Video", relief="groove"), root)
        self.original_video_player.label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.segmented_video_player = VideoPlayer(ttk.Label(video_frame, text="Segmented Video", relief="groove"), root)
        self.segmented_video_player.label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # --- Controls in Top Frame ---
        ttk.Button(top_frame, text="Select Input Video", command=self.select_input).grid(row=0, column=0, padx=5, pady=2, sticky='w')
        ttk.Label(top_frame, textvariable=self.input_path, relief="sunken", width=50).grid(row=0, column=1, padx=5, pady=2, sticky='we')
        
        ttk.Button(top_frame, text="Select Output Folder", command=self.select_output).grid(row=1, column=0, padx=5, pady=2, sticky='w')
        ttk.Label(top_frame, textvariable=self.output_path, relief="sunken", width=50).grid(row=1, column=1, padx=5, pady=2, sticky='we')
        
        ttk.Label(top_frame, text="Player Name:").grid(row=0, column=2, padx=(10, 0), pady=2, sticky='w')
        ttk.Entry(top_frame, textvariable=self.player_name, width=20).grid(row=0, column=3, padx=5, pady=2, sticky='w')
        
        ttk.Label(top_frame, text="Motion Class:").grid(row=1, column=2, padx=(10, 0), pady=2, sticky='w')
        ttk.Entry(top_frame, textvariable=self.motion_class, width=20).grid(row=1, column=3, padx=5, pady=2, sticky='w')

        self.start_button = ttk.Button(top_frame, text="Start Segmentation", command=self.start_full_process)
        self.start_button.grid(row=0, column=4, rowspan=2, padx=20, ipady=10)
        
        top_frame.columnconfigure(1, weight=1)

        # --- Status Bar ---
        ttk.Label(status_frame, textvariable=self.status).pack(side=tk.LEFT)
        self.save_button = ttk.Button(status_frame, text="Open Output Folder", state=tk.DISABLED, command=self.open_output_folder)
        self.save_button.pack(side=tk.RIGHT)

    def update_status(self, message):
        self.root.after(0, lambda: self.status.set(message))

    def select_input(self):
        """NEW: This function now only selects the file path and starts the video player."""
        path = filedialog.askopenfilename(title="Select a video file", filetypes=[("MP4 files", "*.mp4")])
        if not path:
            return
            
        self.input_path.set(path)
        self.original_video_player.load(path)
        self.update_status("Status: Video loaded. Ready to start segmentation.")

    def select_output(self):
        path = filedialog.askdirectory(title="Select a folder for the output")
        if path:
            self.output_path.set(path)

    def start_full_process(self):
        """NEW: This is the main command for the 'Start' button. It runs the entire sequence."""
        if not self.input_path.get() or not self.output_path.get():
            self.update_status("Status: Please select input and output paths first.")
            return

        # --- STAGE 1: Initial Detection (Runs on main GUI thread) ---
        self.update_status("Status: Finding players in the first frame...")
        self.root.update() # Force UI update
        try:
            self.first_frame, self.initial_detections = get_initial_detections(self.input_path.get())
            if len(self.initial_detections) == 0:
                self.update_status("ERROR: No players detected. Please select a different video.")
                return
        except Exception as e:
            self.update_status(f"ERROR during detection: {e}")
            return
            
        # --- STAGE 2: Player Selection (Runs on main GUI thread) ---
        self.select_player_with_cv2()
        
        # --- STAGE 3: Start Background Processing (Only if a player was selected) ---
        if self.selected_player_idx != -1:
            self.start_background_thread()

    def select_player_with_cv2(self):
        """Uses OpenCV to pop up a window for player selection."""
        selection_result = []
        annotated_frame = self.first_frame.copy()
        window_name = "Click the Player to Track (then press any key)"
        
        for i, xyxy in enumerate(self.initial_detections.xyxy):
            cv2.rectangle(annotated_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(annotated_frame, str(i), (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for i, xyxy in enumerate(self.initial_detections.xyxy):
                    if xyxy[0] < x < xyxy[2] and xyxy[1] < y < xyxy[3]:
                        print(f"Player {i} selected!")
                        selection_result.append(i)
                        # Don't destroy window on click, let user see their choice
                        # Redraw with selection highlighted
                        temp_frame = self.first_frame.copy()
                        for j, box in enumerate(self.initial_detections.xyxy):
                            color = (0, 255, 0) if i == j else (0, 0, 255)
                            thickness = 4 if i == j else 2
                            cv2.rectangle(temp_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)
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
        
        thread = threading.Thread(target=self.run_segmentation_in_background)
        thread.daemon = True
        thread.start()

    def run_segmentation_in_background(self):
        self.final_video_path = run_segmentation_pipeline(
            source_video_path_str=self.input_path.get(),
            output_folder_str=self.output_path.get(),
            player_name=self.player_name.get(),
            motion_class=self.motion_class.get(),
            selected_player_idx=self.selected_player_idx,
            first_frame=self.first_frame,
            initial_detections=self.initial_detections,
            status_callback=self.update_status
        )
        self.root.after(0, self.on_processing_finished)

    def on_processing_finished(self):
        self.start_button.config(state=tk.NORMAL)
        if self.final_video_path:
            self.save_button.config(state=tk.NORMAL)
            self.segmented_video_player.load(self.final_video_path)
        else:
            self.update_status("Status: Processing failed. Check console for error details.")
            
    def open_output_folder(self):
        if self.output_path.get():
            folder = self.output_path.get()
            # Platform-independent way to open a file explorer
            if sys.platform == "win32": os.startfile(folder)
            elif sys.platform == "darwin": subprocess.run(["open", folder])
            else: subprocess.run(["xdg-open", folder])

# The main execution block is the same
if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("dotenv library not found, skipping .env file loading.")

    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()