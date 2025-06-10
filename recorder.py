import cv2
import os
import time
import datetime
from threading import Thread
from queue import Queue, Full, Empty

import config

class VideoRecorder:
    def __init__(self, output_folder, resolution=(640, 360), fps=30, segment_duration=3600, max_queue_size=30):  # Default max_queue_size to 30*fps for 1s buffer
        """
        Args:
            output_folder (str): Folder to save recordings
            resolution (tuple): Width and height for recording
            fps (int): Frames per second for recording
            segment_duration (int): Duration of each video file in seconds
            max_queue_size (int): Maximum size of the frame queue
        """
        self.output_folder = output_folder
        self.resolution = resolution
        self.fps = fps if fps > 0 else 10  # Ensure FPS is positive
        self.segment_duration = segment_duration

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Initialize frame queue
        self.frame_queue = Queue(maxsize=max_queue_size)

        # Control flags
        self.is_running = False
        # self.is_recording = False # This flag seems unused, is_running covers it

        # Current video writer
        self.video_writer = None
        self.current_filename = None
        self.segment_start_time = 0
        self.record_thread = None

    def add_frame(self, frame):
        """Add a frame to the recording queue"""
        if not self.is_running or frame is None:  # Added check for None frame
            return

        # Resize the frame to the specified resolution if needed
        # This should be done *before* adding to queue if frames can vary in size
        # Assuming frames passed are already correct or will be resized by the caller if necessary
        # However, the constructor implies the recorder handles resizing. Let's ensure it.
        if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
            try:
                frame_resized = cv2.resize(frame, self.resolution)
            except cv2.error as e:
                print(
                    f"Error resizing frame for recorder: {e}. Original shape: {frame.shape}, Target: {self.resolution}")
                return  # Skip problematic frame
        else:
            frame_resized = frame

        # If queue is full, remove oldest frame (non-blocking)
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                pass  # Should not happen if full() is true, but good for safety

        # Add frame to queue (non-blocking)
        try:
            self.frame_queue.put_nowait(frame_resized)
        except Full:
            # This might happen in a race condition if queue fills between full() check and put_nowait()
            # Or if max_queue_size is very small / processing is slow
            # print("Recorder queue full, frame dropped.") # Can be noisy
            pass

    def start(self):
        # Start recording
        if self.is_running:
            print("Recorder is already running")
            return

        self.is_running = True

        # Start the recording thread
        self.record_thread = Thread(target=self._record_frames, daemon=True)
        # self.record_thread.daemon = True # Already set daemon=True
        self.record_thread.start()

        print(f"Recording started. FPS: {self.fps}, Resolution: {self.resolution}")
        print(f"Videos will be saved to {os.path.abspath(self.output_folder)}")

    def stop(self):
        # Stop recording
        if not self.is_running:
            # print("Recorder is not running.") # Optional: uncomment if useful
            return

        print("Stopping recorder...")
        self.is_running = False

        # Wait for thread to finish
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=5.0)  # Increased timeout for potentially slow disk I/O
            if self.record_thread.is_alive():
                print("Recorder thread did not finish in time.")

        # Release the video writer if it exists
        if self.video_writer is not None:
            try:
                self.video_writer.release()
                print(f"Saved final recording segment to {self.current_filename}")
            except Exception as e:
                print(f"Error releasing video writer: {e}")
            self.video_writer = None

        # Clear the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        print("Recorder queue cleared.")
        print("Recording stopped.")

    def _record_frames(self):
        while self.is_running or not self.frame_queue.empty():  # Process remaining frames after stopping
            try:
                frame = self.frame_queue.get(timeout=0.1)  # Wait briefly for a frame
            except Empty:
                if not self.is_running:  # If stopped and queue is empty, exit
                    break
                continue  # If running, continue waiting

            if frame is None or frame.size == 0:
                # print("Skipping empty frame in recorder thread...") # Can be noisy
                continue

            # Check if we need to create a new video file
            current_time = time.time()

            if self.video_writer is None or \
                    (self.segment_duration > 0 and (current_time - self.segment_start_time) >= self.segment_duration):
                # Close previous writer if it exists
                if self.video_writer is not None:
                    self.video_writer.release()
                    print(f"Saved recording segment to {self.current_filename}")

                # Create a new video file
                timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                self.current_filename = os.path.join(self.output_folder, f"{timestamp}.mp4")

                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
                # Ensure frame dimensions match writer dimensions
                # Frame should already be resized by add_frame or its caller
                height, width = frame.shape[0], frame.shape[1]
                if width != self.resolution[0] or height != self.resolution[1]:
                    print(
                        f"Warning: Frame dimensions ({width}x{height}) mismatch VideoWriter resolution ({self.resolution[0]}x{self.resolution[1]}). This may cause issues.")
                    # Attempt to resize again, though ideally this shouldn't happen
                    try:
                        frame = cv2.resize(frame, self.resolution)
                        height, width = frame.shape[0], frame.shape[1]
                    except cv2.error as e:
                        print(f"Error resizing frame in _record_frames: {e}. Skipping frame.")
                        continue

                self.video_writer = cv2.VideoWriter(
                    self.current_filename, fourcc, self.fps, (width, height)
                )
                if not self.video_writer.isOpened():
                    print(
                        f"Error: Could not open video writer for {self.current_filename}. Recording for this segment will fail.")
                    self.video_writer = None  # Prevent further write attempts if not opened
                    # Potentially retry or handle error more gracefully
                    continue

                self.segment_start_time = current_time
                print(f"Started new recording segment: {self.current_filename}")

            # Write the frame to the video file, if writer is valid
            if self.video_writer and self.video_writer.isOpened():
                try:
                    self.video_writer.write(frame)
                except Exception as e:
                    print(f"Error writing frame to video: {e}")
                    # Consider how to handle this, e.g., stop writer, try new segment

        # Final release if loop exited while writer was active (e.g. is_running became false)
        if self.video_writer is not None and self.video_writer.isOpened():
            self.video_writer.release()
            print(f"Saved final recording segment (on exit) to {self.current_filename}")
            self.video_writer = None


if __name__ == "__main__":
    # Ensure config is accessible for the test
    # This block might need adjustment if config.py relies on ROOT_DIR being set relative to its own location
    # For simplicity, we assume config variables are directly usable.
    # For a real test, you might mock config or ensure paths are absolute.

    # Example: Create a dummy config for testing if needed
    class DummyConfig:
        RAW_RECORDING_DIR = "test_recordings/raw"
        HEADLESS = True  # Don't show cv2 window for this test
        CAMERA_URL = 0  # Use webcam for testing, or a test video file
        # CAMERA_URL = "your_test_video.mp4"
        FPS = 20
        SAVE_PERIOD = 10  # Short segment for testing

        def GET_RESOLUTION(self, res_str, aspect_ratio):
            if res_str == '720p': return (1280, 720)
            return (640, 360)  # Default for test


    # Use actual config if available and suitable for test, else DummyConfig
    try:
        from config import RAW_RECORDING_DIR, HEADLESS, CAMERA_URL, FPS, SAVE_PERIOD, GET_RESOLUTION

        print("Using actual config for recorder test.")
    except ImportError:
        print("Actual config not found or incomplete, using DummyConfig for recorder test.")
        config = DummyConfig()
        RAW_RECORDING_DIR = config.RAW_RECORDING_DIR
        HEADLESS = config.HEADLESS
        CAMERA_URL = config.CAMERA_URL
        FPS = config.FPS
        SAVE_PERIOD = config.SAVE_PERIOD
        GET_RESOLUTION = config.GET_RESOLUTION


    def connect_camera(rtsp_url, max_retries=5, retry_delay=5):
        for attempt in range(max_retries):
            print(f"Connecting to camera ({rtsp_url}), attempt {attempt + 1}/{max_retries}...")
            cap = cv2.VideoCapture(rtsp_url)  # Removed cv2.CAP_FFMPEG for broader compatibility in test
            if cap.isOpened():
                print("Camera connection successful")
                ret, test_frame = cap.read()
                if ret:
                    return cap
                else:
                    cap.release()
                    print(f"Could not read frame. Retrying in {retry_delay} seconds...")
            else:
                print(f"Could not open camera. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        print("Failed to connect to camera after maximum retries")
        return None


    recorder_resolution = GET_RESOLUTION(None, '720p', 16 / 9)  # Test with 720p
    recorder = VideoRecorder(output_folder=RAW_RECORDING_DIR,
                             resolution=recorder_resolution,
                             fps=FPS,
                             segment_duration=SAVE_PERIOD  # Short segments for test
                             )

    cap = connect_camera(CAMERA_URL)  # Use connect_camera for robustness
    if cap is None:
        print("Error capturing video for test. Exiting.")
        exit(-1)

    recorder.start()
    print("Press Esc to exit test.")

    start_time = time.time()
    frames_processed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame from camera test.")
                # Attempt reconnect or use last frame logic if desired for a more robust test
                break  # Simple break for test

            # No need to resize here if recorder.add_frame handles it, but for consistency:
            # frame_display = cv2.resize(frame, recorder_resolution) # Match recorder input if displaying

            recorder.add_frame(frame.copy())  # Pass a copy
            frames_processed += 1

            if not HEADLESS:
                display_frame = cv2.resize(frame, (640, 360))  # Smaller display
                cv2.putText(display_frame, f"FPS: {FPS}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Recorder Test", display_frame)

            if cv2.waitKey(1) & 0xFF == 0x1B:  # Esc
                break

            # Run for a limited time for automated tests, e.g., 2 * SAVE_PERIOD
            if time.time() - start_time > 2.5 * SAVE_PERIOD:
                print(f"Test duration ({2.5 * SAVE_PERIOD}s) reached.")
                break

            # Simulate camera FPS
            time.sleep(1.0 / FPS if FPS > 0 else 0.1)

    except KeyboardInterrupt:
        print("Test interrupted by user.")
    finally:
        print(f"Frames processed during test: {frames_processed}")
        recorder.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print(f'Recording test session ended at {datetime.datetime.now()}')