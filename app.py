import cv2
import numpy as np
from collections import defaultdict
import time
import os
import sys
import config
from ultralytics import YOLO
import sqlite3
import datetime
import threading
import uuid
import json
import base64
import asyncio
import websockets
from pathlib import Path
from queue import Queue
import signal

from recorder import VideoRecorder

"""WebSocket server to stream data to UI"""
class WebSocketServer:
    def __init__(self, host=config.WEBSOCKET_HOST, port=config.WEBSOCKET_PORT, face_images_ref=None):
        self.host = host
        self.port = port
        self.connected_clients = set()
        self.frame_queue = Queue(maxsize=5)
        self.stats_queue = Queue(maxsize=10)
        self.event_queue = Queue(maxsize=10)
        self.server = None
        self.is_running = False
        self.face_images_ref = face_images_ref

    async def __call__(self, websocket, path=None):
        # Register client
        self.connected_clients.add(websocket)
        try:
            # Handle client messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    # Handle requests like face image retrieval
                    if data.get('type') == 'get_face':
                        face_id = data.get('face_id')
                        if face_id is not None:
                            await self.send_face_image(websocket, int(face_id))
                    elif data.get('type') == 'get_all_faces':
                        limit = data.get('limit', 50)
                        await self.send_all_faces(websocket, int(limit))
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"error": "Invalid JSON message"}))
                except ValueError:
                    await websocket.send(json.dumps({"error": "Invalid face_id or limit format"}))
                except Exception as e:
                    print(f"Error processing client message: {e}")
                    await websocket.send(json.dumps({"error": "Internal server error processing request"}))

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            # Unregister client
            self.connected_clients.remove(websocket)

    async def send_face_image(self, websocket, face_id):
        if self.face_images_ref is None:
            await websocket.send(
                json.dumps({"type": "face_image", "face_id": face_id, "error": "Face data not available on server"}))
            return

        face_filename = self.face_images_ref.get(face_id)  # face_id is int
        if face_filename:
            filepath = os.path.join(config.FACE_DIR, face_filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, "rb") as img_file:
                        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                    await websocket.send(json.dumps(
                        {"type": "face_image", "face_id": face_id, "image_data": encoded_string,
                         "filename": face_filename}))
                except Exception as e:
                    print(f"Error reading face image {filepath}: {e}")
                    await websocket.send(json.dumps(
                        {"type": "face_image", "face_id": face_id, "error": f"Error reading face image: {e}"}))
            else:
                await websocket.send(
                    json.dumps({"type": "face_image", "face_id": face_id, "error": "Face image file not found"}))
        else:
            await websocket.send(json.dumps({"type": "face_image", "face_id": face_id, "error": "Face ID not found"}))

    async def send_all_faces(self, websocket, limit):
        if self.face_images_ref is None:
            await websocket.send(json.dumps({"type": "all_faces", "error": "Face data not available on server"}))
            return

        response_faces = []
        try:
            # Create a copy of items to iterate over for thread safety
            current_face_items = list(self.face_images_ref.items())
        except RuntimeError:
            await websocket.send(
                json.dumps({"type": "all_faces", "error": "Face data temporarily unavailable, try again."}))
            return

        sorted_face_items = sorted(current_face_items, key=lambda item: item[1], reverse=True)

        count = 0
        for face_id, face_filename in sorted_face_items:  # face_id is int
            if count >= limit:
                break

            filepath = os.path.join(config.FACE_DIR, face_filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, "rb") as img_file:
                        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                    response_faces.append({"face_id": face_id, "filename": face_filename, "image_data": encoded_string})
                    count += 1
                except Exception as e:
                    print(f"Error reading face image {face_filename} for all_faces: {e}")
            else:
                print(f"Face image file not found: {face_filename} for all_faces (may have been deleted)")

        await websocket.send(json.dumps({"type": "all_faces", "faces": response_faces}))

    async def broadcast_frame(self):
        """Broadcast the latest frame to all connected clients"""
        while self.is_running:
            if not self.frame_queue.empty():
                encoded_frame = self.frame_queue.get()
                
                # Create a copy of the set to prevent RuntimeError during iteration
                clients_to_broadcast = self.connected_clients.copy()
                if clients_to_broadcast:
                    message = json.dumps({
                        "type": "frame",
                        "data": encoded_frame,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    })
    
                    # Iterate over the safe copy
                    for websocket in clients_to_broadcast:
                        try:
                            await websocket.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            # If a connection is closed, remove it from the original set
                            self.connected_clients.discard(websocket)
    
            await asyncio.sleep(0.03)
    
    
    async def broadcast_stats(self):
        """Broadcast the latest statistics to all connected clients"""
        while self.is_running:
            if not self.stats_queue.empty():
                stats = self.stats_queue.get()
                
                # Create a copy of the set to prevent RuntimeError during iteration
                clients_to_broadcast = self.connected_clients.copy()
                if clients_to_broadcast:
                    message = json.dumps({
                        "type": "stats",
                        "data": stats,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    })
    
                    # Iterate over the safe copy
                    for websocket in clients_to_broadcast:
                        try:
                            await websocket.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            # If a connection is closed, remove it from the original set
                            self.connected_clients.discard(websocket)
    
            await asyncio.sleep(0.5)
    
    
    async def broadcast_event(self):
        """Broadcast event information to all connected clients"""
        while self.is_running:
            if not self.event_queue.empty():
                event_data = self.event_queue.get()
                
                # Create a copy of the set to prevent RuntimeError during iteration
                clients_to_broadcast = self.connected_clients.copy()
                if clients_to_broadcast:
                    message = json.dumps({
                        "type": "event",
                        "data": event_data,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    })
    
                    # Iterate over the safe copy
                    for websocket in clients_to_broadcast:
                        try:
                            await websocket.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            # If a connection is closed, remove it from the original set
                            self.connected_clients.discard(websocket)
    
            await asyncio.sleep(0.1)

    
    def add_frame(self, frame):
        """Add a frame to the queue for broadcasting"""
        try:
            # Convert frame to JPEG to reduce size, using quality from config
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.WEBSOCKET_STREAM_QUALITY])
            # Convert to base64 for easy transport over WebSocket
            encoded_frame = base64.b64encode(buffer).decode('utf-8')

            # If queue is full, remove oldest frame
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass

            self.frame_queue.put(encoded_frame)
        except Exception as e:
            print(f"Error adding frame to queue: {e}")

    def update_stats(self, count_in, count_out, face_count, daily_event_count, event_active, current_event_count_in):
        """Update statistics for broadcasting, including event info"""
        stats = {
            "count_in": count_in,
            "count_out": count_out,
            "face_count": face_count,
            "daily_event_count": daily_event_count,  # Total events today
            "event_active": event_active,  # Is an event currently active?
            "current_event_count_in": current_event_count_in  # Count for the current active event
        }

        # If queue is full, remove oldest stats
        if self.stats_queue.full():
            try:
                self.stats_queue.get_nowait()
            except:
                pass

        self.stats_queue.put(stats)

    def send_event_update(self, event_data):
        """Add event data to the queue for broadcasting"""
        try:
            # If queue is full, remove oldest event data
            if self.event_queue.full():
                try:
                    self.event_queue.get_nowait()
                except:
                    pass
            self.event_queue.put(event_data)
        except Exception as e:
            print(f"Error adding event data to queue: {e}")

    async def start_server(self):
        """Start the WebSocket server"""
        self.is_running = True
        self.server = await websockets.serve(
            self, self.host, self.port
        )
        print(f"WebSocket server started at ws://{self.host}:{self.port}")

        # Start broadcasting tasks
        asyncio.create_task(self.broadcast_frame())
        asyncio.create_task(self.broadcast_stats())
        asyncio.create_task(self.broadcast_event())

        # Keep server running
        await self.server.wait_closed()

    def start(self):
        """Start the WebSocket server in a separate thread"""
        def run_server():
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start_server())

        threading.Thread(target=run_server, daemon=True).start()

    def stop(self):
        """Stop the WebSocket server"""
        self.is_running = False
        if self.server:
            self.server.close()


if __name__ == '__main__':
    # Suppress console output from YOLO
    os.environ["ULTRALYTICS_QUIET"] = "1"  # Set environment variable to quiet mode

    # Setup database
    crowd_records = sqlite3.connect(config.DB_PATH)
    cursor = crowd_records.cursor()

    # Ensure tables exist
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS person
                   (
                       track_id  TEXT PRIMARY KEY,
                       timestamp TEXT,
                       direction TEXT
                   )
                   ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS daily_report
                   (
                       date      TEXT PRIMARY KEY,
                       count_in  INTEGER,
                       count_out INTEGER,
                       event_count INTEGER
                   )
                   ''')

    # New table for events
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS event
                   (
                       date DATE,
                       start_time TIME,
                       end_time TIME,
                       count_in INTEGER
                   )
                   ''')

    crowd_records.commit()

    # Create directory for faces if it doesn't exist
    os.makedirs(config.FACE_DIR, exist_ok=True)

    # Dictionary to store face image filenames (track_id: filename)
    # This will be shared with the WebSocketServer
    face_images = {}
    
    # Initialize WebSocket server
    ws_server = WebSocketServer(
        host=getattr(config, 'WEBSOCKET_HOST', config.WEBSOCKET_HOST),
        port=getattr(config, 'WEBSOCKET_PORT', config.WEBSOCKET_PORT),
        face_images_ref=face_images  # Pass the reference to the dictionary
    )
    ws_server.start()


    # Check if a point is close enough to a line segment to be considered crossing it.
    def point_in_line(point, a, b, tolerance=10):
        p_x, p_y = point
        a_x, a_y = a
        b_x, b_y = b

        # Calculate the distance from point to line
        distance_numerator = np.abs((b_y - a_y) * p_x - (b_x - a_x) * p_y + b_x * a_y - b_y * a_x)
        distance_denominator = np.sqrt((b_y - a_y) ** 2 + (b_x - a_x) ** 2)
        if distance_denominator == 0:  # Points a and b are the same
            distance = np.sqrt((p_x - a_x) ** 2 + (p_y - a_y) ** 2)
        else:
            distance = distance_numerator / distance_denominator

        # Check if point is within the line segment (with tolerance)
        if distance <= tolerance:
            # Check if point is within the bounding box of the line
            if (min(a_x, b_x) - tolerance <= p_x <= max(a_x, b_x) + tolerance and
                    min(a_y, b_y) - tolerance <= p_y <= max(a_y, b_y) + tolerance):
                return True
        return False


    # Determine if a point is to the left of a line in vector terms.
    def is_left_to_the_line(point, a, b):
        p_x, p_y = point
        a_x, a_y = a
        b_x, b_y = b

        # Determinant (D)
        d = ((b_x - a_x) * (p_y - a_y) - (b_y - a_y) * (p_x - a_x))

        # D > 0 -> left side; D < 0 -> right side; D = 0 -> on the line
        return d > 0


    def calculate_distance(point1, point2):
        # Euclidean distance between two points
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


    def save_face(frame, box):
        """Save a detected face to the face directory"""
        try:
            x1, y1, x2, y2 = box
            face_img = frame[int(y1):int(y2), int(x1):int(x2)]

            # Skip if the detected face is too small
            if face_img.shape[0] < 30 or face_img.shape[1] < 30:
                return None

            # Generate a unique filename
            timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S-%f")[
                        :-3]  # Added milliseconds for uniqueness
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp}_{unique_id}.jpg"
            filepath = os.path.join(config.FACE_DIR, filename)

            # Save the face image
            cv2.imwrite(filepath, face_img)
            return filename
        except Exception as e:
            print(f"Error saving face: {e}")
            return None


    # Import yolo model (Using the new model that can detect both people and faces)
    model = YOLO(config.MODEL_PATH)
    model.verbose = False

    # Define a function to establish and reconnect camera
    def connect_camera(rtsp_url, max_retries=5, retry_delay=5):
        for attempt in range(max_retries):
            print(f"Connecting to camera, attempt {attempt + 1}/{max_retries}...")
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
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

    # Connect to camera
    cap = connect_camera(config.CAMERA_URL)
    if cap is None:
        print("Could not establish camera connection. Exiting.")
        if ws_server: ws_server.stop()
        sys.exit(1)

    # Get frame dimensions
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or config.FPS

    # Counting line
    line_start = config.GET_POINT(resolution='960p', point='a')
    line_end = config.GET_POINT(resolution='960p', point='b')

    line_start_cropped = config.GET_POINT(resolution='720p', point='a')  # For 720p recording frame
    line_end_cropped = config.GET_POINT(resolution='720p', point='b')  # For 720p recording frame

    # Tracking parameters
    track_history = defaultdict(lambda: [])  # Track history for each object ID
    track_time = {}  # Last time an object was seen
    counted_in = set()  # IDs of objects counted entering
    counted_out = set()  # IDs of objects counted exiting
    crossing_status = {}  # Track crossing status to prevent multiple counts
    count_in = 0  # Count of people entering
    count_out = 0  # Count of people exiting

    # Event counting variables
    daily_event_count = 0  # Total number of events today
    people_in_last_hour = 0  # Number of people entered in the last hour
    last_hour_start_time = time.time()  # Start time of the last hour
    event_active = False  # Flag to indicate if an event is currently active
    consecutive_low_hours = 0  # keep track of how many hours with low entry rate
    event_start_time = None  # Timestamp when the current event started
    current_event_count_in = 0  # Total count_in during the current active event

    # Set for tracking detected face IDs to avoid saving duplicates
    detected_faces = set()
    # face_images dictionary is initialized above and passed to ws_server

    timer_start = datetime.datetime.now().strftime(config.TIME_FORMAT)

    # For handling frame errors
    consecutive_errors = 0
    max_consecutive_errors = 1080000  # Up to 12 hours of reconnecting attempts (at 1 frame per 0.04s)
    # More reasonably, if trying to read once per second, this is 300 hours.
    # Let's assume it means attempts, not fixed time.
    last_successful_frame = None  # Keep the last good frame to use when encountering errors

    # Setup recorder
    recorder = VideoRecorder(
        output_folder=config.ANNOTATED_RECORDING_DIR,
        resolution=config.GET_RESOLUTION(resolution='360p', aspect_ratio=config.ASPECT_RATIO),
        # Use 360p for recordings
        fps=config.FPS,  # Use config.FPS for recorder, not hardcoded 10
        segment_duration=config.SAVE_PERIOD
    )

    # Handle graceful shutdown
    def signal_handler(sig, frame_signal):  # Renamed 'frame' to 'frame_signal' to avoid conflict
        print("Shutting down gracefully...")
        if cap is not None:
            cap.release()
        if config.RECORDING:
            recorder.stop()
        if ws_server is not None:
            ws_server.stop()
        cv2.destroyAllWindows()

        timer_end = datetime.datetime.now().strftime(config.TIME_FORMAT)
        # Print final counts
        print(f"Final count - IN: {count_in}, OUT: {count_out}")
        print(
            f"Total faces detected and saved (this session): {len(face_images)}")  # face_images better reflects saved items
        print(f"Daily event count: {daily_event_count}")  # Print daily event count
        print(f"Time range: {timer_start} - {timer_end}")

        try:
            if crowd_records:
                crowd_records.commit()
                crowd_records.close()
        except Exception as e:
            print(f"Error closing database: {e}", file=sys.stderr)  # Use stderr for errors

        sys.exit(0)


    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start the recorder if enabled
        if config.RECORDING:
            recorder.start()

        # Update stats periodically
        last_stats_update = time.time()
        stats_update_interval = 0.5  # Update stats every 0.5 seconds

        while True:
            # --- ROBUST FRAME READING & RECONNECTION LOGIC ---
            if not (cap and cap.isOpened()):
                # This handles the case where the cap object itself is invalid or has been released
                ret = False
            else:
                # Try to read a frame from the existing camera object
                ret, frame = cap.read()

            # If reading fails for any reason, enter the reconnection sub-loop
            if not ret:
                print("Stream lost or frame could not be read. Attempting to reconnect...")
                
                # Cleanly release the old camera object if it exists
                if cap:
                    cap.release()

                # Enter a dedicated reconnection loop that will block until successful
                reconnect_delay = 5  # Seconds between reconnection attempts
                while True:
                    print(f"Attempting to connect to camera: {config.CAMERA_URL}")
                    # Use your existing connect_camera function which has its own internal retries
                    cap = connect_camera(config.CAMERA_URL)

                    if cap is not None and cap.isOpened():
                        print("Reconnection successful. Resuming stream processing.")
                        # Invalidate the last good frame to ensure we don't process a stale image
                        last_successful_frame = None
                        # Break the inner reconnection loop to resume the main processing loop
                        break
                    else:
                        print(f"Reconnect attempt failed. Retrying in {reconnect_delay} seconds...")
                        time.sleep(reconnect_delay)
                # After successful reconnection, use 'continue' to skip the rest of this
                # loop iteration and start fresh with the new camera connection.
                continue
            # If we reach here, 'ret' was True, so we have a valid frame
            # Store a copy of the good frame to use as a fallback if needed (though the new logic minimizes this)
            last_successful_frame = frame.copy()

            # Resize frame for processing
            processed_frame = cv2.resize(frame, config.GET_RESOLUTION('960p', config.ASPECT_RATIO))

            # Run YOLO prediction with tracking - detect both people and faces
            try:
                # Class 0 = face, Class 1 = person
                results = model.track(processed_frame, persist=True, verbose=False,
                                      classes=[0, 1])  # Explicitly track person and face
            except Exception as e:
                print(f"Error in YOLO tracking: {e}")
                continue

            if results is None or len(results) == 0 or results[0].boxes is None:
                # If no detections, still send frame and update stats
                annotated_frame = processed_frame.copy()  # Use processed_frame if no detections
            else:
                # Create a copy of the frame for drawing
                annotated_frame = processed_frame.copy()
                result = results[0]

                if (hasattr(result.boxes, 'cls') and result.boxes.cls is not None):
                    boxes_xywh = result.boxes.xywh.cpu()
                    boxes_xyxy = result.boxes.xyxy.cpu()
                    classes = result.boxes.cls.cpu().tolist()
                    track_ids_list = result.boxes.id.cpu().tolist() if hasattr(result.boxes,
                                                                               'id') and result.boxes.id is not None else None

                    for i, (box_xywh, box_xyxy_coords, class_id) in enumerate(zip(boxes_xywh, boxes_xyxy, classes)):
                        current_track_id = int(track_ids_list[i]) if track_ids_list is not None else None

                        if int(class_id) == 1:  # class "People"
                            x_center, y_center, w, h = box_xywh
                            x1, y1, x2, y2 = box_xyxy_coords
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                            if current_track_id is not None:
                                track_point = (int(x_center), int(y_center + h / 2))
                                track_history[current_track_id].append(track_point)
                                track_history[current_track_id] = track_history[current_track_id][-20:]
                                track_time[current_track_id] = time.time()

                                if (len(track_history[current_track_id]) >= config.MIN_TRACK_HISTORY and
                                        point_in_line(track_point, line_start, line_end)):
                                    first_point = track_history[current_track_id][0]
                                    if calculate_distance(first_point, track_point) >= config.MIN_DISTANCE:
                                        first_left = is_left_to_the_line(first_point, line_start, line_end)
                                        current_left = is_left_to_the_line(track_point, line_start, line_end)
                                        crossing_key = f"{current_track_id}_{first_left}_{current_left}"

                                        if first_left and not current_left:  # Entering
                                            if (current_track_id not in counted_in and
                                                    current_track_id not in counted_out and
                                                    crossing_key not in crossing_status):
                                                count_in += 1
                                                counted_in.add(current_track_id)
                                                crossing_status[crossing_key] = True
                                                try:
                                                    cursor.execute(
                                                        "INSERT INTO person VALUES(?, ?, ?)",
                                                        (f"{datetime.datetime.now().strftime(config.SQL_TIME_FORMAT)}-{current_track_id}",
                                                         datetime.datetime.now().strftime(config.TIME_FORMAT), 'IN')
                                                    )
                                                    crowd_records.commit()
                                                except sqlite3.IntegrityError:
                                                    pass  # Ignore if already exists
                                                people_in_last_hour += 1
                                                if event_active: current_event_count_in += 1

                                        elif not first_left and current_left:  # Exiting
                                            if (current_track_id not in counted_out and
                                                    current_track_id not in counted_in and
                                                    crossing_key not in crossing_status):
                                                count_out += 1
                                                counted_out.add(current_track_id)
                                                crossing_status[crossing_key] = True
                                                try:
                                                    cursor.execute(
                                                        "INSERT INTO person VALUES(?, ?, ?)",
                                                        (f"{datetime.datetime.now().strftime(config.SQL_TIME_FORMAT)}-{current_track_id}",
                                                         datetime.datetime.now().strftime(config.TIME_FORMAT), 'OUT')
                                                    )
                                                    crowd_records.commit()
                                                except sqlite3.IntegrityError:
                                                    pass

                        elif int(class_id) == 0:  # class "Face"
                            x1, y1, x2, y2 = box_xyxy_coords
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)

                            if current_track_id is not None:
                                face_id_int = int(current_track_id)  # Ensure int for dict keys and set
                                if face_id_int not in detected_faces and np.random.random() < config.FACE_SAVING_PROB:
                                    face_filename = save_face(processed_frame,
                                                              box_xyxy_coords)  # Save from original processed_frame
                                    if face_filename:
                                        detected_faces.add(face_id_int)
                                        face_images[face_id_int] = face_filename  # Store with int key

                    current_time_tracks = time.time()
                    for track_id_key in list(track_time.keys()):
                        if current_time_tracks - track_time[track_id_key] > config.TRACK_TIMEOUT:
                            if track_id_key in track_history: del track_history[track_id_key]
                            if track_id_key in track_time: del track_time[track_id_key]

            # --- Event logic ---
            current_time_event_check = time.time()
            if current_time_event_check - last_hour_start_time >= 3600:
                print(f"Hourly check: {people_in_last_hour} people entered in the last hour.")
                if people_in_last_hour > 50:  # Event condition
                    if not event_active:
                        event_active = True
                        event_start_time = datetime.datetime.now()
                        consecutive_low_hours = 0
                        print(f"Event started at {event_start_time.strftime('%H:%M:%S')}!")
                        ws_server.send_event_update({
                            "status": "started",
                            "start_time": event_start_time.strftime('%H:%M:%S')
                        })
                    else:  # Event continues
                        consecutive_low_hours = 0
                        print("Event continues.")
                else:  # Low hourly count
                    if event_active:
                        consecutive_low_hours += 1
                        print(f"Low count for past hour. Consecutive low hours: {consecutive_low_hours}")
                        if consecutive_low_hours >= 2:
                            event_active = False
                            event_end_time = datetime.datetime.now()
                            daily_event_count += 1
                            print(
                                f"Event ended at {event_end_time.strftime('%H:%M:%S')}! Total IN: {current_event_count_in}. Daily Events: {daily_event_count}")
                            try:
                                cursor.execute(
                                    "INSERT INTO event VALUES (?, ?, ?, ?)",
                                    (event_start_time.strftime('%Y-%m-%d'),
                                     event_start_time.strftime('%H:%M:%S'),
                                     event_end_time.strftime('%H:%M:%S'),
                                     current_event_count_in)
                                )
                                crowd_records.commit()
                                print("Event recorded in database.")
                            except Exception as e:
                                print(f"Error recording event: {e}")
                            ws_server.send_event_update({
                                "status": "ended",
                                "start_time": event_start_time.strftime('%H:%M:%S'),
                                "end_time": event_end_time.strftime('%H:%M:%S'),
                                "count_in": current_event_count_in
                            })
                            current_event_count_in = 0
                            event_start_time = None
                            consecutive_low_hours = 0
                people_in_last_hour = 0
                last_hour_start_time = current_time_event_check

            # --- Annotation and display ---
            h_ann, w_ann = annotated_frame.shape[:2]
            cv2.rectangle(annotated_frame, (0, 0), (int(w_ann * 0.27), int(h_ann * 0.22)), (155, 155, 155), cv2.FILLED)

            for track_id_hist, points in track_history.items():
                color = (0, 255, 0) if track_id_hist in counted_in else (0, 0,
                                                                         255) if track_id_hist in counted_out else (200, 200, 200)
                for i_pt in range(1, len(points)):
                    if points[i_pt - 1] is None or points[i_pt] is None: continue
                    cv2.line(annotated_frame, points[i_pt - 1], points[i_pt], color, 1)

            cv2.line(annotated_frame, line_start, line_end, (255, 255, 255), 2)
            text_y_offset = 40
            cv2.putText(annotated_frame, f"IN: {count_in}", (20, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"OUT: {count_out}", (20, text_y_offset * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"FACES: {len(face_images)}", (20, text_y_offset * 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Events Today: {daily_event_count}", (20, text_y_offset * 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            if event_active:
                cv2.putText(annotated_frame, f"Event IN: {current_event_count_in}", (20, text_y_offset * 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if config.RECORDING:
                recording_frame = cv2.resize(annotated_frame, config.GET_RESOLUTION('360p', config.ASPECT_RATIO))
                rec_line_start = config.GET_POINT(resolution='360p', point='a')
                rec_line_end = config.GET_POINT(resolution='360p', point='b')
                cv2.line(recording_frame, rec_line_start, rec_line_end, (255, 255, 255), 1)
                recorder.add_frame(recording_frame)
        
            ws_server.add_frame(annotated_frame)

            if not config.HEADLESS:
                display_frame = cv2.resize(annotated_frame, (1280, 720))
                cv2.imshow('Live stream at [LOCATION_NAME]', display_frame)

            current_time_stats = time.time()
            if current_time_stats - last_stats_update >= stats_update_interval:
                ws_server.update_stats(count_in, count_out, len(face_images), daily_event_count, event_active,
                                       current_event_count_in)
                last_stats_update = current_time_stats

            # --- Daily report logic ---
            now = datetime.datetime.now()
            report_time_obj = datetime.datetime.strptime(config.REPORT_TIME, "%H:%M:%S").time()
            if now.time() >= report_time_obj and (
                    now - datetime.timedelta(seconds=max(1, stats_update_interval * 2))).time() < report_time_obj:
                today_date_str = now.strftime('%Y-%m-%d')
                try:
                    cursor.execute("SELECT 1 FROM daily_report WHERE date = ?", (today_date_str,))
                    if cursor.fetchone() is None:
                        cursor.execute(
                            "INSERT INTO daily_report VALUES (?, ?, ?, ?)",
                            (today_date_str, count_in, count_out, daily_event_count)
                        )
                        crowd_records.commit()
                        print(f"Daily report generated for {today_date_str}")

                        count_in = 0
                        count_out = 0
                        daily_event_count = 0
                        detected_faces.clear()
                        face_images.clear()
                        if event_active:
                            print(f"Note: An event was active during daily report generation. Resetting event state.")
                        event_active = False
                        current_event_count_in = 0
                        consecutive_low_hours = 0
                        event_start_time = None
                        people_in_last_hour = 0
                        last_hour_start_time = time.time()
                    else:
                        print(f"Daily report for {today_date_str} already exists. Skipping generation.")
                except sqlite3.Error as e:
                    print(f"SQLite error generating daily report: {e}")
                    error_log_path = os.path.join(config.ROOT_DIR, 'daily_report_error.log')
                    with open(error_log_path, mode='a', encoding='utf-8') as error_log:
                        error_log.write(
                            f"{datetime.datetime.now()}: Error for {today_date_str} | IN: {count_in} | OUT: {count_out} | EVENTS: {daily_event_count} | DB_Error: {e}\n")
                except Exception as e:
                    print(f"General error generating daily report: {e}")

            if cv2.waitKey(1) & 0xFF == 0x1B:  # Escape key
                break

    except Exception as e:
        print(f"Unexpected error in main loop: {e}")
        
        import traceback
        traceback.print_exc()
    finally:
        signal_handler(None, None)
