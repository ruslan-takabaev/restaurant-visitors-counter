import os

ROOT_DIR = r"/home/mitc/count_people/"  # Path to the project root directory

HEADLESS = False  # "True" to run without display window; "False" to show display window;
RECORDING = False  # "True" to record video; "False" to run without recording.

"""WEBSOCKET"""
WEBSOCKET_HOST = '0.0.0.0'  # Listen on all interfaces
WEBSOCKET_PORT = 8765  # Port for WebSocket connections
WEBSOCKET_STREAM_QUALITY = 60  # JPEG quality for WebSocket stream (0-100, higher is better quality, larger size)

"""INTERNAL MODULES"""
# Path to directory for saving raw recordings
RAW_RECORDING_DIR = os.path.join(ROOT_DIR, "rec/raw/")
# Path to directory for saving annotated recordings
ANNOTATED_RECORDING_DIR = os.path.join(ROOT_DIR, "rec/annotated/")
# Path to directory for saving faces
FACE_DIR = os.path.join(ROOT_DIR, "faces/")
# Path to object detection model
MODEL_PATH = os.path.join(ROOT_DIR, "models/yolov8s_custom_v1.1.pt")
# Path to database file
DB_PATH = os.path.join(ROOT_DIR, "DB/crowd_records.db")

"""VIDEO PARAMETERS"""
ASPECT_RATIO = 16 / 9  # Default video aspect ratio (16:9)
FPS = 25  # Default video FPS (25)
DEFAULT_RESOLUTION = '1080p'  # Default video resolution (1920x1080)

"""CAMERA RTSP ADDRESS"""
CAMERA_URL = "rtsp://ipcam.uzcloud.uz:8080/rtsp/18269399/ad4ac096b8eeaca291aa"

"""==============================================================================================================="""
"""===================DO NOT CHANGE ANYTHING BELOW THIS LINE UNLESS YOU KNOW WHAT YOU ARE DOING==================="""
"""==============================================================================================================="""


# Get proper recording resolution according to aspect ratio (default 16:9)
def GET_RESOLUTION(resolution=DEFAULT_RESOLUTION, aspect_ratio=ASPECT_RATIO):
    height = round(int(resolution.split('p')[0].split('P')[0]), 0)
    width = round((height * aspect_ratio), 0)
    return int(width), int(height)


# Get a point for a counting line (A or B)
def GET_POINT(resolution=DEFAULT_RESOLUTION, aspect_ratio=ASPECT_RATIO, point='a'):
    a, b = GET_RESOLUTION(resolution, aspect_ratio)

    if point == 'a':
        return int(a * 0.80), int(b * 0.35)
    elif point == 'b':
        return int(a * 0.80), int(b * 0.85)

    return 0


"""TRACKING QUALITY CONSTANTS"""
MIN_TRACK_HISTORY = 8  # Minimum number of points needed for a reliable direction
MIN_DISTANCE = 20  # Minimum distance required between the first and current point
TRACK_TIMEOUT = 2.0  # Time in seconds after which a track is removed
FACE_SAVING_PROB = 0.15  # Probability of saving a picture of a new detected face

"""TIME CONSTANTS"""
TIME_FORMAT = "%d-%m-%Y_%H:%M:%S"  # Time format for datetime.datetime.now().strftime()
SQL_TIME_FORMAT = "%Y%m%d_%H%M%S"
REPORT_TIME = "03:00:00"  # Timestamp to reset counters and submit daily reports to db
SAVE_PERIOD = 3600  # Time period between recording (saves in seconds)