# Local Modules
import util

from loguru import logger
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

config = (util.get_config()).com_config

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=config.model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1)

def estimate_pose(video_path: str):
    """Returns 33 segment pose estimation from video path"""
    cap = cv2.VideoCapture(video_path)
    results = []
    try:
        logger.info("Beginning Pose Estimating Job...") 

        fps = cap.get(cv2.CAP_PROP_FPS)
        index = 0

        with PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                index += 1
                timestamp = int((index / fps) * 1000)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                results.append(landmarker.detect_for_video(mp_image, timestamp))
    
    except Exception as e:
        logger.error(f"Error in processing pose: {repr(e)}")
    
    finally:
        cap.release() 







