from loguru import logger
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# Local Modules
import util

config = (util.get_config()).com_config

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=config.model_path),
    running_mode=VisionRunningMode.VIDEO)

def estimate_pose(ret, frame):
    """Returns 33 segment pose estimation from opencv """


    logger.info("Beginning Pose Estimating Job...")
    with PoseLandmarker.create_from_options(options) as landmarker:
        logger.info("Initialized Pose Estimator")

