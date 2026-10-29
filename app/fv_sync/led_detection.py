"""
LED Detection

Finds the LED in a video and turns its blinking into a per-frame on/off signal.
The view configs (crop regions, plate swap) live here too.
"""

__author__ = 'Aiden Lee'

from typing import Optional, Tuple
import cv2
import numpy as np
import pandas as pd


class LEDConfig:
    # Name: Long, Top, Side1, Side2
    view_name: str = "place holder"

    # Frame dimensions
    frame_width: int = 1920
    frame_height: int = 1080

    # Rectangular crop: (x1, y1) top-left corner, (x2, y2) bottom-right corner
    led_crop_x1: int = 0
    led_crop_y1: int = 0
    led_crop_x2: int = 0
    led_crop_y2: int = 0

    # Offset from the LED template's top-left corner to get to the LED center, remeasure if new LED
    template_center_offset_x: int = 45
    template_center_offset_y: int = 47

    # Number of pixels from center in rectangle, so 5 = 11x11 box
    signal_box: int = 5

    # How many frames to sample when locating the LED
    detection_frames: int = 11

    # True when this view sees the plates mirrored relative to the force file, so
    # fp1 and fp2 are swapped and we treat the force file as the absolute
    swap_plate: bool = False

    # Placeholder for LED template
    led_template: Optional[np.ndarray] = None

    def __init__(self):
        #Create the LED template if one wasn't made
        if self.led_template is None:
            self.led_template = self.create_led_template()

    def create_led_template(self) -> np.ndarray:
        #Blank canvas the LED shape is drawn onto, 71 rows by 91 columns
        template = np.zeros((71, 91), dtype=np.uint8)
        cv2.rectangle(template, (20, 27), (71, 68), 200, -1)  # outer led rectangle
        cv2.rectangle(template, (42, 30), (49, 45), 10, -1)   # dark LED core that was better
        #Blur so template matching tolerates a few pixels of misalignment
        return cv2.blur(template, (5, 5))

    def get_crop_region(self, frame: np.ndarray) -> np.ndarray:
        #Return the frame cropped to the detection region
        return frame[self.led_crop_y1:self.led_crop_y2,
                     self.led_crop_x1:self.led_crop_x2]

    @staticmethod
    def process_crop_for_matching(crop: np.ndarray) -> np.ndarray:
        #Blue minus green: the LED housing pops, everything else flattens.
        return cv2.blur(cv2.subtract(crop[:, :, 0], crop[:, :, 1]), (10, 10))


class LEDDetectionLong(LEDConfig):
    view_name = "Long View"
    led_crop_x1, led_crop_y1 = 850, 950
    led_crop_x2, led_crop_y2 = 1050, 1080
    swap_plate = False


class LEDDetectionTop(LEDConfig):
    view_name = "Top View"
    led_crop_x1, led_crop_y1 = 850, 950
    led_crop_x2, led_crop_y2 = 1050, 1080
    swap_plate = False


class LEDDetectionSide1(LEDConfig):
    view_name = "Side1 View"           # LED on the left
    led_crop_x1, led_crop_y1 = 0, 360
    led_crop_x2, led_crop_y2 = 640, 720
    swap_plate = False


class LEDDetectionSide2(LEDConfig):
    view_name = "Side2 View"           # LED on the right
    led_crop_x1, led_crop_y1 = 1280, 360
    led_crop_x2, led_crop_y2 = 1920, 720
    swap_plate = True


view_map = {
    "Long View":  LEDDetectionLong,
    "Top View":   LEDDetectionTop,
    "Side1 View": LEDDetectionSide1,
    "Side2 View": LEDDetectionSide2,
}


def get_config(view_type: str) -> LEDConfig:
    #Error if the view isn't one of the known views
    if view_type not in view_map:
        raise ValueError(f"Unknown view type {view_type!r}, "
                         f"expected one of {list(view_map)}")
    #Create the config for the specific view
    return view_map[view_type]()


def open_video(video_path: str) -> cv2.VideoCapture:
    #Read the video
    cap = cv2.VideoCapture(video_path)
    #Error if the video can't be read
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    return cap


def clamp_center(center: Tuple[int, int], width: int, height: int,
                 d: int) -> Tuple[int, int]:
    #Keep the center d pixels away from every edge so the signal box stays on the frame
    cx = int(np.clip(center[0], d, width - d - 1))
    cy = int(np.clip(center[1], d, height - d - 1))
    return cx, cy


def match_led(frame: np.ndarray, config: LEDConfig) -> Tuple[int, int]:
    #Crop the frame to the set region
    processed = config.process_crop_for_matching(config.get_crop_region(frame))

    #Match the LED template from the cropped frame
    result = cv2.matchTemplate(processed, config.led_template, cv2.TM_SQDIFF)
    #Finds the minimum and maximum pixels of the LED template
    _, _, min_loc, _ = cv2.minMaxLoc(result)

    #Get the center from the min pixel corners of the led template
    return (min_loc[0] + config.template_center_offset_x + config.led_crop_x1,
            min_loc[1] + config.template_center_offset_y + config.led_crop_y1)


def find_led(video_path: str, config: LEDConfig) -> Tuple[int, int]:
    #Read the video
    cap = open_video(video_path)

    #Number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #Center array across samples
    centers = []

    #Linearly spacing the number of detection frames from all frames in the video
    for idx in np.linspace(0, max(total_frames - 1, 0), config.detection_frames, dtype=int):
        #Jumps directly to the frame sampling for LED
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        #Reads the frame
        ok, frame = cap.read()
        if not ok:
            continue
        #Where the template matched on this frame
        centers.append(match_led(frame, config))
    #Release the cap
    cap.release()

    #If the centers is empty then we can't detect the LED problem with CROP
    if not centers:
        raise ValueError(f"LED never detected in {video_path} check cropping")

    #Turn the list of centers into a numpy array
    points = np.asarray(centers)
    #Determine far the LED center moved between the sampled frames (x and y)
    spread = points.max(axis=0) - points.min(axis=0)
    #Warn if the LED moved more than 40 pixels, detection might be off
    if spread.max() > 40:
        print(f"[WARN] {config.view_name}: LED moved {spread} px across sampled frames")

    #Return the median center as for average a bad frame can throw it off
    cx, cy = np.median(points, axis=0).astype(int)
    return int(cx), int(cy)


def video_led_signal(video_path: str, center: Tuple[int, int],
                     config: LEDConfig) -> pd.DataFrame:
    #Read the video
    cap = open_video(video_path)

    #Half size of the box around the LED center
    d = config.signal_box
    #Frame size, to keep the box from running off the edge
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #Pull the center in from the edge, a negative slice would wrap around and give NaN
    cx, cy = clamp_center(center, width, height, d)
    if (cx, cy) != (int(center[0]), int(center[1])):
        print(f"[WARN] {config.view_name}: LED center {tuple(center)} too close to "
              f"the edge, moved to {(cx, cy)}")
    #Red score for every frame
    scores = []

    #Go through every frame in the video
    while True:
        #Reads the next frame
        ok, frame = cap.read()
        #Stop when there are no more frames
        if not ok:
            break
        #Mean red in the box around the LED, ROI only so we don't copy whole frames
        scores.append(frame[cy - d:cy + d + 1, cx - d:cx + d + 1, 2].mean())

    #Release the cap
    cap.release()

    #Turn the scores into a numpy array
    scores = np.asarray(scores)
    #Threshold midway between the off state and the on state
    threshold = (np.percentile(scores, 25) + np.percentile(scores, 75)) / 2

    #Return frame number, red score, and LED signal (+1 on, -1 off) for every frame
    return pd.DataFrame({
        'FrameNumber': np.arange(len(scores)),
        'RedScore': scores,
        'Video_LED_Signal': np.sign(scores - threshold),
    })


def process_view(view_type: str, video_path: str,
                 led_center: Optional[Tuple[int, int]] = None,
                 show_mask: bool = False) -> pd.DataFrame:
    #Create the config for the specific view, errors if the view is unknown
    config = get_config(view_type)
    #Use the hand picked center if one was given, otherwise auto detect it
    if led_center is not None:
        center = (int(led_center[0]), int(led_center[1]))
        print(f"[{config.view_name}] LED set manually at {center}")
    else:
        #Find where the LED is in the video
        center = find_led(video_path, config)
        print(f"[{config.view_name}] LED found at {center}")
    #Pop up the black and white mask to see what the auto detection is matching on
    if show_mask:
        #Imported here because manual_led imports this module
        from .manual_led import show_led_mask
        show_led_mask(video_path, view_type, center)
    #Get the LED on/off signal for every frame
    return video_led_signal(video_path, center, config)
