"""
Force Video Sync
"""

__author__ = 'Aiden Lee'

import os
from typing import Optional, Tuple
import cv2
import numpy as np
import pandas as pd
from scipy import signal as sps

# Force plate runs at 1200 Hz, video at 120 fps -> 10 force rows per video frame.
force_per_fps = 10

force_columns = {
    'abs time (s)': 'Time(s)',
    'Fx':     'FP1_Fx',  'Fy':     'FP1_Fy',  'Fz':     'FP1_Fz',
    '|Ft|':   'FP1_|F|', 'Ax':     'FP1_Ax',  'Ay':     'FP1_Ay',
    'Fx.1':   'FP2_Fx',  'Fy.1':   'FP2_Fy',  'Fz.1':   'FP2_Fz',
    '|Ft|.1': 'FP2_|F|', 'Ax.1':   'FP2_Ax',  'Ay.1':   'FP2_Ay',
    'Fx.2':   'FP3_Fx',  'Fy.2':   'FP3_Fy',  'Fz.2':   'FP3_Fz',
    '|Ft|.2': 'FP3_|F|', 'Ax.2':   'FP3_Ax',  'Ay.2':   'FP3_Ay',
}

force_plate_pairs = [('FP1_Fx', 'FP2_Fx'), ('FP1_Fy', 'FP2_Fy'), ('FP1_Fz', 'FP2_Fz'),
                 ('FP1_|F|', 'FP2_|F|'), ('FP1_Ax', 'FP2_Ax'), ('FP1_Ay', 'FP2_Ay')]

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

    """
    True when this view sees the plates mirrored relative to the force file, 
    so fp1 and fp2 are swapped relative to force file and we treat the force file as the absolute
    """
    swap_plate: bool = False

    # Placeholder for LED template
    led_template: Optional[np.ndarray] = None

    def __init__(self):
        #Creating and LED template if not made
        if self.led_template is None:
            self.led_template = self.create_led_template()

    def create_led_template(self) -> np.ndarray:
        template = np.zeros((71, 91), dtype=np.uint8)
        cv2.rectangle(template, (20, 27), (71, 68), 200, -1)  # outer led rectangle
        cv2.rectangle(template, (42, 30), (49, 45), 10, -1)   # dark LED core that was better
        return cv2.blur(template, (5, 5))

    def get_crop_region(self, frame: np.ndarray) -> np.ndarray:
        #returning the frame cropped
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
    swap_plate = True   # TODO: confirm against a real top-view recording


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

def find_led(video_path: str, config: LEDConfig) -> Tuple[int, int]:
    #Read the video
    cap = cv2.VideoCapture(video_path)
    #Error if the video can't be read
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

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

        #Crop the frame to the set region
        processed = config.process_crop_for_matching(config.get_crop_region(frame))

        #Match the LED template from the cropped frame
        result = cv2.matchTemplate(processed, config.led_template, cv2.TM_SQDIFF)
        #Finds the minimum and maximum pixels of the LED template
        _, _, min_loc, _ = cv2.minMaxLoc(result)

        #Get the center from the min pixel corners of the led template
        centers.append((min_loc[0] + config.template_center_offset_x + config.led_crop_x1,
                        min_loc[1] + config.template_center_offset_y + config.led_crop_y1))
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
    return tuple(np.median(points, axis=0).astype(int))


def video_led_signal(video_path: str, center: Tuple[int, int],
                     config: LEDConfig) -> pd.DataFrame:
    #Read the video
    cap = cv2.VideoCapture(video_path)
    #Error if the video can't be read
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    #Unpack the LED center
    cx, cy = center
    #Half size of the box around the LED center
    d = config.signal_box
    #Red score for every frame
    scores = []

    #Go through every frame in the video
    while True:
        #Reads the next frame
        ok, frame = cap.read()
        #Stop when there are no more frames
        if not ok:
            break
        # Red channel, ROI only no full-frame copies.
        scores.append(frame[cy - d:cy + d + 1, cx - d:cx + d + 1, 2].mean())

    #Release the cap
    cap.release()

    #Turn the scores into a numpy array
    scores = np.asarray(scores)
    # Threshold midway between the off-state and on-state.
    threshold = (np.percentile(scores, 25) + np.percentile(scores, 75)) / 2

    #Return frame number, red score, and LED signal (+1 on, -1 off) for every frame
    return pd.DataFrame({
        'FrameNumber': np.arange(len(scores)),
        'RedScore': scores,
        'Video_LED_Signal': np.sign(scores - threshold),
    })


def process_view(view_type: str, video_path: str) -> pd.DataFrame:
    #Error if the view isn't one of the known views
    if view_type not in view_map:
        raise ValueError(f"Unknown view type {view_type!r}, "
                         f"expected one of {list(view_map)}")

    #Create the config for the specific view
    config = view_map[view_type]()
    #Find where the LED is in the video
    center = find_led(video_path, config)
    print(f"[{config.view_name}] LED found at {center}")
    #Get the LED on/off signal for every frame
    return video_led_signal(video_path, center, config)

def load_force(force_path: str) -> pd.DataFrame:
    #Read the tab separated force file, where headers start on row 17
    df = (pd.read_csv(force_path, header=17, delimiter='\t', encoding='latin1')
            #Drop the units row under the header
            .drop(0)
            #Convert everything to numbers, other values become NaN
            .apply(pd.to_numeric, errors='coerce')
            #Rename columns to FP1/FP2/FP3 names
            .rename(columns=force_columns)
            #Reset the index back to start at 0
            .reset_index(drop=True))
    # Channel 3 is wired to the LED.
    df['FP_LED_Signal'] = np.sign(df['FP3_Fz'])
    return df


def swap_force_plates(df: pd.DataFrame) -> pd.DataFrame:
    #Copy so the original dataframe isn't changed
    df = df.copy()
    #Go through each FP1/FP2 column pair
    for a, b in force_plate_pairs:
        #Only swap if both columns exist
        if a in df.columns and b in df.columns:
            #Swap FP1 and FP2 values
            df[a], df[b] = df[b].copy(), df[a].copy()
    return df

def find_lag(video_sig: np.ndarray, force_sig: np.ndarray) -> Tuple[int, float]:
    #Normalize a signal to mean 0 and std 1 so both signals are on the same scale
    def z(a):
        a = np.asarray(a, dtype=float)
        #Skip dividing if the signal is flat (std 0)
        return (a - a.mean()) / a.std() if a.std() > 0 else a

    #Slide the force signal along the video signal and score how well they match
    corr = sps.correlate(z(video_sig), z(force_sig), mode="valid")
    #The frame offset for each correlation score
    lags = sps.correlation_lags(video_sig.size, force_sig.size, mode="valid")
    #Index of the best match
    best = int(np.argmax(corr))
    #Return the best lag and its correlation score
    return int(lags[best]), float(corr[best])


def sync(view: str, parent_path: str, video_file: str, force_file: str):
    #Create the config for this view, None if the view is unknown
    config = view_map[view]() if view in view_map else None
    #Error if the view isn't one of the known views
    if config is None:
        raise ValueError(f"Unknown view type {view!r}, expected one of {list(view_map)}")

    #Get the LED signal from the video
    df_video = process_view(view, os.path.join(parent_path, video_file))
    #Load the force plate data
    df_force = load_force(os.path.join(parent_path, force_file))

    #Keep every 10th force row so force matches the video frame rate
    df_dec = df_force.iloc[::force_per_fps].reset_index(drop=True)
    #Find how many frames the force data is shifted from the video
    lag, score = find_lag(df_video['Video_LED_Signal'].to_numpy(),
                          df_dec['FP_LED_Signal'].to_numpy())
    print(f"[{view}] lag = {lag} frames, correlation = {score:.1f}")

    #Give each downsampled force row the video frame it lines up with
    df_dec['FrameNumber'] = np.arange(lag, lag + len(df_dec))
    #Join the force and video data on frame number
    df_frames = df_dec.merge(df_video, on='FrameNumber', how='left')

    #Copy the 1200 Hz force data
    df_full = df_force.copy()
    #Fractional number for every force row (10 rows per frame)
    df_full['FrameNumber'] = lag + df_full.index / force_per_fps

    #Swap FP1 and FP2 if this view sees the plates mirrored
    if config.swap_plate:
        print(f"[{view}] swapping FP1 <-> FP2")
        df_frames = swap_force_plates(df_frames)
        df_full = swap_force_plates(df_full)

    return lag, df_frames, df_full


def new_led(self, view, parent_path, video_file, force_file):
    #Run the sync for this view
    lag, df_frames, df_full = sync(view, parent_path, video_file, force_file)
    #Save the full force data on the app state if it has one
    if hasattr(self, 'state'):
        self.state.df_aligned_full = df_full
    return lag, df_frames