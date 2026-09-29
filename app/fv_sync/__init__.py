"""
Force Video Sync

Aligns a force plate file to a video by cross-correlating the two LED signals,
then hands back the force data reindexed onto video frame numbers.
"""

__author__ = 'Aiden Lee'

import os
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy import signal as sps

from .led_detection import (LEDConfig, LEDDetectionLong, LEDDetectionTop,
                            LEDDetectionSide1, LEDDetectionSide2, view_map,
                            find_led, video_led_signal, process_view)
from .manual_led import pick_led_center, manual_led

# Re-exported so callers can reach the whole pipeline from app.fv_sync
__all__ = ['LEDConfig', 'LEDDetectionLong', 'LEDDetectionTop', 'LEDDetectionSide1',
           'LEDDetectionSide2', 'view_map', 'find_led', 'video_led_signal',
           'process_view', 'pick_led_center', 'manual_led',
           'load_force', 'swap_force_plates', 'find_lag', 'sync', 'new_led']


# ============================================================= force data ===

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
    #Channel 3 is wired to the LED
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


# ============================================================== alignment ===

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


# =========================================================== entry points ===

def sync(view: str, parent_path: str, video_file: str, force_file: str,
         led_center: Optional[Tuple[int, int]] = None):
    #Create the config for this view, None if the view is unknown
    config = view_map[view]() if view in view_map else None
    #Error if the view isn't one of the known views
    if config is None:
        raise ValueError(f"Unknown view type {view!r}, expected one of {list(view_map)}")

    #Get the LED signal from the video, using the hand picked center if given
    df_video = process_view(view, os.path.join(parent_path, video_file), led_center)
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
    #Use a hand picked LED center for this view if the manual picker saved one
    state = getattr(self, 'state', None)
    led_center = getattr(state, 'manual_led_centers', {}).get(view) if state is not None else None
    #Run the sync for this view
    lag, df_frames, df_full = sync(view, parent_path, video_file, force_file, led_center)
    #Save the full force data on the app state if it has one
    if hasattr(self, 'state'):
        self.state.df_aligned_full = df_full
    return lag, df_frames
