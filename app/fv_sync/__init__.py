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
                            get_config, find_led, video_led_signal, process_view)
from .manual_led import pick_led_center, manual_led, show_led_mask
from .sync_plot import plot_sync

__all__ = ['LEDConfig', 'LEDDetectionLong', 'LEDDetectionTop', 'LEDDetectionSide1',
           'LEDDetectionSide2', 'view_map', 'get_config', 'find_led',
           'video_led_signal', 'process_view', 'pick_led_center', 'manual_led',
           'show_led_mask', 'plot_sync', 'load_force', 'swap_force_plates',
           'correlate_signals', 'find_lag', 'sync', 'sync_led']


# Force plate runs at 1200 Hz, video at 120 fps -> 10 force rows per video frame.
force_per_fps = 10

# Below this correlation (-1 to 1) at the best lag the sync is probably wrong
min_correlation = 0.7
# The best peak has to beat the next best peak by this much, or the lag is ambiguous
min_peak_ratio = 1.2
# Lags this close to the best one count as the same peak, half a second at 120 fps
peak_exclusion_frames = 60

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

def normalize(a: np.ndarray) -> np.ndarray:
    #Normalize a signal to mean 0 and std 1 so both signals are on the same scale
    a = np.asarray(a, dtype=float)
    #Mean and std ignoring missing samples
    mean, std = np.nanmean(a), np.nanstd(a)
    #Skip dividing if the signal is flat (std 0)
    a = (a - mean) / std if std > 0 else a - mean
    #Missing samples become 0 (neutral), one NaN would otherwise make every score NaN
    return np.nan_to_num(a, nan=0.0)


def correlate_signals(video_sig: np.ndarray,
                      force_sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #Slide the force signal along the video signal and score how well they match
    corr = sps.correlate(normalize(video_sig), normalize(force_sig), mode="valid")
    #The frame offset for each correlation score
    lags = sps.correlation_lags(len(video_sig), len(force_sig), mode="valid")
    return lags, corr


def find_lag(video_sig: np.ndarray, force_sig: np.ndarray) -> Tuple[int, float, float]:
    #Correlation for every lag
    lags, corr = correlate_signals(video_sig, force_sig)
    #Index of the best match
    best = int(np.argmax(corr))
    lag = int(lags[best])

    #Pearson r over just the frames the two signals share at the best lag
    v, f = normalize(video_sig), normalize(force_sig)
    start, stop = max(0, -lag), min(len(f), len(v) - lag)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.corrcoef(v[start + lag:stop + lag], f[start:stop])[0, 1]
    #A flat signal gives NaN, which means no match at all
    r = float(r) if np.isfinite(r) else 0.0

    #Best peak against the best one outside of it, near 1 means two lags fit about as well
    outside = np.abs(lags - lag) > peak_exclusion_frames
    second = corr[outside].max() if outside.any() else 0.0
    peak_ratio = float(corr[best] / second) if second > 0 else float('inf')

    #Return the best lag, its correlation, and how much it beats the runner up
    return lag, r, peak_ratio


def sync(view: str, parent_path: str, video_file: str, force_file: str,
         led_center: Optional[Tuple[int, int]] = None, show_mask: bool = False,
         save_plot: bool = True):
    #Create the config for this view, errors if the view is unknown
    config = get_config(view)

    #Get the LED signal from the video, using the hand picked center if given
    df_video = process_view(view, os.path.join(parent_path, video_file),
                            led_center, show_mask)
    #Load the force plate data
    df_force = load_force(os.path.join(parent_path, force_file))

    #Keep every 10th force row so force matches the video frame rate
    df_dec = df_force.iloc[::force_per_fps].reset_index(drop=True)
    #Find how many frames the force data is shifted from the video
    video_sig = df_video['Video_LED_Signal'].to_numpy()
    force_sig = df_dec['FP_LED_Signal'].to_numpy()
    lag, r, peak_ratio = find_lag(video_sig, force_sig)
    print(f"[{view}] lag = {lag} frames, r = {r:.2f}, peak ratio = {peak_ratio:.2f}")
    #Warn when the match is weak or another lag fits nearly as well
    if r < min_correlation or peak_ratio < min_peak_ratio:
        print(f"[WARN] {view}: low sync confidence (r < {min_correlation} or "
              f"peak ratio < {min_peak_ratio}), check the sync plot")

    #Give each downsampled force row the video frame it lines up with
    df_dec['FrameNumber'] = np.arange(lag, lag + len(df_dec))
    #Join the force and video data on frame number
    df_frames = df_dec.merge(df_video, on='FrameNumber', how='left')

    #Save a picture of the aligned LED signals next to the video
    if save_plot:
        plot_path = os.path.join(parent_path,
                                 f"{os.path.splitext(video_file)[0]}_sync_check.png")
        lags, corr = correlate_signals(video_sig, force_sig)
        plot_sync(df_frames, lags, corr, lag, r, peak_ratio, view, plot_path)
        print(f"[{view}] sync plot saved to {plot_path}")

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


def sync_led(self, view, parent_path, video_file, force_file, show_mask=False):
    #Use a hand picked LED center for this video if the manual picker saved one
    state = getattr(self, 'state', None)
    video_path = os.path.abspath(os.path.join(parent_path, video_file))
    led_center = getattr(state, 'manual_led_centers', {}).get(video_path) if state is not None else None
    #Run the sync for this view
    lag, df_frames, df_full = sync(view, parent_path, video_file, force_file,
                                   led_center, show_mask)
    #Save the full force data on the app state if it has one
    if hasattr(self, 'state'):
        self.state.df_aligned_full = df_full
    return lag, df_frames
