"""
Sync Plot

Saves a picture of the two LED signals after alignment, so a bad sync is easy
to spot by eye.
"""

__author__ = 'Aiden Lee'

import numpy as np
import pandas as pd
# Figure instead of pyplot so saving never opens a window or fights with cv2's
from matplotlib.figure import Figure

# How many frames the zoomed panel shows, 4 seconds at 120 fps
zoom_frames = 480


def plot_sync(df_frames: pd.DataFrame, lags: np.ndarray, corr: np.ndarray,
              lag: int, r: float, peak_ratio: float, title: str,
              save_path: str) -> None:
    #Only the frames that have both a video and a force LED value
    df = df_frames.dropna(subset=['Video_LED_Signal', 'FP_LED_Signal'])
    frames = df['FrameNumber'].to_numpy()
    video = df['Video_LED_Signal'].to_numpy()
    force = df['FP_LED_Signal'].to_numpy()
    #Share of frames where the two LEDs agree on on/off
    agree = np.mean((video > 0) == (force > 0)) if len(df) else 0.0

    fig = Figure(figsize=(12, 9), layout='constrained')
    ax_corr, ax_full, ax_zoom = fig.subplots(3, 1)
    fig.suptitle(f"{title}: lag = {lag} frames, r = {r:.2f}, "
                 f"peak ratio = {peak_ratio:.2f}, LEDs agree on {agree:.0%} of frames")

    #Correlation for every lag tried, one clear spike means a confident sync
    ax_corr.plot(lags, corr, lw=1)
    ax_corr.axvline(lag, color='red', ls='--', lw=1, label=f'best lag {lag}')
    ax_corr.set_xlabel('lag (frames)')
    ax_corr.set_ylabel('correlation')
    ax_corr.legend(loc='upper right')

    #Both signals over the whole video, video drawn above force so they don't overlap
    for ax in (ax_full, ax_zoom):
        ax.step(frames, video + 3, where='post', lw=1, label='video LED')
        ax.step(frames, force, where='post', lw=1, label='force LED')
        ax.set_yticks([0, 3], ['force', 'video'])
        ax.set_xlabel('video frame')
    ax_full.legend(loc='upper right')

    #Zoom in starting just before the first time the video LED changes
    if len(frames):
        changes = np.flatnonzero(np.diff(video) != 0)
        start = frames[changes[0]] - 60 if len(changes) else frames[0]
        ax_zoom.set_xlim(start, start + zoom_frames)
    ax_zoom.set_title('zoomed, edges should line up')

    fig.savefig(save_path, dpi=100)
