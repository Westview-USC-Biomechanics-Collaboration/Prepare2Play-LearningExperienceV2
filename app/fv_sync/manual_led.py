"""
Manual LED Picker

Fallback for when find_led locks onto the wrong spot. Shows one frame of the
video, takes a click or a drag on the LED, and hands back those pixel
coordinates as the center video_led_signal samples.

Also has show_led_mask, a popup of the black and white mask find_led matches
on, to check what the auto detection is seeing.
"""

__author__ = 'Aiden Lee'

import os
from typing import List, Optional, Tuple
import cv2
import numpy as np

from .led_detection import (LEDConfig, get_config, open_video, clamp_center,
                            match_led)


# Biggest window we are willing to put on screen, a 1920x1080 frame gets shrunk to fit
max_display_width = 1280
max_display_height = 720

# Colors in BGR
color_marker = (0, 0, 255)      # crosshair and signal box
color_box = (0, 255, 255)       # crop region
color_match = (0, 255, 0)       # where the template matched on the current frame
color_text = (255, 255, 255)
color_text_bg = (0, 0, 0)

# Keys that step through the video, a/d = 1 frame, A/D = 30 frames
step_keys = (ord('a'), ord('d'), ord('A'), ord('D'))


def grab_frame(video_path: str, frame_index: int) -> np.ndarray:
    #Read the video
    cap = open_video(video_path)

    #Number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #Clamp the requested frame so we can't seek past the end
    frame_index = int(np.clip(frame_index, 0, max(total_frames - 1, 0)))

    #Jumps directly to the frame we want to show
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    #Reads the frame
    ok, frame = cap.read()
    #Release the cap
    cap.release()

    #Error if that frame couldn't be read
    if not ok:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def count_frames(video_path: str) -> int:
    #Read the video
    cap = open_video(video_path)

    #Number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #Release the cap
    cap.release()
    return total_frames


def display_scale(frame: np.ndarray) -> float:
    #Full frame size we are scaling down from
    height, width = frame.shape[:2]
    #Shrink to fit the window limits, never scale up, so a small video stays at its real size
    return min(max_display_width / width, max_display_height / height, 1.0)


def step_frame(key: int, frame_index: int, total_frames: int) -> int:
    #a/d step one frame, A/D step thirty, to find a frame where the LED is lit
    step = 1 if key in (ord('a'), ord('d')) else 30
    #a and A step backwards, d and D step forwards
    step = -step if key in (ord('a'), ord('A')) else step
    #Clamp so we stay inside the video
    return int(np.clip(frame_index + step, 0, max(total_frames - 1, 0)))


def draw_help(canvas: np.ndarray, lines: List[str]) -> None:
    #Draw each line bottom up, 22 px apart
    for i, line in enumerate(lines):
        y = canvas.shape[0] - 10 - (len(lines) - 1 - i) * 22
        #Size of the text, so the backing box fits it exactly
        (w, h), base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        #Dark box first so the text stays readable over the video, a thick black copy
        #of the text doesn't line up because thicker text is wider in OpenCV 5
        cv2.rectangle(canvas, (6, y - h - 4), (14 + w, y + base), color_text_bg, -1)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color_text, 1, cv2.LINE_AA)


def close_window(window: str) -> None:
    #Destroy the window
    cv2.destroyWindow(window)
    #Extra waitKey calls, some platforms need them to actually close the window
    for _ in range(4):
        cv2.waitKey(1)


def draw_overlay(frame: np.ndarray, center: Optional[Tuple[int, int]],
                 config: LEDConfig, frame_index: int, total_frames: int,
                 scale: float) -> np.ndarray:
    #Shrink the frame to window size
    canvas = cv2.resize(frame, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else frame.copy()

    #Draw the crop region find_led searches for debugging to know where it was searching
    cv2.rectangle(canvas,
                  (int(config.led_crop_x1 * scale), int(config.led_crop_y1 * scale)),
                  (int(config.led_crop_x2 * scale), int(config.led_crop_y2 * scale)),
                  color_box, 1)

    #Draw the picked center if there is one
    if center is not None:
        #Unpack the picked center
        cx, cy = center
        #Scale the point to window coordinates, for drawing only
        sx, sy = int(cx * scale), int(cy * scale)
        #Crosshair on the picked LED center
        cv2.drawMarker(canvas, (sx, sy), color_marker, cv2.MARKER_CROSS, 24, 2)
        #Half size of the box video_led_signal averages the red channel over
        d = config.signal_box
        #Draw that box so the sampled region is visible
        cv2.rectangle(canvas,
                      (int((cx - d) * scale), int((cy - d) * scale)),
                      (int((cx + d) * scale), int((cy + d) * scale)),
                      color_marker, 1)

    #Current center as text, or a placeholder before anything is picked
    position = f"{center}" if center is not None else "none yet"
    #Help text along the bottom
    draw_help(canvas, [
        f"{config.view_name}  frame {frame_index}/{max(total_frames - 1, 0)}  LED: {position}",
        "click or drag onto the LED   a/d = step 1 frame   A/D = step 30",
        "ENTER = confirm   r = clear   ESC = cancel",
    ])

    return canvas

def pick_led_center(video_path: str, view_type: str,
                    frame_index: int = 10) -> Optional[Tuple[int, int]]:
    """
    Opens a frame to pick the LED center and returns that center
    """
    #Create the config for the specific view drawing the cropped box, errors if the view is unknown
    config = get_config(view_type)
    #Number of frames
    total_frames = count_frames(video_path)
    #The frame currently on screen
    frame = grab_frame(video_path, frame_index)
    #How much that frame is shrunk to fit the window
    scale = display_scale(frame)

    #State modified by the mouse callback
    state = {'center': None, 'dragging': False}

    def on_mouse(event, x, y, flags, param):
        #Convert window coordinates to the full frame coordinates, clamped so the
        #signal box we search doesn't go off frame
        fx, fy = clamp_center((round(x / scale), round(y / scale)),
                              frame.shape[1], frame.shape[0], config.signal_box)

        #Pressing the button sets the point and starts a drag
        if event == cv2.EVENT_LBUTTONDOWN:
            state['dragging'] = True
            state['center'] = (fx, fy)
        #Dragging keeps updating the point, for fine tuning
        elif event == cv2.EVENT_MOUSEMOVE and state['dragging']:
            state['center'] = (fx, fy)
        #Releasing ends the drag and keeps the last point
        elif event == cv2.EVENT_LBUTTONUP:
            state['dragging'] = False
            state['center'] = (fx, fy)

    #Window title
    window = f"Pick LED - {config.view_name}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    #Route the mouse events onto on_mouse
    cv2.setMouseCallback(window, on_mouse)

    #The point we hand back, stays None if the user cancels
    result = None
    try:
        #Keep redrawing until the user confirms or cancels
        while True:
            #Redraw with the current point and help text
            cv2.imshow(window, draw_overlay(frame, state['center'], config,
                                            frame_index, total_frames, scale))
            #Wait 20 ms for a key, short enough to keep the drag smooth
            key = cv2.waitKey(20) & 0xFF

            #ESC cancels everything
            if key == 27:
                break
            #ENTER confirms double checking, but only once a point has been picked
            if key in (13, 10):
                if state['center'] is not None:
                    result = state['center']
                    break
                print("[manual LED] click the LED first, nothing picked yet")
            #r clears the point so you can start over
            elif key == ord('r'):
                state['center'] = None
            #Step through the video to find a frame where the LED is lit
            elif key in step_keys:
                new_index = step_frame(key, frame_index, total_frames)
                #Only re-read if the frame changed
                if new_index != frame_index:
                    frame_index = new_index
                    frame = grab_frame(video_path, frame_index)

            #Close the window with the x button
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        close_window(window)

    #Log the result so the console matches the auto detection path
    if result is None:
        print("[manual LED] cancelled, no center picked")
    else:
        print(f"[{config.view_name}] LED set manually at {result}")
    return result


def draw_mask(frame: np.ndarray, center: Optional[Tuple[int, int]],
              config: LEDConfig, frame_index: int, total_frames: int) -> np.ndarray:
    #Crop the frame to the region find_led searches
    crop = config.get_crop_region(frame)
    #Same blue minus green image the template is matched against
    processed = config.process_crop_for_matching(crop)
    #Otsu picks the threshold that best splits it into black and white
    _, mask = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #Where the template matched on this frame alone, in crop coordinates
    mx, my = match_led(frame, config)
    mx, my = mx - config.led_crop_x1, my - config.led_crop_y1
    #Template outline: back from the center to its top-left corner
    th, tw = config.led_template.shape
    tx, ty = mx - config.template_center_offset_x, my - config.template_center_offset_y

    #Left panel is the color crop, right panel is the mask, both get the same markings
    panels = [crop.copy(), cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]
    for panel in panels:
        #This frame's template match
        cv2.rectangle(panel, (tx, ty), (tx + tw - 1, ty + th - 1), color_match, 1)
        cv2.drawMarker(panel, (mx, my), color_match, cv2.MARKER_CROSS, 12, 1)
        #The box the red signal is actually averaged over
        if center is not None:
            cx, cy = center[0] - config.led_crop_x1, center[1] - config.led_crop_y1
            d = config.signal_box
            cv2.rectangle(panel, (cx - d, cy - d), (cx + d, cy + d), color_marker, 1)

    #Side by side, then scale to fill the window, up or down since some crops are tiny
    canvas = np.hstack(panels)
    scale = min(max_display_width / canvas.shape[1], max_display_height / canvas.shape[0])
    #Nearest neighbour keeps the mask pixels sharp when scaling up
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    #The center as text, or a placeholder if none was given
    position = f"{center}" if center is not None else "none"
    #Help text along the bottom
    draw_help(canvas, [
        f"{config.view_name}  frame {frame_index}/{max(total_frames - 1, 0)}  "
        f"LED used: {position}  this frame: {(mx + config.led_crop_x1, my + config.led_crop_y1)}",
        "left = crop  right = mask   green = this frame's match   red = signal box",
        "a/d = step 1 frame   A/D = step 30   ENTER/ESC = close",
    ])
    return canvas


def show_led_mask(video_path: str, view_type: str,
                  center: Optional[Tuple[int, int]] = None,
                  frame_index: int = 10) -> None:
    """
    Opens a popup of the black and white mask the auto detection matches on
    """
    #Create the config for the specific view, errors if the view is unknown
    config = get_config(view_type)
    #Number of frames
    total_frames = count_frames(video_path)
    #The frame currently on screen
    frame = grab_frame(video_path, frame_index)

    #Window title
    window = f"LED mask - {config.view_name}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        #Keep redrawing until the user closes it
        while True:
            cv2.imshow(window, draw_mask(frame, center, config, frame_index, total_frames))
            key = cv2.waitKey(20) & 0xFF

            #ENTER or ESC closes
            if key in (13, 10, 27):
                break
            #Step through the video to compare LED on and off frames
            if key in step_keys:
                new_index = step_frame(key, frame_index, total_frames)
                #Only re-read if the frame changed
                if new_index != frame_index:
                    frame_index = new_index
                    frame = grab_frame(video_path, frame_index)

            #Close the window with the x button
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        close_window(window)


def manual_led(self, view, parent_path, video_file, frame_index=0):
    """
    Start of GUI
    """
    #Build the full path to the video the same way sync does
    video_path = os.path.join(parent_path, video_file)
    #Open the picker and wait for the user
    center = pick_led_center(video_path, view, frame_index)

    #Save the picked center
    if center is not None and hasattr(self, 'state'):
        #Make the dict the first time a center is picked
        if not hasattr(self.state, 'manual_led_centers'):
            self.state.manual_led_centers = {}
        #Keyed by the video's full path so a new video for the same view doesn't reuse it
        self.state.manual_led_centers[os.path.abspath(video_path)] = center

    return center
