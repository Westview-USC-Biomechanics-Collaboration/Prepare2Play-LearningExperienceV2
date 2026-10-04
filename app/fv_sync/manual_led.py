"""
Manual LED Picker

Fallback for when find_led locks onto the wrong spot. Shows one frame of the
video, takes a click or a drag on the LED, and hands back those pixel
coordinates as the center video_led_signal samples.
"""

__author__ = 'Aiden Lee'

import os
from typing import Optional, Tuple
import cv2
import numpy as np

from .led_detection import LEDConfig, view_map


# Biggest window we are willing to put on screen, a 1920x1080 frame gets shrunk to fit
max_display_width = 1280
max_display_height = 720

# Colors in BGR
color_marker = (0, 0, 255)      # crosshair and signal box
color_box = (0, 255, 255)       # crop region
color_text = (255, 255, 255)
color_text_bg = (0, 0, 0)


def grab_frame(video_path: str, frame_index: int) -> np.ndarray:
    #Read the video
    cap = cv2.VideoCapture(video_path)
    #Error if the video can't be read
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

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
    cap = cv2.VideoCapture(video_path)
    #Error if the video can't be read
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

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
    lines = [
        f"{config.view_name}  frame {frame_index}/{max(total_frames - 1, 0)}  LED: {position}",
        "click or drag onto the LED   a/d = step 1 frame   A/D = step 30",
        "ENTER = confirm   r = clear   ESC = cancel",
    ]
    #Draw each line bottom up, 22 px apart
    for i, line in enumerate(lines):
        y = canvas.shape[0] - 10 - (len(lines) - 1 - i) * 22
        #Dark backing first so the text stays readable over the video
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color_text_bg, 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color_text, 1, cv2.LINE_AA)

    return canvas

def pick_led_center(video_path: str, view_type: str,
                    frame_index: int = 10) -> Optional[Tuple[int, int]]:
    """
    Opens a frame to pick the LED center and returns that center
    """
    #Error if the view isn't one of the known views
    if view_type not in view_map:
        raise ValueError(f"Unknown view type {view_type!r}, "
                         f"expected one of {list(view_map)}")

    #Create the config for the specific view drawing the cropped box
    config = view_map[view_type]()
    #Number of frames
    total_frames = count_frames(video_path)
    #The frame currently on screen
    frame = grab_frame(video_path, frame_index)
    #How much that frame is shrunk to fit the window
    scale = display_scale(frame)

    #State modified by the mouse callback
    state = {'center': None, 'dragging': False}

    def on_mouse(event, x, y, flags, param):
        #Convert window coordinates to the full frame coordinates for the pixel coordinates
        fx = int(round(x / scale))
        fy = int(round(y / scale))
        #Clamping the point so the signal box we search doesn't go off frame
        fx = int(np.clip(fx, config.signal_box, frame.shape[1] - config.signal_box - 1))
        fy = int(np.clip(fy, config.signal_box, frame.shape[0] - config.signal_box - 1))

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
            #a/d step one frame, A/D step thirty, to find a frame where the LED is lit
            elif key in (ord('a'), ord('d'), ord('A'), ord('D')):
                step = 1 if key in (ord('a'), ord('d')) else 30
                #a and A step backwards, d and D step forwards
                step = -step if key in (ord('a'), ord('A')) else step
                #Clamp so we stay inside the video
                new_index = int(np.clip(frame_index + step, 0, max(total_frames - 1, 0)))
                #Only re-read if the center is moved
                if new_index != frame_index:
                    frame_index = new_index
                    frame = grab_frame(video_path, frame_index)

            #Close the window with the x button
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        #Destroy the window
        cv2.destroyWindow(window)
        #Extra waitKey calls, some platforms need them to actually close the window
        for _ in range(4):
            cv2.waitKey(1)

    #Log the result so the console matches the auto detection path
    if result is None:
        print("[manual LED] cancelled, no center picked")
    else:
        print(f"[{config.view_name}] LED set manually at {result}")
    return result


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
        self.state.manual_led_centers[view] = center

    return center
