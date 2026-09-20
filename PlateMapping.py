"""
Create a transformation matrix based off forceplate corners to convert
real world points to coordinates on screen.
"""

import cv2 as cv
import numpy as np

class PlateMapping:
    def __init__(self, video_path):
        self.video_path = video_path
        
    def FindCorners(self):
        cap = cv.VideoCapture(self.video_path)
        
        if cap.get(cv.CAP_PROP_FRAME_COUNT) < 20:
            print("Video Too Short")
            cap.release()
            return None
            
        corners = [
            (0, 0), (0, 0), (0, 0), (0, 0)
        ]
            
        x_cutoff = 400
        y_cutoff = 700
        
        low_yellow = np.array([25, 60, 90])
        high_yellow = np.array([40, 255, 255])
        
        frame_index = 0
        good_frames = 0
        while frame_index < 50:
            ret, frame = cap.read()
            if not ret:
                print("Can't Find Frame")
                break

            ###                Crop the frame                      ###
            frame = frame[y_cutoff:, x_cutoff:]
            frame = frame[:300, :1000]

            ###       Filtering frame to make border obvious       ###
            # Change to HSV
            mask = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            # Highlight yellow colors
            mask = cv.inRange(mask, low_yellow, high_yellow)
            # Close gaps
            kernel = np.ones((20, 20), np.uint8)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
            
            ###               Find the corners                     ###
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            # No contours found
            if len(contours) == 0:
                continue            
            contour = max(contours, key=cv.contourArea)
            
            # Otherwise take the largest (Forceplate) and find it's corners
            peri = cv.arcLength(contour, True)
            frame_corners = cv.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(frame_corners) == 4:
                good_frames += 1
                frame_corners = frame_corners.reshape(-1, 2)
                frame_corners = sorted(frame_corners, key=lambda x: x[0])
                
                
                frame_corners = np.array(frame_corners) + [x_cutoff, y_cutoff]
                corners += frame_corners
                
            frame_index += 1

        if good_frames == 0:
            print("No Good Frames Found")
            return None
        
        # Take the average corner placements
        corners /= good_frames

        return corners;


    # def CreateTransformationMatrix(self):
        
    # def ConvertPressurePointToScreenCoord(self):