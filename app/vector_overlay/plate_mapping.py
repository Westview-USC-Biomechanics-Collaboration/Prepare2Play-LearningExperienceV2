"""
Create a transformation matrix based off forceplate corners to convert
real world points to coordinates on screen.
"""

import cv2 as cv
import numpy as np

class PlateMapping:
    def __init__(self, video_path):
        self.video_path = video_path
        
    def find_corners(self):
        """
        Analyzes video and returns the average corner placement of forceplates.
        """
        cap = cv.VideoCapture(self.video_path)
        
        if cap.get(cv.CAP_PROP_FRAME_COUNT) < 200:
            print("Video Too Short")
            cap.release()
            return None
            
        # Order: bottom left, top left, top right, bottom right
        corners = np.zeros((4, 2), dtype=np.float32)
            
        x_cutoff = 400
        y_cutoff = 700
        
        low_yellow = np.array([23, 50, 80])
        high_yellow = np.array([50, 255, 255])
        
        frame_index = 0
        good_frames = 0
        while frame_index < 200:
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
            kernel = np.ones((10, 10), np.uint8)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
            # Add blurring (helps to smoothen out lines)
            mask = cv.GaussianBlur(mask, (7, 7), 0)
            
            ###               Find the corners                     ###
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # If there are no contours, skip this frame.
            if len(contours) == 0:
                frame_index += 1
                continue            
            # Otherwise, take the largest contour's corners (AKA the forceplate)
            contour = max(contours, key=cv.contourArea)
            peri = cv.arcLength(contour, True)
            frame_corners = cv.approxPolyDP(contour, 0.01 * peri, True)
            
            if len(frame_corners) == 4:
                good_frames += 1
                frame_corners = frame_corners.reshape(-1, 2)
                frame_corners = sorted(frame_corners, key=lambda x: x[0])
                frame_corners = np.array(frame_corners) + [x_cutoff, y_cutoff]
                corners += frame_corners
            
            frame_index += 1

        if good_frames == 0:
            print("No Good Frames Found")
            cap.release()
            return None
        
        corners /= good_frames

        return corners;


    def CreateTransformationMatrix(self):
        """
        With real life points and screen pixels, create the transformation matrix.
        """
        
        #   +0.300  TL ┌──────────┐     ┌──────────┐ TR
        #              │ Plate 1  │ gap │ Plate 2  │
        #   -0.300  BL └──────────┘     └──────────┘ BR
        # 
        #     -0.902                0                +0.902

        # May tweak later to match actual forceplate sizes
        forceplate_x = 0.902
        forceplate_y = 0.300
        
        # Order: BL TL TR BR
        forceplate_on_screen = self.find_corners()
        forceplate_real_life = np.float32([[-forceplate_x, -forceplate_y],
                                           [-forceplate_x, forceplate_y],
                                           [forceplate_x, forceplate_y],
                                           [forceplate_x, -forceplate_y]])
        
        matrix = cv.getPerspectiveTransform(forceplate_real_life, forceplate_on_screen)
        
        return matrix
        
        
    # def ConvertPressurePointToScreenCoord(self):
