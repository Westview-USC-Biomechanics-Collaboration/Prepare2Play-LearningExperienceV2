import cv2
import io
import numpy as np
import pandas as pd
import time
import os
import PIL
import glob
from matplotlib import pyplot as plt
from scipy import signal
from scipy.stats import cumfreq
from scipy import ndimage, datasets

class VectorOverlay:
    fps_annotated = 120
    force_scaling = 0.8
    video = cap.VideoCapture(video)

    def __init__(self, video_path)
    video = cv2.VideoCapture(video_path)
    self.frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_capture.relase()
    def matrix_transformation(self, pointx, pointy, view):
        #something something
        point_x_transformed = 5 * pointx

        point_y_transformed = 5 * pointy

        #This is random made up stuff for real function

        return point_x_transformed, point_y_transformed
        

    frame_width = int(cap.get(cap.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cap.CAP_PROP_FRAME_HEIGHT))
    def draw_vectors_on_video(self, df_aligned_and_scaled, video, view):
       
        for index, row in df_aligned_and_scaled.iterrows():

            video.set(cap.CAP_PROP_POS_FRAMES, index)
            ret, frame = video.read()
            if not ret:
                break

            #Sets start and end points for the vectors based on the force data in the dataframe
            if view.lower() == "long": 
                #Fx is the point of pressure in the x direction, Fy is the point of pressure in the y direction
                start_vector_x_fp1, start_vector_y_fp1 = self.matrix_transformation([int(row['FP1_Az'])], [int(row['FP1_Ay'])], view)
            
                #Fx is the magnitude and direction of the force vector in the x direction, Fy is the magnitude and direction of the force vector in the y direction
                #FP1_Ax -> pressure point,  FP1_Fx -> Magnitude+Direction
                end_vector_x_fp1 = [start_vector_x_fp1 + int(row['FP1_Fz'])]

                end_vector_y_fp1 = [start_vector_y_fp1 + int(row['FP1_Fy'])]

                start_vector_x_fp2, start_vector_y_fp2 = self.matrix_transformation([int(row['FP2_Az'])], [int(row['FP2_Ay'])], view)

                end_vector_x_fp2 = [start_vector_x_fp2 + int(row['FP2_Fz'])]

                end_vector_y_fp2 = [start_vector_y_fp2 + int(row['FP2_Fy'])]
            elif view.lower() == "top":
                start_vector_x_fp1, start_vector_y_fp1 = self.matrix_transformation([int(row['FP1_Ax'])], [int(row['FP1_Ay'])], view)

                end_vector_x_fp1 = [start_vector_x_fp1 + int(row['FP1_Fx'])]

                end_vector_y_fp1 = [start_vector_y_fp1 + int(row['FP1_Fy'])]

                start_vector_x_fp2, start_vector_y_fp2 = self.matrix_transformation([int(row['FP2_Ax'])], [int(row['FP2_Ay'])], view)

                end_vector_x_fp2 = [start_vector_x_fp2 + int(row['FP2_Fx'])]

                end_vector_y_fp2 = [start_vector_y_fp2 + int(row['FP2_Fy'])]
            elif view.lower() == "short":
                start_vector_x_fp1, start_vector_y_fp1 = self.matrix_transformation([int(row['FP1_Ax'])], [int(row['FP1_Az'])], view)
            
                end_vector_x_fp1 = [start_vector_x_fp1 + int(row['FP1_Fx'])]

                end_vector_y_fp1 = [start_vector_y_fp1 + int(row['FP1_Fz'])]

                start_vector_x_fp2, start_vector_y_fp2 = self.matrix_transformation([int(row['FP2_Ax'])], [int(row['FP2_Az'])], view)

                end_vector_x_fp2 = [start_vector_x_fp2 + int(row['FP2_Fx'])]

                end_vector_y_fp2 = [start_vector_y_fp2 + int(row['FP2_Fz'])]
            else:
                raise ValueError(f"Unknown view: {view!r}")
            

            cap.arrowedLine(frame, (start_vector_x_fp1, start_vector_y_fp1), (end_vector_x_fp1, end_vector_y_fp1), (0, 255, 0), 2)
            cap.arrowedLine(frame, (start_vector_x_fp2, start_vector_y_fp2), (end_vector_x_fp2, end_vector_y_fp2), (0, 0, 255), 2)
