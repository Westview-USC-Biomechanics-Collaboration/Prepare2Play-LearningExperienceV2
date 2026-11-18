"Vector Overlay Module"
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
    video = cv2.VideoCapture()
    frame_width = 0
    frame_height = 0
    video_fps = 0

    # Stores video width, height, fps, frame number
    def store_frame_data(self):
        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.video_frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Video Properties:")
        print(f"  Width: {self.frame_width}")
        print(f"  Height: {self.frame_height}")
        print(f"  FPS: {self.video_fps}")
        print(f"  Frame Count: {self.video_frame_count}")
        
    # Copies force data, finds largest force from the dataframe, divides each force by
    # largest force to convert the dataframe to values between 0 and 1. Multiplies each 
    # value by a scale factor of the frame height.
    def copy_and_scale_force_data(self, df_aligned):
        """
        Copies and scales the force data.

        Parameters:
        self 
        alligned dataframe(important since we want each vector to correspond to its frame)
        scale_factor()
        Returns:
        Scaled Dataframe with scaled force data.
        """
        # Copy the force data from the DataFrame
        force_data = df_aligned.copy()

        # Calculate maximum absolute values out of all forces
        max_z_force = max(force_data['FP1_Az'].abs().max(), df_aligned['FP2_Az'].abs().max())
        max_x_force = max(force_data['FP1_Ax'].abs().max(), df_aligned['FP2_Ax'].abs().max())
        max_y_force = max(force_data['FP1_Ay'].abs().max(), df_aligned['FP2_Ay'].abs().max())
        max_total_force = max(max_x_force, max_y_force, max_z_force)

        # Sets each force to a value between 0 and 1(1 being the maximum force) in order to apply scale factor

        force_magnitudes = ['FP1_Az', 'FP1_Ay', 'FP2_Az', 'FP2_Ay', 'FP1_Ax', 'FP2_Ax'] 
        for col in force_magnitudes:
            force_data[col] = (force_data[col] / max_total_force).round(4)

        # Determines and applies scale factor to force 0-1 values
        # Max height of a force is 0.8 frame height
        scale_factor = self.frame_height*self.force_scaling
        scale_factor = scale_factor.astype(int)

        for col in force_magnitudes:
            force_data[col] *= scale_factor.astype(int)
        return force_data


class LongViewVectorOverlay(VectorOverlay):
    def __init__(self, frame_height, force_scaling):
        self.frame_height = frame_height
        self.force_scaling = force_scaling

        self.video = cv2.VideoCapture()
        self.store_frame_data()

    # Uses aligned dataframe force and pressure values to define start and end points for vectors
    # Start Point + vector direction & magnitude = End Point
    # Draws vectors using cv2
    def draw_vectors_on_video(self, df_aligned_and_scaled, video):
        for index, row in df_aligned_and_scaled.iterrows():

            video.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = video.read()
            if not ret:
                break

            #Sets start and end points for the vectors based on the force data in the dataframe

            #Fx is the point of pressure in the x direction, Fy is the point of pressure in the y direction
            start_vector_z_fp1, start_vector_y_fp1 = self.matrix_transformation_long([int(row['Fp1_Fz'])], [int(row['Fp1_Fy'])])

            #Ax is the magnitude and direction of the force vector in the x direction, Ay is the magnitude and direction of the force vector in the y direction
            end_vector_z_fp1 = [start_vector_z_fp1 + int(row['FP1_Az'])]

            end_vector_y_fp1 = [start_vector_y_fp1 + int(row['FP1_Ay'])]

            start_vector_z_fp2, start_vector_y_fp2 = self.matrix_transformation_long([int(row['Fp2_Fz'])], [int(row['Fp2_Fy'])])


            end_vector_z_fp2 = [start_vector_z_fp2 + int(row['FP2_Az'])]

            end_vector_y_fp2 = [start_vector_y_fp2 + int(row['FP2_Ay'])]

            cv2.arrowedLine(frame, (start_vector_z_fp1, start_vector_y_fp1), (end_vector_z_fp1, end_vector_y_fp1), (0, 255, 0), 2)
            cv2.arrowedLine(frame, (start_vector_z_fp2, start_vector_y_fp2), (end_vector_z_fp2, end_vector_y_fp2), (0, 0, 255), 2)

class ShortAViewVectorOverlay(VectorOverlay):
    def __init__(self, frame_height, force_scaling):
        self.frame_height = frame_height
        self.force_scaling = force_scaling

        self.video = cv2.VideoCapture()
        self.store_frame_data()
    def draw_vectors_on_video(self, df_aligned_and_scaled, video):
        for index, row in df_aligned_and_scaled.iterrows():
            video.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = video.read()
            if not ret:
                break

            #Sets start and end points for the vectors based on the force data in the dataframe

            #Fx is the point of pressure in the x direction, Fy is the point of pressure in the y direction
            start_vector_z_fp1, start_vector_x_fp1 = self.matrix_transformation_long([int(row['Fp1_Fz'])], [int(row['Fp1_Fx'])])

            #Ax is the magnitude and direction of the force vector in the x direction, Ay is the magnitude and direction of the force vector in the y direction
            end_vector_z_fp1 = [start_vector_z_fp1 + int(row['FP1_Az'])]

            end_vector_x_fp1 = [start_vector_x_fp1 - int(row['FP1_Ax'])] 
            # Subtracting because the x-axis is inverted in this view

            start_vector_z_fp2, start_vector_x_fp2 = self.matrix_transformation_long([int(row['Fp2_Fz'])], [int(row['Fp2_Fx'])])

            end_vector_z_fp2 = [start_vector_z_fp2 + int(row['FP2_Az'])]

            end_vector_x_fp2 = [start_vector_x_fp2 - int(row['FP2_Ax'])]
            # Subtracting because the x-axis is inverted in this view

            cv2.arrowedLine(frame, (start_vector_z_fp1, start_vector_x_fp1), (end_vector_z_fp1, end_vector_x_fp1), (0, 255, 0), 2)
            cv2.arrowedLine(frame, (start_vector_z_fp2, start_vector_x_fp2), (end_vector_z_fp2, end_vector_x_fp2), (0, 0, 255), 2)


class ShortBViewVectorOverlay(VectorOverlay):  
    def __init__(self, frame_height, force_scaling):
        self.frame_height = frame_height
        self.force_scaling = force_scaling

        self.video = cv2.VideoCapture()
        self.store_frame_data()
    def draw_vectors_on_video(self, df_aligned_and_scaled, video):
        for index, row in df_aligned_and_scaled.iterrows():
            video.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = video.read()
            if not ret:
                break

            #Sets start and end points for the vectors based on the force data in the dataframe

            #Fx is the point of pressure in the x direction, Fy is the point of pressure in the y direction
            start_vector_z_fp1, start_vector_x_fp1 = self.matrix_transformation_long([int(row['Fp1_Fz'])], [int(row['Fp1_Fx'])])

            #Ax is the magnitude and direction of the force vector in the x direction, Ay is the magnitude and direction of the force vector in the y direction
            end_vector_z_fp1 = [start_vector_z_fp1 + int(row['FP1_Az'])]

            end_vector_x_fp1 = [start_vector_x_fp1 + int(row['FP1_Ax'])] 
            # Adding because the x-axis is not inverted in this view

            start_vector_z_fp2, start_vector_x_fp2 = self.matrix_transformation_long([int(row['Fp2_Fz'])], [int(row['Fp2_Fx'])])

            end_vector_z_fp2 = [start_vector_z_fp2 + int(row['FP2_Az'])]

            end_vector_x_fp2 = [start_vector_x_fp2 + int(row['FP2_Ax'])]
            # Adding because the x-axis is not inverted in this view

            cv2.arrowedLine(frame, (start_vector_z_fp1, start_vector_x_fp1), (end_vector_z_fp1, end_vector_x_fp1), (0, 255, 0), 2)
            cv2.arrowedLine(frame, (start_vector_z_fp2, start_vector_x_fp2), (end_vector_z_fp2, end_vector_x_fp2), (0, 0, 255), 2)




class TopViewVectorOverlay(VectorOverlay):
    def __init__(self, frame_height, force_scaling):
        self.frame_height = frame_height
        self.force_scaling = force_scaling

        self.video = cv2.VideoCapture()
        self.store_frame_data()
    def draw_vectors_on_video(self, df_aligned_and_scaled, video):
        for index, row in df_aligned_and_scaled.iterrows():
        
            video.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = video.read()
            if not ret:
                break
        
            #Sets start and end points for the vectors based on the force data in the dataframe
        
            #Fx is the point of pressure in the x direction, Fy is the point of pressure in the y direction
            start_vector_x_fp1, start_vector_y_fp1 = self.matrix_transformation_long([int(row['Fp1_Fx'])], [int(row['Fp1_Fy'])])                    #Ax is the magnitude and direction of the force vector in the x direction, Ay is the magnitude and direction of the force vector in the y direction
            end_vector_x_fp1 = [start_vector_x_fp1 + int(row['FP1_Ax'])]
            end_vector_y_fp1 = [start_vector_y_fp1 + int(row['FP1_Ay'])]
        
            start_vector_x_fp2, start_vector_y_fp2 = self.matrix_transformation_long([int(row['Fp2_Fx'])], [int(row['Fp2_Fy'])])
        
        
            end_vector_x_fp2 = [start_vector_x_fp2 + int(row['FP2_Ax'])]
        
            end_vector_y_fp2 = [start_vector_y_fp2 + int(row['FP2_Ay'])]
        
            cv2.arrowedLine(frame, (start_vector_x_fp1, start_vector_y_fp1), (end_vector_x_fp1, end_vector_y_fp1), (0, 255, 0), 2)
            cv2.arrowedLine(frame, (start_vector_x_fp2, start_vector_y_fp2), (end_vector_x_fp2, end_vector_y_fp2), (0, 0, 255), 2)










