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
    " abt fps"
    force_scaling = 0.8

    frame_width = int(cv2.VideoCapture().get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cv2.VideoCapture().get(cv2.CAP_PROP_FRAME_HEIGHT))

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


    


    

    
