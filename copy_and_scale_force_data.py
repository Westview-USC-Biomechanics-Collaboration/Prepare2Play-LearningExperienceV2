import cv2 as cap
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

        frame_width = int(cap.get(cap.CAP_PROP_FRFME_WIDTH))
        frame_height = int(cap.get(cap.CAP_PROP_FRFME_HEIGHT))
        # Copy the force data from the DataFrame
        force_data = df_aligned.copy()

        # Calculate maximum absolute values out of all forces
        max_z_force = max(force_data['FP1_Fz'].abs().max(), df_aligned['FP2_Fz'].abs().max())
        max_x_force = max(force_data['FP1_Fx'].abs().max(), df_aligned['FP2_Fx'].abs().max())
        max_y_force = max(force_data['FP1_Fy'].abs().max(), df_aligned['FP2_Fy'].abs().max())
        max_total_force = max(max_x_force, max_y_force, max_z_force)

        # Sets each force to a value between 0 and 1(1 being the maximum force) in order to apply scale factor
        force_data['FP1_Fz'] = (force_data['FP1_Fz']/max_total_force).round(4)
        force_data['FP1_Fy'] = (force_data['FP1_Fy']/max_total_force).round(4)
        force_data['FP2_Fz'] = (force_data['FP2_Fz']/max_total_force).round(4)
        force_data['FP2_Fy'] = (force_data['FP2_Fy']/max_total_force).round(4)
        force_data['FP1_Fx'] = (force_data['FP1_Fx']/max_total_force).round(4)
        force_data['FP2_Fx'] = (force_data['FP2_Fx']/max_total_force).round(4)

        # Determines and applies scale factor to force 0-1 values
        # Max height of a force is 0.8 frame height
        scale_factor = self.frame_height*self.force_scaling

        force_data['FP1_Fz'] *= scale_factor.astype(int)
        force_data['FP1_Fy'] *= scale_factor.astype(int)
        force_data['FP2_Fz'] *= scale_factor.astype(int)
        force_data['FP2_Fy'] *= scale_factor.astype(int)
        force_data['FP1_Fx'] *= scale_factor.astype(int)
        force_data['FP2_Fx'] *= scale_factor.astype(int)

        return force_data


    
