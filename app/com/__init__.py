"""
Center of Mass Calculator:
"""

__author__ = 'ryanh41137@gmail.com'

# Local modules
import util
import com_calculator
import pose_estimator

from loguru import logger

config = (util.get_config()).com_config

class ComCalculator:
    