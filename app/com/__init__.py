"""
Center of Mass Calculator:
"""

__author__ = 'ryanh41137@gmail.com'

from loguru import logger
# Local modules
import util
import com_calculator
import pose_estimator

config = (util.get_config()).com_config

class ComCalculator:
    