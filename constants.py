"""
File containing all the constants used in the different files
"""
import numpy as np

# Main Constants
CAMERA_RESOLUTION = (96, 54)
# Regions of interest
MAX_WIDTH = CAMERA_RESOLUTION[0]
MAX_HEIGHT = CAMERA_RESOLUTION[1]
ROI = [0, 0, MAX_WIDTH, MAX_HEIGHT]
# predict how many frames ahead
TIME_GAP = 25
TOTAL_FRAME = 5000
LOADLABELS_SEQUENCE = False
DATASET_SEQUENCE = False
# dataset sample frame interval
FRAME_INTERVAL = 12
# concatenate two images as input?
ONE_IMG_ONLY = False
# if take two images as input, what is the frame gap between two images
TWO_IMG_GAP = 6
# Training
FACTOR = 3  # Resize factor
INPUT_HEIGHT = ROI[3] // FACTOR
INPUT_WIDTH = ROI[2] // FACTOR
SPLIT_SEED = 42 # For train/val/test split
WEIGHTS_PTH = "cnn_model_25_tmp.pth"  # Path to the trained model
RES_DIR = "./Pre/results/"
