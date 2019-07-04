"""
File containing all the constants used in the different files
"""
# Main Constants
CAMERA_RESOLUTION = (96, 54)
# Regions of interest
MAX_WIDTH = CAMERA_RESOLUTION[0]
MAX_HEIGHT = CAMERA_RESOLUTION[1]
ROI = [0, 0, MAX_WIDTH, MAX_HEIGHT]
# predict how many frames ahead

# dataset sample frame interval
FRAME_INTERVAL = 12

# Training
FACTOR = 1  # Resize factor
INPUT_HEIGHT = ROI[3] // FACTOR
INPUT_WIDTH = ROI[2] // FACTOR


WEIGHTS_PTH = "cnn_model_25_tmp.pth"  # Path to the trained model
RES_DIR = "./Pre/results/"


LEN_SEQ = 40
SEQ_PER_EPISODE_C = 9

RANDS = 13
